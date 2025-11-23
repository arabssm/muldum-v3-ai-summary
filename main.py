import json
import os
import re
import uuid
import shutil
import tempfile
import subprocess
from urllib.parse import quote_plus, urljoin, urlparse, parse_qs
import requests
from typing import Dict, Any, List, Optional, Union
from urllib.parse import unquote

from bs4 import BeautifulSoup
import json as _json_unescape  # for decoding escaped strings in JSON blobs
import xml.etree.ElementTree as ET
import torchaudio
# pyannote 3.3.x AudioDecoder 이름 오류 방어용 + torchcodec 미설치 대비
try:
    from pyannote.audio.core.io import AudioDecoder  # type: ignore
except Exception:
    AudioDecoder = None  # type: ignore
try:
    import pyannote.audio.core.io as _pyannote_io  # type: ignore
except Exception:
    _pyannote_io = None
if _pyannote_io is not None and getattr(_pyannote_io, "AudioDecoder", None) is None:
    import torch

    class _FallbackAudioStreamMetadata:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.duration_seconds_from_header = num_frames / sample_rate if sample_rate else 0.0

    class _FallbackAudioSamples:
        def __init__(self, data: torch.Tensor, sample_rate: int):
            self.data = data
            self.sample_rate = sample_rate

    class _FallbackAudioDecoder:
        def __init__(self, source):
            self.source = source
            import soundfile as sf
            import numpy as np

            data, sr = sf.read(source, always_2d=True)
            # soundfile returns [frames, channels]; convert to [channel, time]
            data = np.asarray(data, dtype="float32").T
            waveform = torch.from_numpy(data)
            self._waveform = waveform
            self._sr = sr
            self.metadata = _FallbackAudioStreamMetadata(sr, waveform.shape[1], waveform.shape[0])

        def get_all_samples(self):
            return _FallbackAudioSamples(self._waveform, self._sr)

        def get_samples_played_in_range(self, start: float, end: float):
            start_frame = int(max(0, start * self._sr))
            end_frame = int(min(self._waveform.shape[1], end * self._sr))
            data = self._waveform[:, start_frame:end_frame]
            return _FallbackAudioSamples(data, self._sr)

    _pyannote_io.AudioDecoder = _FallbackAudioDecoder  # type: ignore
    _pyannote_io.AudioStreamMetadata = _FallbackAudioStreamMetadata  # type: ignore
    _pyannote_io.AudioSamples = _FallbackAudioSamples  # type: ignore
    AudioDecoder = _FallbackAudioDecoder  # type: ignore
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pyannote.audio import Pipeline
from pyannote.audio.core.model import Model
from faster_whisper import WhisperModel
from summa.summarizer import summarize

# =========================
# 환경 변수 / 전역 설정
# =========================

load_dotenv()

# 일부 배포 환경에서는 torchaudio에 list_audio_backends가 없어 speechbrain 초기화가 실패하므로 방어적으로 추가
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

# pyannote 내부에서 Model.from_pretrained에 'repo@rev' 문자열을 바로 넘기는 경우가 있어
# revision 인자를 자동 분리하도록 monkey patch
_orig_model_from_pretrained = Model.from_pretrained


def _patched_model_from_pretrained(checkpoint, *args, revision=None, **kwargs):
    if isinstance(checkpoint, str) and "@" in checkpoint and revision is None:
        base, rev = checkpoint.split("@", 1)
        return _orig_model_from_pretrained(base, *args, revision=rev, **kwargs)
    return _orig_model_from_pretrained(checkpoint, *args, revision=revision, **kwargs)


Model.from_pretrained = staticmethod(_patched_model_from_pretrained)  # type: ignore[assignment]

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HUGGINGFACE_TOKEN 환경 변수를 설정해주세요.")

def _parse_model_id_and_revision(model_id: str) -> tuple[str, Optional[str]]:
    """환경 변수에 'repo@rev' 형태가 들어오면 revision으로 분리."""
    if "@" in model_id:
        base, revision = model_id.split("@", 1)
        return base, revision
    return model_id, None

PYANNOTE_MODEL_ID_RAW = os.getenv(
    "PYANNOTE_MODEL_ID",
    "pyannote/speaker-diarization"  # 필요하면 다른 모델 ID로 바꿔도 됨
)
PYANNOTE_MODEL_ID, PYANNOTE_MODEL_REVISION = _parse_model_id_and_revision(PYANNOTE_MODEL_ID_RAW)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")  # tiny/small/medium/large 등
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 기본값을 v1 API용 최신 플래시로 설정 (환경 변수로 덮어쓰기 가능)
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
# 기본 API 버전 (권장 v1, 필요 시 v1beta)
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1")

# =========================
# 모델 로딩 (서버 시작 시 1회)
# =========================

print("Loading pyannote pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    PYANNOTE_MODEL_ID,
    token=HF_TOKEN,
    revision=PYANNOTE_MODEL_REVISION
)

print("Loading Whisper model...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda" if USE_GPU else "cpu",
    compute_type="int8"  # 속도/메모리 절약용
)

app = FastAPI(title="Meeting Summarizer API", version="0.1.0")

# CORS 설정
origins = [
    "http://localhost:3000",
    "https://v2.muldum.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Gemini 쇼핑 추천 프롬프트 구성
# =========================

GEMINI_SYSTEM_PROMPT = (
    "너는 DeviceMart/11번가 가격 비교를 돕는 쇼핑 어시스턴트다. "
    "무료배송 상품을 최우선으로, 같은 조건이라면 더 저렴한 상품을 추천한다. "
    "지정된 JSON 외의 말은 절대 하지 않는다."
)

GEMINI_RESPONSE_SCHEMA_EXAMPLE = {
    "summary": "거절 사유를 어떻게 개선했는지 1-2문장",
    "recommendations": [
        {
            "itemId": 123,
            "productName": "상품명",
            "source": "DeviceMart | 11번가",
            "price": 19000,
            "deliveryPrice": "무료배송 or 2000",
            "estimatedDelivery": "2025-02-03",
            "productUrl": "https://...",
            "imageUrl": "https://...",
            "reason": "가격/배송/거절사유 개선 포인트 한 줄"
        }
    ]
}


class BaseItem(BaseModel):
    id: int
    product_name: str = Field(..., alias="productName")
    price: int
    link: str
    # teamId가 숫자/문자 모두 들어오는 상황을 허용
    team_id: Optional[Union[str, int]] = Field(None, alias="teamId")
    reject_reason: Optional[str] = Field(None, alias="rejectReason")

    class Config:
        allow_population_by_field_name = True


class CandidateItem(BaseModel):
    item_id: int = Field(..., alias="itemId")
    product_name: str = Field(..., alias="productName")
    price: int
    delivery_price: Optional[str] = Field(None, alias="deliveryPrice")
    delivery_time: Optional[str] = Field(None, alias="deliveryTime")
    link: Optional[str] = None
    image_url: Optional[str] = Field(None, alias="imageUrl")
    recent_registered_at: Optional[str] = Field(None, alias="recentRegisteredAt")
    same_team: Optional[bool] = Field(None, alias="sameTeam")
    source: Optional[str] = Field(None, description="DeviceMart 또는 11번가")

    class Config:
        allow_population_by_field_name = True


class RecommendationRequest(BaseModel):
    base_item: BaseItem = Field(..., alias="baseItem")
    candidates: List[CandidateItem] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    """pydantic v1/v2 호환용 dict 추출"""
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude_none=True)  # type: ignore[attr-defined]
    return model.dict(by_alias=True, exclude_none=True)


def build_recommendation_prompt(
    base_item: BaseItem,
    candidates: List[CandidateItem]
) -> Dict[str, Any]:
    """
    Gemini에 전달할 system/user 프롬프트와 응답 스키마 예시를 구성한다.
    """
    # 기준 상품과 동일한 ID/이름은 후보에서 제외
    candidate_payload = []
    for c in candidates:
        if c.item_id == base_item.id:
            continue
        if c.product_name.strip() == base_item.product_name.strip():
            continue
        candidate_payload.append(_model_dump(c))

    reject_reason = base_item.reject_reason or "없음"
    base_link = base_item.link or "없음"
    has_candidates = len(candidate_payload) > 0

    if has_candidates:
        user_prompt = (
            "기준 상품:\n"
            f"- 이름: {base_item.product_name}\n"
            f"- 가격: {base_item.price}\n"
            f"- 링크: {base_link}\n"
            f"- 팀ID: {base_item.team_id or '미지정'}\n"
            f"- 거절 사유: {reject_reason}\n\n"
            "후보 목록(JSON 배열):\n"
            f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
            "요청:\n"
            "- 기준 상품과 동일한 이름/ID의 후보는 추천에서 제외.\n"
            "- 각 추천은 productUrl(후보의 link)과 imageUrl(후보의 imageUrl)을 반드시 포함. 없으면 해당 후보는 제외하거나 합리적인 값으로 채워.\n"
            "1) 무료배송이 아닌 후보는 제외하고 최대 3개만 추천. 단, 모두 유료배송이면 그 사실을 명시하고 최저가 순으로 3개 추천.\n"
            "2) 각 추천마다 (가격, 배송비, 추천 이유)을 한 줄 요약으로 설명.\n"
            f"3) 거절 사유({reject_reason})를 해결/개선하는 후보만 고른다.\n"
            "4) JSON으로만 답하고, 다음 스키마를 따른다."
        )
    else:
        user_prompt = (
            "기준 상품:\n"
            f"- 이름: {base_item.product_name}\n"
            f"- 가격: {base_item.price}\n"
            f"- 링크: {base_link}\n"
            f"- 팀ID: {base_item.team_id or '미지정'}\n"
            f"- 거절 사유: {reject_reason}\n\n"
            "후보 목록이 제공되지 않았습니다.\n\n"
            "요청:\n"
            "- 기준 상품과 동일한 이름/ID는 절대 추천하지 말 것.\n"
            "- productUrl, imageUrl을 합리적인 값(예: https://example.com/item/..., https://example.com/img/...)으로 채워서 최대 3개 제안.\n"
            "1) 기준 상품과 거절 사유를 참고하여 조건을 만족하는 대체 상품을 최대 3개 생성해서 추천.\n"
            "2) 무료배송을 우선, 없으면 유료배송이라도 최저가 순으로 제안.\n"
            "3) 가격/배송비/도착예상일/링크/이미지를 합리적 값으로 채워 넣되, 거절 사유를 해결하도록 선택.\n"
            "4) JSON으로만 답하고, 다음 스키마를 따른다."
        )

    return {
        "system_prompt": GEMINI_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response_schema_example": GEMINI_RESPONSE_SCHEMA_EXAMPLE,
    }


def call_gemini(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Gemini REST API 호출. 성공 시 원문과 파싱된 JSON(가능하면)을 반환한다.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY 환경 변수를 설정해주세요.")

    def _do_call(version: str):
        url = f"https://generativelanguage.googleapis.com/{version}/models/{GEMINI_MODEL_ID}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
            },
        }
        try:
            return requests.post(url, json=payload, timeout=60)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {e}")

    # 우선 설정된 버전으로 호출, 404면 v1<->v1beta 교차 재시도
    resp = _do_call(GEMINI_API_VERSION)
    if resp.status_code == 404 and GEMINI_API_VERSION == "v1beta":
        resp = _do_call("v1")
    elif resp.status_code == 404 and GEMINI_API_VERSION == "v1":
        resp = _do_call("v1beta")

    if resp.status_code != 200:
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Gemini 호출이 과도합니다. 잠시 후 다시 시도하세요.")
        raise HTTPException(status_code=resp.status_code, detail=f"Gemini 오류: {resp.text}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=502, detail="Gemini 응답 파싱 실패")

    parsed_json: Optional[Dict[str, Any]] = None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # ```json ... ``` 형태 제거
        cleaned_lines = cleaned.splitlines()
        if cleaned_lines and cleaned_lines[0].startswith("```"):
            cleaned_lines = cleaned_lines[1:]
        if cleaned_lines and cleaned_lines[-1].startswith("```"):
            cleaned_lines = cleaned_lines[:-1]
        cleaned = "\n".join(cleaned_lines).strip()
    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        parsed_json = None

    return {
        "raw": text,
        "parsed": parsed_json,
    }


# =========================
# 유틸 함수들
# =========================

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """업로드된 파일을 임시 경로에 저장하고 파일 경로 리턴"""
    suffix = os.path.splitext(upload_file.filename or "")[1]
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
    return tmp_path


def cut_audio_segment(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
    sample_rate: int = 16000
) -> None:
    """
    ffmpeg를 이용해 특정 구간만 잘라내기
    - start, end: 초 단위
    - mono, 16kHz로 리샘플링
    """
    cmd = [
        "ffmpeg",
        "-y",  # 덮어쓰기
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        output_path
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode('utf-8')}")


def summarize_text(text: str, ratio: float = 0.2, max_sentences: int = 5) -> str:
    """
    TextRank 기반 추출 요약.
    - 텍스트가 너무 짧으면 그냥 원문 리턴.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # 대충 길이 기준으로 요약 시도 여부 판단 (필요하면 더 고급 로직으로 바꿔도 됨)
    if len(text.split()) < 30:
        return text

    try:
        summary = summarize(text, ratio=ratio)
        if not summary:
            return text

        # 문장 수 제한
        sentences = [s.strip() for s in summary.split("\n") if s.strip()]
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        return " ".join(sentences)
    except Exception:
        # summa 내부 에러 시 그냥 원문 반환
        return text


def crawl_11st_by_category(category_no: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    최신 11번가 카테고리 페이지 크롤링 (2025 대응)
    data-log-body 안의 JSON을 파싱해 상품 정보를 추출한다.
    """
    url = f"https://search.11st.co.kr/Search.tmall?ctgrNo={category_no}"
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    results = []

    for tag in soup.find_all(attrs={"data-log-body": True}):
        raw = tag.get("data-log-body")
        data = _parse_json_attr(raw)
        if not isinstance(data, dict):
            continue

        product_id = data.get("content_no") or data.get("productNo")
        if not product_id:
            continue

        # redirect URL 추출
        product_url = None
        link_url = data.get("link_url")
        if link_url and "redirect=" in link_url:
            try:
                parsed = urlparse(link_url)
                qs = parse_qs(parsed.query)
                product_url = qs.get("redirect", [None])[0]
            except:
                pass

        if not product_url:
            product_url = f"https://www.11st.co.kr/products/{product_id}"

        name = (
                data.get("productName")
                or data.get("snippet_object", {}).get("name")
                or ""
        )
        if not name:
            continue

        price = (
                data.get("last_discount_price")
                or data.get("productPrice")
                or None
        )
        price = _clean_price(str(price)) if price else None

        img_url = data.get("productImageUrl") or data.get("imageUrl")

        delivery = data.get("snippet_object", {}).get("delivery_price")

        results.append({
            "productName": name,
            "price": price,
            "productUrl": product_url,
            "imageUrl": img_url,
            "source": "11번가",
            "deliveryPrice": delivery,
            "reason": "11번가 카테고리 data-log-body",
        })

        if len(results) >= limit:
            break

    return results


# =========================
# 메인 분석 로직
# =========================

def run_diarization(audio_path: str) -> List[Dict[str, Any]]:
    """
    pyannote로 화자 분리 수행.
    return 형식: [{"speaker": "SPEAKER_00", "start": 0.3, "end": 4.2}, ...]
    """
    diarization = diarization_pipeline(audio_path, AudioDecoder=AudioDecoder)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end),
        })

    return segments


def transcribe_segment(audio_segment_path: str, language: str = "ko") -> str:
    """
    Whisper로 한 세그먼트에 대해 STT 수행.
    """
    text = ""
    segments, _ = whisper_model.transcribe(
        audio_segment_path,
        language=language,
        beam_size=5,
        vad_filter=True
    )
    for seg in segments:
        text += seg.text + " "
    return text.strip()


def process_audio_file(audio_path: str) -> Dict[str, Any]:
    """
    전체 파이프라인:
    - diarization
    - 각 화자별로 오디오 자르기 + STT
    - 화자별 요약
    - 전체 요약
    """
    diarization_segments = run_diarization(audio_path)

    # 화자별 텍스트 모으기
    speaker_texts: Dict[str, str] = {}
    speaker_segments: Dict[str, List[Dict[str, float]]] = {}

    for seg in diarization_segments:
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]

        # 세그먼트 오디오 임시 파일로 자르기
        seg_tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"{uuid.uuid4().hex}.wav"
        )
        cut_audio_segment(audio_path, start, end, seg_tmp_path)

        # Whisper로 텍스트 변환
        try:
            text = transcribe_segment(seg_tmp_path, language="ko")
        finally:
            # 임시 세그먼트 오디오 삭제
            if os.path.exists(seg_tmp_path):
                os.remove(seg_tmp_path)

        if not text:
            continue

        speaker_texts.setdefault(speaker, "")
        speaker_texts[speaker] += " " + text

        speaker_segments.setdefault(speaker, [])
        speaker_segments[speaker].append({"start": start, "end": end})

    # 화자별 요약
    speaker_summaries = {
        speaker: summarize_text(text, ratio=0.2, max_sentences=5)
        for speaker, text in speaker_texts.items()
    }

    # 전체 요약
    full_text = " ".join(speaker_texts.values())
    meeting_summary = summarize_text(full_text, ratio=0.15, max_sentences=7)

    # 응답 구조로 정리
    speakers_result = []
    for speaker_id, full_text in speaker_texts.items():
        speakers_result.append({
            "id": speaker_id,
            "summary": speaker_summaries.get(speaker_id, ""),
            "full_text": full_text.strip(),
            "segments": speaker_segments.get(speaker_id, [])
        })

    result = {
        "speakers": speakers_result,
        "meeting_summary": meeting_summary,
    }
    return result


# =========================
# 간단한 쇼핑 크롤러 (DeviceMart, 11번가)
# =========================

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}


def _clean_price(text: str) -> Optional[int]:
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text


def _decode_json_string(value: str) -> str:
    """HTML 내 JSON 블롭에서 추출한 문자열의 \\uXXXX 등을 디코드."""
    try:
        return _json_unescape.loads(f'"{value}"')
    except Exception:
        return value


def _parse_json_attr(value: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(value)
    except Exception:
        try:
            return _json_unescape.loads(value)
        except Exception:
            return None


def _extract_11st_product_info_from_url(url: str) -> Dict[str, Optional[str]]:
    info = {"product_id": None, "category": None}
    parsed = urlparse(url)

    # 1) productId 추출
    m = re.search(r"/products/(\d+)", parsed.path)
    if m:
        info["product_id"] = m.group(1)

    # 2) URL 쿼리에서 카테고리 먼저 확인 (최우선)
    qs = parse_qs(parsed.query)
    if "trCtgrNo" in qs and qs["trCtgrNo"]:
        info["category"] = qs["trCtgrNo"][0]
        return info

    # 3) HTML fallback
    try:
        html = _fetch_html(url)
    except Exception:
        return info

    cat_patterns = [
        r'dispCtgrNo"\s*[:=]\s*"?(?P<num>\d+)',
        r'ctgrNo"\s*[:=]\s*"?(?P<num>\d+)',
        r'categoryNo"\s*[:=]\s*"?(?P<num>\d+)',
    ]

    for pat in cat_patterns:
        mm = re.search(pat, html)
        if mm:
            info["category"] = mm.group("num")
            break

    return info



def _fetch_11st_category_info(disp_ctgr_no: str) -> List[Dict[str, str]]:
    """
    11번가 카테고리 서비스로 하위 카테고리 정보를 조회.
    """
    url = f"http://api.11st.co.kr/rest/cateservice/category/{disp_ctgr_no}"
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    categories: List[Dict[str, str]] = []
    for cat in root.findall(".//{*}category"):
        categories.append(
            {
                "depth": (cat.findtext("depth") or "").strip(),
                "dispNm": (cat.findtext("dispNm") or "").strip(),
                "dispNo": (cat.findtext("dispNo") or "").strip(),
                "parentDispNo": (cat.findtext("parentDispNo") or "").strip(),
            }
        )
    return categories


def _is_relevant(name: str, query: str, *, min_matches: int = 2) -> bool:
    """
    간단한 토큰 기반 매칭: 검색어 토큰 일부가 상품명에 포함되면 관련성 있다고 판단.
    - min_matches: 이 수만큼 토큰이 포함되어야 함 (기본 2개). 토큰이 1개뿐이면 1개 매칭만 요구.
    """
    name_lower = name.lower()
    tokens = [t for t in re.split(r"[\s\-]+", query.lower()) if len(t) >= 2]
    if not tokens:
        return True
    match_count = sum(1 for tok in tokens if tok in name_lower)
    required = min(min_matches, len(tokens))
    return match_count >= required


def _dedupe_by_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        name = item.get("productName", "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def crawl_devicemart(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    search_url = f"https://www.devicemart.co.kr/goods/search?searchword={quote_plus(query)}"
    html = _fetch_html(search_url)
    soup = BeautifulSoup(html, "html.parser")

    def _collect(min_matches: int) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for card in soup.select("li"):
            link_tag = card.find("a", href=re.compile(r"/goods/view"))
            if not link_tag or not link_tag.get("href"):
                continue

            name = link_tag.get_text(" ", strip=True)
            if not name:
                continue

            if not _is_relevant(name, query, min_matches=min_matches):
                continue

            price_tag = card.find("strong", class_=re.compile("price")) or card.find("span", class_=re.compile("price"))
            price = _clean_price(price_tag.get_text(" ", strip=True)) if price_tag else None
            img_tag = card.find("img")
            img_url = urljoin("https://www.devicemart.co.kr", img_tag["src"]) if img_tag and img_tag.get("src") else None
            # 카테고리/플레이스홀더 이미지는 건너뛴다.
            if img_url and "/category/" in img_url:
                img_url = None

            collected.append(
                {
                    "productName": name,
                    "price": price,
                    "productUrl": urljoin("https://www.devicemart.co.kr", link_tag["href"]),
                    "imageUrl": img_url,
                    "source": "DeviceMart",
                    "deliveryPrice": None,
                    "estimatedDelivery": None,
                    "reason": "크롤링 결과(디바이스마트) 상위 노출",
                }
            )
            if len(collected) >= limit:
                break
        return collected

    results = _collect(min_matches=2)
    if len(results) < limit:
        results.extend(_collect(min_matches=1))
        results = _dedupe_by_name(results)
    return results[:limit]


def crawl_11st(query: str, limit: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
    query = unquote(query).strip()
    if query.startswith("http"):
        info = _extract_11st_product_info_from_url(query)
        category_no = info.get("category")
        if category_no:
            return crawl_11st_by_category(category_no, limit=limit)

    # 🔥 반드시 있어야 하는 부분 — 너 코드에는 없음
    search_url = f"https://search.11st.co.kr/Search.tmall?kwd={quote_plus(query)}"
    html = _fetch_html(search_url)
    soup = BeautifulSoup(html, "html.parser")


    def _collect_from_cards(min_matches: Optional[int]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for card in soup.select("div.c_listing li, ul.c_listing li, div.listing ul li"):
            link_tag = card.find("a", href=re.compile(r"11st\\.co\\.kr/products/"))
            if not link_tag or not link_tag.get("href"):
                continue

            name = link_tag.get_text(" ", strip=True)
            if not name:
                continue

            if min_matches is not None and not _is_relevant(name, query, min_matches=min_matches):
                continue

            price_tag = card.find("strong", class_=re.compile("price")) or card.find("span", class_=re.compile("price"))
            price = _clean_price(price_tag.get_text(" ", strip=True)) if price_tag else None
            if price is None:
                price = _find_price_near(card)

            img_tag = card.find("img")
            img_url = None
            if img_tag:
                img_url = img_tag.get("data-original") or img_tag.get("data-src") or img_tag.get("src")
            if img_url and img_url.startswith("//"):
                img_url = "https:" + img_url

            delivery_text = None
            delivery_tag = card.find(string=re.compile("무료"))
            if delivery_tag:
                delivery_text = "무료배송"

            collected.append(
                {
                    "productName": name,
                    "price": price,
                    "productUrl": link_tag["href"],
                    "imageUrl": img_url,
                    "source": "11번가",
                    "deliveryPrice": delivery_text,
                    "estimatedDelivery": None,
                    "reason": "크롤링 결과(11번가) 상위 노출",
                }
            )
            if len(collected) >= limit:
                break
        return collected

    def _collect_from_links(min_matches: Optional[int]) -> List[Dict[str, Any]]:
        """
        카드 셀렉터가 깨졌을 때 대비: 페이지 내 모든 product 링크를 훑으며 수집.
        """
        collected: List[Dict[str, Any]] = []
        seen_links = set()
        for link_tag in soup.find_all("a", href=re.compile(r"11st\\.co\\.kr/products/")):
            href = link_tag.get("href")
            if not href or href in seen_links:
                continue
            seen_links.add(href)

            name = link_tag.get_text(" ", strip=True)
            if not name:
                continue

            if min_matches is not None and not _is_relevant(name, query, min_matches=min_matches):
                continue

            parent = link_tag.find_parent(["li", "div"]) or link_tag
            price = _find_price_near(parent)

            img_url = None
            img_tag = parent.find("img")
            if img_tag:
                img_url = img_tag.get("data-original") or img_tag.get("data-src") or img_tag.get("src")
            if img_url and img_url.startswith("//"):
                img_url = "https:" + img_url

            collected.append(
                {
                    "productName": name,
                    "price": price,
                    "productUrl": href,
                    "imageUrl": img_url,
                    "source": "11번가",
                    "deliveryPrice": None,
                    "estimatedDelivery": None,
                    "reason": "크롤링 결과(11번가) 링크 파싱",
                }
            )
            if len(collected) >= limit:
                break
        return collected

    def _collect_from_json_blob() -> List[Dict[str, Any]]:
        """
        HTML 내 스크립트 JSON에서 product 정보를 정규식으로 추출.
        """
        collected: List[Dict[str, Any]] = []
        pattern = re.compile(
            r'"productname"\\s*:\\s*"(?P<name>[^"]+?)".*?"productid"\\s*:\\s*"?(?P<pid>\\d+)"?.*?"productprice"\\s*:\\s*"?(?P<price>\\d+)"?.*?"productimage"\\s*:\\s*"(?P<img>[^"]+?)"',
            re.DOTALL | re.IGNORECASE,
        )
        for m in pattern.finditer(html):
            name_raw = m.group("name")
            pid = m.group("pid")
            price_val = _clean_price(m.group("price"))
            img_url = m.group("img")
            name = _decode_json_string(name_raw)
            if not name:
                continue
            collected.append(
                {
                    "productName": name,
                    "price": price_val,
                    "productUrl": f"https://www.11st.co.kr/products/{pid}",
                    "imageUrl": img_url,
                    "source": "11번가",
                    "deliveryPrice": None,
                    "estimatedDelivery": None,
                    "reason": "크롤링 결과(11번가) JSON 파싱",
                }
            )
            if len(collected) >= limit:
                break
        return collected

    def _collect_from_data_log_body() -> List[Dict[str, Any]]:
        collected = []

        for tag in soup.find_all(attrs={"data-log-body": True}):
            raw = tag.get("data-log-body")
            data = _parse_json_attr(raw)
            if not isinstance(data, dict):
                continue

            product_id = data.get("content_no") or data.get("productNo")
            if not product_id:
                continue

            # 광고 redirect URL에서 진짜 상품URL 추출
            link_url = data.get("link_url")
            product_url = None
            if link_url and "redirect=" in link_url:
                try:
                    parsed = urlparse(link_url)
                    qs = parse_qs(parsed.query)
                    product_url = qs.get("redirect", [None])[0]
                except:
                    pass

            if not product_url:
                product_url = f"https://www.11st.co.kr/products/{product_id}"

            # 상품명
            name = data.get("productName") or data.get("snippet_object", {}).get("name") or ""
            if not name:
                continue

            # 가격
            price = data.get("last_discount_price") or data.get("productPrice")
            price = _clean_price(str(price)) if price else None

            # 이미지
            img_url = data.get("productImageUrl") or data.get("imageUrl")

            # 배송비
            delivery = data.get("snippet_object", {}).get("delivery_price")

            collected.append({
                "productName": name,
                "price": price,
                "productUrl": product_url,
                "imageUrl": img_url,
                "source": "11번가",
                "deliveryPrice": delivery,
                "reason": "11번가 data-log-body JSON",
            })

            if len(collected) >= limit:
                break

        return collected


def crawl_products(query: str, limit_total: int = 6, sources: Optional[List[str]] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    검색어로 각 쇼핑몰을 크롤링.
    - sources: ["devicemart", "11st"] 중 선택. None이면 둘 다.
    """
    normalized = [s.lower() for s in sources] if sources else ["devicemart", "11st"]
    items: List[Dict[str, Any]] = []

    crawlers: List = []
    if any(s in normalized for s in ("devicemart", "device", "dm")):
        crawlers.append(crawl_devicemart)
    if any(s in normalized for s in ("11st", "11번가", "eleven")):
        crawlers.append(crawl_11st)
    if not crawlers:
        crawlers = [crawl_devicemart, crawl_11st]

    per_site = max(1, limit_total // max(1, len(crawlers)))

    for crawler in crawlers:
        try:
            if crawler is crawl_11st:
                items.extend(crawler(query, limit=per_site, category=category))
            else:
                items.extend(crawler(query, limit=per_site))
        except Exception as e:
            # 크롤링 실패 시 다른 사이트라도 계속 시도
            print(f"[crawler] {crawler.__name__} failed: {e}")

    items = _dedupe_by_name(items)
    return items[:limit_total]


# =========================
# FastAPI 엔드포인트
# =========================


@app.post("/recommendations/prompt")
async def recommendation_prompt(payload: RecommendationRequest):
    """
    기준 상품 + 후보 목록을 받아 Gemini에 전달할 system/user 프롬프트와 응답 스키마 예시를 반환,
    동시에 Gemini를 호출해 결과를 함께 제공.
    """
    prompts = build_recommendation_prompt(payload.base_item, payload.candidates)
    try:
        gemini_result = call_gemini(prompts["system_prompt"], prompts["user_prompt"])
        return JSONResponse(
            content={
                **prompts,
                "gemini_raw": gemini_result["raw"],
                "gemini_parsed": gemini_result["parsed"],
            }
        )
    except HTTPException as e:
        # 429 등 LLM 호출 실패 시에도 프롬프트는 내려서 프런트가 직접 호출하거나 재시도 가능하도록 한다.
        if e.status_code == 429:
            return JSONResponse(
                status_code=200,
                content={
                    **prompts,
                    "gemini_raw": None,
                    "gemini_parsed": None,
                    "gemini_error": "Gemini 호출이 제한되었습니다. 프런트에서 직접 호출하거나 잠시 후 재시도하세요.",
                },
            )
        raise


@app.post("/analyze")
async def analyze_meeting_audio(file: UploadFile = File(...)):
    """
    회의 음성 파일 업로드 → 화자별 요약 + 전체 요약 반환
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")

    # 간단한 확장자 체크 (원하면 더 강화 가능)
    if not (file.filename.endswith(".wav") or file.filename.endswith(".mp3") or file.filename.endswith(".m4a")):
        raise HTTPException(status_code=400, detail="wav/mp3/m4a 형식만 지원합니다.")

    tmp_path = save_upload_file_tmp(file)

    try:
        result = process_audio_file(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 원본 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# pyannote 3.3.x에서 AudioDecoder 심볼 누락 시 diarization_pipeline 호출에 넘겨주기 위한 fallback
from fastapi import Request


@app.post("/debug/pyannote")
async def debug_pyannote(request: Request):
    """
    pyannote AudioDecoder가 import 가능한지 확인용 간단 엔드포인트
    """
    ok = AudioDecoder is not None
    return {"AudioDecoder_present": ok}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/recommendations/crawl")
async def crawl_recommendations(q: str, limit: int = 6, source: Optional[str] = None, category: Optional[str] = None):
    """
    Gemini 없이 간단히 검색어 기반 크롤링으로 상위 상품을 수집해 반환한다.
    - q: 검색어 (예: 기준 상품명)
    - limit: 최대 결과 수
    - source: "devicemart" 또는 "11st" 중 선택 (None이면 둘 다)
    - category: 11번가 카테고리 번호. q가 11번가 상품 URL이면 trCtgrNo를 자동 추출해 사용한다.
    """
    limit = max(1, min(10, limit))
    sources = [source] if source else None
    items = crawl_products(q, limit_total=limit, sources=sources, category=category)

    return JSONResponse(
        content={
            "query": q,
            "source": source or "all",
            "category": category,
            "count": len(items),
            "items": items,
            "note": "검색 결과가 없으면 빈 배열을 반환합니다.",
            "debug": {
                "devicemart_enabled": not source or source.lower() in ("devicemart", "device", "dm"),
                "eleven_enabled": not source or source.lower() in ("11st", "11번가", "eleven"),
            },
        },
        status_code=200,
    )
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Query
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
kw_model = KeyBERT(model)

def extract_main_keyword(text: str) -> str:
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=10
    )
    return keywords[0][0]

API_KEY = "ff49fbaa914833d531a36ada7b3c3ac0"



def search_11st_products(keyword: str, limit: int = 20):
    url = "http://openapi.11st.co.kr/openapi/OpenApiService.tmall"
    params = {
        "key": API_KEY,
        "apiCode": "ProductSearch",
        "keyword": keyword,
        "pageSize": limit,
        "sortCd": "CP"
    }

    xml_response = requests.get(url, params=params).text
    return parse_product_xml(xml_response)


def parse_product_xml(xml_data: str):
    root = ET.fromstring(xml_data)
    products = root.find("Products")

    if products is None:
        return []

    result = []
    for product in products.findall("Product"):
        def get(tag):
            e = product.find(tag)
            return e.text if e is not None else None

        result.append({
            "productCode": get("ProductCode"),
            "name": get("ProductName"),
            "price": get("ProductPrice"),
            "image": get("ProductImage300") or get("ProductImage"),
            "detailUrl": get("DetailPageUrl"),
            "seller": get("SellerNick"),
        })

    return result


@app.get("/recommend/11st")
def recommend_from_name(
    name: str = Query(..., description="상품명 그대로 입력"),
    limit: int = 20
):
    keyword = extract_main_keyword(name)
    print(" ⬇️ 추출된 핵심 키워드:", keyword)

    items = search_11st_products(keyword, limit)

    return {
        "query": name,
        "keyword": keyword,
        "count": len(items),
        "items": items
    }
