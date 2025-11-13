import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import re
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    google = "google"
    aws = "aws"
    azure = "azure"
    whisper = "whisper"


class SpeakerSegment(BaseModel):
    speaker: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


class SummaryRequest(BaseModel):
    text: Optional[str] = None
    segments: Optional[List[SpeakerSegment]] = None


class SummaryResponse(BaseModel):
    summary: str


def extract_word_units(whisper_result: dict) -> List[dict]:
    """Flatten Whisper output into time-aligned word units."""
    words: List[dict] = []
    for segment in whisper_result.get("segments", []):
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                text = (word.get("word") or word.get("text") or "").strip()
                if not text:
                    continue
                start = float(word.get("start", segment_start))
                end = float(word.get("end", segment_end))
                words.append({"text": text, "start": start, "end": end})
        else:
            text = segment.get("text", "").strip()
            if text:
                words.append({"text": text, "start": segment_start, "end": segment_end})
    return words


def build_segments_from_intervals(words: List[dict], intervals: List[tuple]) -> List[SpeakerSegment]:
    segments: List[SpeakerSegment] = []
    for speaker, start, end in intervals:
        tokens = [
            w["text"]
            for w in words
            if not (w["end"] < start or w["start"] > end)
        ]
        text = " ".join(tokens).strip()
        if text:
            segments.append(SpeakerSegment(speaker=str(speaker), text=text, start=start, end=end))
    return segments


class TranscriptionResponse(BaseModel):
    provider: ProviderType
    segments: List[SpeakerSegment]
    summary: str


app = FastAPI()

allowed_origins_env = os.environ.get("CORS_ALLOW_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if allowed_origins_env
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def encode_text(text: str, max_length: int):
    """Tokenize text for KoBART without token_type_ids."""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=False,
    ).to(device)


def is_repetitive(text: str, min_tokens: int = 10, ratio_threshold: float = 0.3) -> bool:
    """Return True if the text mostly repeats the same tokens."""
    tokens = text.split()
    if len(tokens) < min_tokens:
        return False
    return (len(set(tokens)) / len(tokens)) < ratio_threshold


def collapse_repetitions(text: str, max_repeat: int = 1) -> str:
    """Limit consecutive duplicate tokens to reduce degenerate summaries."""
    tokens = re.split(r"(\s+)", text)
    result = []
    prev = None
    repeat = 0
    for token in tokens:
        if token.isspace():
            result.append(token)
            continue
        norm = token.strip()
        if not norm:
            result.append(token)
            continue
        if prev and norm == prev:
            repeat += 1
            if repeat < max_repeat:
                result.append(token)
        else:
            prev = norm
            repeat = 0
            result.append(token)
    return "".join(result)


def deduplicate_sentences(text: str, min_chars: int = 5) -> str:
    """Remove duplicate sentences while preserving order."""
    stripped = text.strip()
    if not stripped:
        return stripped

    parts = re.split(r"(?<=[.!?])\s+|\n+", stripped)
    seen = set()
    ordered: List[str] = []

    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if len(sentence) < min_chars:
            ordered.append(sentence)
            continue
        key = re.sub(r"\s+", " ", sentence).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(sentence)

    return " ".join(ordered).strip()


def clean_summary_text(text: str) -> str:
    """Apply token collapse and sentence deduplication."""
    return deduplicate_sentences(collapse_repetitions(text))


async def summarize_text(
    text: str,
    *,
    max_length: int = 220,
    min_length: int = 60,
    num_beams: int = 4,
    **generate_overrides,
):
    """Run KoBART summarization off the event loop with safer defaults."""
    clean_text = text.strip()
    if not clean_text:
        return ""

    clean_text = clean_summary_text(clean_text)

    tokens = clean_text.split()
    if len(tokens) < 5:
        return clean_text

    generate_params = dict(
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        no_repeat_ngram_size=4,
        repetition_penalty=1.4,
        length_penalty=1.0,
        early_stopping=True,
    )
    generate_params.update(generate_overrides)

    def _generate():
        with torch.no_grad():
            inputs = encode_text(clean_text, max_length=1024)
            summary_ids = model.generate(
                **inputs,
                **generate_params,
            ).cpu()
        return clean_summary_text(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    return await asyncio.to_thread(_generate)


def extract_input_text(payload: SummaryRequest) -> str:
    """Return a normalized transcript string from raw text or speaker segments."""
    if payload.text and payload.text.strip():
        return payload.text.strip()

    if payload.segments:
        joined_lines = []
        for segment in payload.segments:
            content = (segment.text or "").strip()
            if not content:
                continue
            speaker = (segment.speaker or "").strip() or "발화자"
            joined_lines.append(f"{speaker}: {content}")
        return "\n".join(joined_lines).strip()

    return ""


def segments_to_text(segments: List[SpeakerSegment]) -> str:
    return "\n".join(f"{seg.speaker}: {seg.text}" for seg in segments if seg.text).strip()


def diarize_with_pyannote(audio_path: str) -> List[tuple]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio 패키지가 설치되어 있지 않습니다.") from exc

    token = os.environ.get("PYANNOTE_AUTH_TOKEN")
    if not token:
        raise RuntimeError("PYANNOTE_AUTH_TOKEN 환경 변수를 설정하세요.")
    model_id = os.environ.get("PYANNOTE_MODEL_ID", "pyannote/speaker-diarization")
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    diarization = pipeline(audio_path)
    intervals = [
        (str(speaker), float(turn.start), float(turn.end))
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return intervals


def diarize_with_speechbrain(audio_path: str) -> List[tuple]:
    try:
        import numpy as np
        import torchaudio
        from speechbrain.pretrained import EncoderClassifier
        from spectralcluster import SpectralClusterer
    except ImportError as exc:
        raise RuntimeError("speechbrain, torchaudio, spectralcluster 패키지를 설치하세요.") from exc

    encoder_source = os.environ.get("SPEECHBRAIN_ENCODER", "speechbrain/spkrec-ecapa-voxceleb")
    encoder_savedir = os.environ.get(
        "SPEECHBRAIN_SAVEDIR",
        os.path.join(tempfile.gettempdir(), "speechbrain_spkrec"),
    )
    chunk_seconds = float(os.environ.get("SPEECHBRAIN_CHUNK_SECONDS", "1.5"))
    hop_ratio = float(os.environ.get("SPEECHBRAIN_HOP_RATIO", "0.5"))
    target_sr = int(os.environ.get("SPEECHBRAIN_SAMPLE_RATE", "16000"))
    try:
        max_speakers = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "6"))
    except ValueError:
        max_speakers = 6

    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source=encoder_source,
        savedir=encoder_savedir,
        run_opts={"device": run_device},
    )

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    signal = waveform.squeeze(0)

    chunk_size = max(target_sr // 2, int(chunk_seconds * sr))
    hop_size = max(1, int(chunk_size * hop_ratio))
    if hop_size >= chunk_size:
        hop_size = max(1, chunk_size // 2)

    embeddings = []
    windows = []
    total_len = signal.shape[0]
    if total_len <= chunk_size:
        padded = F.pad(signal, (0, chunk_size - total_len))
        chunk = padded.unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(chunk.to(run_device))
        embeddings.append(emb.squeeze(0).cpu().numpy())
        windows.append((0.0, chunk_size / sr))
    else:
        for start in range(0, total_len - chunk_size + 1, hop_size):
            end = start + chunk_size
            chunk = signal[start:end]
            chunk = chunk.unsqueeze(0)
            with torch.no_grad():
                emb = classifier.encode_batch(chunk.to(run_device))
            embeddings.append(emb.squeeze(0).cpu().numpy())
            windows.append((start / sr, end / sr))
        remainder = total_len % hop_size
        if remainder and (total_len - chunk_size) > 0:
            start = total_len - chunk_size
            chunk = signal[start:]
            if chunk.shape[0] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))
            chunk = chunk.unsqueeze(0)
            with torch.no_grad():
                emb = classifier.encode_batch(chunk.to(run_device))
            embeddings.append(emb.squeeze(0).cpu().numpy())
            windows.append((start / sr, total_len / sr))

    if not embeddings:
        return []

    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] == 1:
        labels = np.array([0])
    else:
        try:
            clusterer = SpectralClusterer(
                min_clusters=1,
                max_clusters=max(1, max_speakers),
                p_percentile=0.90,
                gaussian_blur_sigma=1,
            )
        except TypeError:
            clusterer = SpectralClusterer(
                min_clusters=1,
                max_clusters=max(1, max_speakers),
            )
        labels = clusterer.predict(embeddings)

    intervals: List[tuple] = []
    current_label = None
    current_start = None
    current_end = None
    for label, window in zip(labels, windows):
        start, end = window
        if current_label is None:
            current_label = label
            current_start = start
            current_end = end
            continue
        if label != current_label:
            intervals.append((f"화자{int(current_label) + 1}", current_start, current_end))
            current_label = label
            current_start = start
            current_end = end
        else:
            current_end = end

    if current_label is not None and current_start is not None and current_end is not None:
        intervals.append((f"화자{int(current_label) + 1}", current_start, current_end))

    return intervals


def diarize_with_resemblyzer(audio_path: str, max_speakers: int) -> List[tuple]:
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
        from resemblyzer.hparams import sampling_rate
        from spectralcluster import SpectralClusterer
    except ImportError as exc:
        raise RuntimeError("resemblyzer 또는 spectralcluster 패키지가 없습니다.") from exc

    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    _, partial_embeddings, wav_slices = encoder.embed_utterance(
        wav,
        return_partials=True,
        rate=16,
    )

    if len(partial_embeddings) == 0:
        return []

    partial_embeddings = np.asarray(partial_embeddings)
    if partial_embeddings.shape[0] == 1:
        labels = np.array([0])
    else:
        try:
            clusterer = SpectralClusterer(
                min_clusters=1,
                max_clusters=max(1, max_speakers),
                p_percentile=0.90,
                gaussian_blur_sigma=1,
            )
        except TypeError:
            clusterer = SpectralClusterer(
                min_clusters=1,
                max_clusters=max(1, max_speakers),
            )
        labels = clusterer.predict(partial_embeddings)

    intervals: List[tuple] = []
    current_label = None
    current_start = 0.0
    current_end = 0.0

    for label, slice_window in zip(labels, wav_slices):
        start_idx = getattr(slice_window, "start", 0)
        end_idx = getattr(slice_window, "stop", start_idx)
        start_time = start_idx / sampling_rate
        end_time = end_idx / sampling_rate
        if current_label is None:
            current_label = label
            current_start = start_time
            current_end = end_time
            continue
        if label != current_label:
            intervals.append((f"화자{int(current_label) + 1}", current_start, current_end))
            current_label = label
            current_start = start_time
            current_end = end_time
        else:
            current_end = end_time

    if current_label is not None:
        intervals.append((f"화자{int(current_label) + 1}", current_start, current_end))

    return intervals


async def summarize_segments(segments: List[SpeakerSegment]) -> str:
    transcript_text = collapse_repetitions(segments_to_text(segments))
    summary = await summarize_text(transcript_text)
    if is_repetitive(summary):
        summary = await summarize_text(
            transcript_text,
            num_beams=1,
            no_repeat_ngram_size=6,
            repetition_penalty=1.8,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
    if is_repetitive(summary):
        summary = clean_summary_text(transcript_text[:500])
    return summary


def guess_media_format(path: str) -> str:
    ext = Path(path).suffix.lstrip(".").lower()
    return ext or "wav"


async def transcribe_audio(provider: ProviderType, audio_bytes: bytes, audio_path: str) -> List[SpeakerSegment]:
    if provider == ProviderType.google:
        return await asyncio.to_thread(transcribe_with_google, audio_bytes)
    if provider == ProviderType.aws:
        media_format = guess_media_format(audio_path)
        return await asyncio.to_thread(transcribe_with_aws, audio_path, media_format)
    if provider == ProviderType.azure:
        return await asyncio.to_thread(transcribe_with_azure, audio_path)
    if provider == ProviderType.whisper:
        return await asyncio.to_thread(transcribe_with_whisper_resemblyzer, audio_path)
    raise RuntimeError(f"Unsupported provider: {provider}")


def transcribe_with_google(audio_bytes: bytes, language_code: str = "ko-KR") -> List[SpeakerSegment]:
    try:
        from google.cloud import speech
    except ImportError as exc:
        raise RuntimeError("google-cloud-speech 패키지가 설치되어 있지 않습니다.") from exc

    client = speech.SpeechClient()
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=6,
    )
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        language_code=language_code,
        model="latest_long",
        enable_word_time_offsets=True,
        diarization_config=diarization_config,
    )
    audio = speech.RecognitionAudio(content=audio_bytes)
    response = client.recognize(config=config, audio=audio)
    if not response.results:
        return []

    words = response.results[-1].alternatives[0].words
    segments: List[SpeakerSegment] = []
    current_tag = None
    current_words: List[str] = []
    start_time = 0.0
    end_time = 0.0

    for word in words:
        tag = word.speaker_tag or 0
        word_text = word.word
        if current_tag is None:
            current_tag = tag
            start_time = word.start_time.total_seconds()
        elif tag != current_tag:
            text = " ".join(current_words).strip()
            if text:
                segments.append(
                    SpeakerSegment(
                        speaker=f"화자{current_tag}",
                        text=text,
                        start=start_time,
                        end=word.start_time.total_seconds(),
                    )
                )
            current_words = []
            current_tag = tag
            start_time = word.start_time.total_seconds()
        current_words.append(word_text)
        end_time = word.end_time.total_seconds()

    if current_words:
        segments.append(
            SpeakerSegment(
                speaker=f"화자{current_tag}",
                text=" ".join(current_words).strip(),
                start=start_time,
                end=end_time,
            )
        )

    return segments


def transcribe_with_aws(audio_path: str, media_format: str, language_code: str = "ko-KR") -> List[SpeakerSegment]:
    try:
        import boto3
        import requests
    except ImportError as exc:
        raise RuntimeError("boto3 또는 requests 패키지가 설치되어 있지 않습니다.") from exc

    bucket = os.environ.get("AWS_TRANSCRIBE_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_TRANSCRIBE_BUCKET 환경 변수를 설정하세요.")

    s3 = boto3.client("s3")
    key = f"transcribe/{uuid.uuid4().hex}{Path(audio_path).suffix}"
    s3.upload_file(audio_path, bucket, key)

    transcribe = boto3.client("transcribe")
    job_name = f"job-{uuid.uuid4().hex}"
    media_uri = f"s3://{bucket}/{key}"

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat=media_format,
        LanguageCode=language_code,
        Settings={
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": 6,
        },
    )

    while True:
        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(5)

    if status == "FAILED":
        raise RuntimeError(job.get("FailureReason", "AWS Transcribe 실패"))

    transcript_uri = job["Transcript"]["TranscriptFileUri"]
    response = requests.get(transcript_uri, timeout=30)
    response.raise_for_status()
    payload = response.json()

    speaker_segments = payload.get("results", {}).get("speaker_labels", {}).get("segments", [])
    items = payload.get("results", {}).get("items", [])

    speaker_map = {}
    for segment in speaker_segments:
        speaker = segment.get("speaker_label", "화자?")
        for item in segment.get("items", []):
            start_time = item.get("start_time")
            if start_time:
                speaker_map[start_time] = speaker

    segments: List[SpeakerSegment] = []
    current_speaker = None
    current_words: List[str] = []
    start_time = None
    end_time = None

    for item in items:
        if item.get("type") != "pronunciation":
            continue
        start = item.get("start_time")
        end = item.get("end_time")
        word = item.get("alternatives", [{}])[0].get("content", "")
        speaker = speaker_map.get(start, "화자?")

        if current_speaker is None:
            current_speaker = speaker
            start_time = float(start)
        elif speaker != current_speaker:
            text = " ".join(current_words).strip()
            if text:
                segments.append(
                    SpeakerSegment(
                        speaker=current_speaker,
                        text=text,
                        start=start_time,
                        end=float(start),
                    )
                )
            current_words = []
            current_speaker = speaker
            start_time = float(start)

        current_words.append(word)
        end_time = float(end)

    if current_words:
        segments.append(
            SpeakerSegment(
                speaker=current_speaker or "화자?",
                text=" ".join(current_words).strip(),
                start=start_time,
                end=end_time,
            )
        )

    return segments


def transcribe_with_azure(audio_path: str, language_code: str = "ko-KR") -> List[SpeakerSegment]:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise RuntimeError("azure-cognitiveservices-speech 패키지가 없습니다.") from exc

    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not speech_key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION 환경 변수를 설정하세요.")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_recognition_language = language_code
    speech_config.set_service_property(
        name="diarizationEnabled",
        value="true",
        channel=speechsdk.ServicePropertyChannel.UriQueryParameter,
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_RequestWordLevelTimestamps, "true"
    )

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    segments: List[SpeakerSegment] = []
    done = threading.Event()

    def handle_transcribed(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            speaker_id = evt.result.speaker_id or f"화자{evt.result.channel_id}"
            text = evt.result.text.strip()
            if text:
                segments.append(SpeakerSegment(speaker=speaker_id, text=text))

    def stop_handler(_):
        done.set()

    transcriber.transcribed.connect(handle_transcribed)
    transcriber.canceled.connect(stop_handler)
    transcriber.session_stopped.connect(stop_handler)

    transcriber.start_transcribing_async().get()
    done.wait()
    transcriber.stop_transcribing_async().get()

    return segments


def transcribe_with_whisper_resemblyzer(audio_path: str) -> List[SpeakerSegment]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("whisper 패키지가 설치되어 있지 않습니다.") from exc

    whisper_model_name = os.environ.get("WHISPER_MODEL_NAME", "small")
    try:
        max_speakers = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "6"))
    except ValueError:
        max_speakers = 6

    whisper_model = whisper.load_model(whisper_model_name)
    whisper_result = whisper_model.transcribe(
        audio_path,
        language="ko",
        word_timestamps=True,
        verbose=False,
    )

    words = extract_word_units(whisper_result)
    segments: List[SpeakerSegment] = []
    intervals: List[tuple] = []

    try:
        intervals = diarize_with_speechbrain(audio_path)
    except Exception as exc:
        logger.warning("SpeechBrain diarization 실패: %s", exc)

    if not intervals and os.environ.get("PYANNOTE_AUTH_TOKEN"):
        try:
            intervals = diarize_with_pyannote(audio_path)
        except Exception as exc:
            logger.warning("pyannote diarization 실패: %s", exc)

    if not intervals:
        try:
            intervals = diarize_with_resemblyzer(audio_path, max_speakers)
        except Exception as exc:
            logger.warning("Resemblyzer diarization 실패: %s", exc)

    if intervals:
        segments = build_segments_from_intervals(words, intervals)

    if not segments:
        summary_text = whisper_result.get("text", "").strip()
        if summary_text:
            segments.append(SpeakerSegment(speaker="전체", text=summary_text))

    return segments


@app.post("/summarize", response_model=SummaryResponse)
async def summarize_endpoint(payload: SummaryRequest):
    clean_full_text = clean_summary_text(extract_input_text(payload))
    if not clean_full_text:
        raise HTTPException(status_code=400, detail="요약할 텍스트를 입력해주세요.")

    summary = await summarize_text(clean_full_text)
    if is_repetitive(summary):
        summary = await summarize_text(
            clean_full_text,
            num_beams=1,
            no_repeat_ngram_size=6,
            repetition_penalty=1.8,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
    if is_repetitive(summary):
        summary = clean_summary_text(clean_full_text[:500])

    return SummaryResponse(summary=summary)


@app.post("/transcribe-and-summarize", response_model=TranscriptionResponse)
async def transcribe_and_summarize(provider: ProviderType = Form(...), audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments = await transcribe_audio(provider, audio_bytes, tmp_path)
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"{provider.value} 음성 인식 실패: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("임시 파일 삭제 실패: %s", tmp_path)

    if not segments:
        raise HTTPException(status_code=500, detail="음성을 텍스트로 변환하지 못했습니다.")

    summary = await summarize_segments(segments)
    return TranscriptionResponse(provider=provider, segments=segments, summary=summary)
