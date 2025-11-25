from typing import Any, Dict, List, Optional

import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

app = FastAPI(title="11st Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 미리 SentenceTransformer 임베딩 모델을 로드해 KeyBERT가 재사용하도록 한다.
_sentence_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
_kw_model = KeyBERT(_sentence_model)

API_KEY = "ff49fbaa914833d531a36ada7b3c3ac0"


def extract_main_keyword(text: str) -> str:
    keywords = _kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=7,
    )
    return keywords[0][0] if keywords else text


def search_11st_products(keyword: str, limit: int = 20) -> List[Dict[str, Optional[str]]]:
    url = "http://openapi.11st.co.kr/openapi/OpenApiService.tmall"
    params = {
        "key": API_KEY,
        "apiCode": "ProductSearch",
        "keyword": keyword,
        "pageSize": limit,
        "sortCd": "CP",
    }

    xml_response = requests.get(url, params=params, timeout=10).text
    return parse_product_xml(xml_response)


def parse_product_xml(xml_data: str) -> List[Dict[str, Optional[str]]]:
    root = ET.fromstring(xml_data)
    products = root.find("Products")

    if products is None:
        return []

    result = []
    for product in products.findall("Product"):
        def get(tag: str) -> Optional[str]:
            node = product.find(tag)
            return node.text if node is not None else None

        result.append(
            {
                "productCode": get("ProductCode"),
                "name": get("ProductName"),
                "price": get("ProductPrice"),
                "image": get("ProductImage300") or get("ProductImage"),
                "detailUrl": get("DetailPageUrl"),
                "seller": get("SellerNick"),
            }
        )

    return result


@app.get("/recommend/11st")
def recommend_from_name(
    name: str = Query(..., description="상품명 그대로 입력"),
    limit: int = Query(20, ge=1, le=40),
) -> Dict[str, Any]:
    keyword = extract_main_keyword(name)
    items = search_11st_products(keyword, limit)

    return {
        "query": name,
        "keyword": keyword,
        "count": len(items),
        "items": items,
    }
