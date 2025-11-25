# muldum-v3-ai-summary

## POST /recommendations/prompt 명세
- 설명: 기준 상품과 후보 목록을 받아 Gemini에 전달할 system/user 프롬프트와 응답 스키마 예시를 반환합니다. 실제 Gemini 호출은 클라이언트가 수행합니다.
- URL: `/recommendations/prompt`
- Method: `POST`
- Content-Type: `application/json`

### 요청 바디
```json
{
  "baseItem": {
    "id": 1900158141440,
    "productName": "농가살리기 30년 전통 통영할매 원조 생굴무침 660g (330g*2통)",
    "price": 20900,
    "link": "https://devicemart.example/item/1900158141440",
    "teamId": "team-abc",
    "rejectReason": "너무 배송이 오래 걸림"
  },
  "candidates": [
    {
      "itemId": 111,
      "productName": "생굴무침 660g",
      "price": 19900,
      "deliveryPrice": "무료배송",
      "deliveryTime": "2024-02-10T00:00:00",
      "link": "https://11st.example/item/111",
      "imageUrl": "https://11st.example/item/111.jpg",
      "recentRegisteredAt": "2024-01-20",
      "sameTeam": false,
      "source": "11번가"
    }
  ]
}
```

필드 요약:
- `baseItem.productName`(string), `baseItem.price`(int), `baseItem.link`(string) 필수
- `rejectReason`(string) 있을 경우 추천 사유 개선 기준으로 사용
- `candidates[].deliveryPrice`: 숫자 또는 `"무료배송"`
- `candidates[].source`: `"DeviceMart"` 또는 `"11번가"`

### 응답 바디
```json
{
  "system_prompt": "…Gemini system prompt…",
  "user_prompt": "…기준/후보 포함된 사용자 프롬프트…",
  "response_schema_example": {
    "summary": "거절 사유 개선에 대한 1-2문장",
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
}
```

동작 요약:
- 무료배송 후보를 우선 반영해 최대 3개만 추천. 전부 유료배송이면 그 사실을 명시하고 최저가 순으로 3개 추천하도록 프롬프트가 구성됨.
- `rejectReason`를 개선하는 후보만 선택하도록 지시된 프롬프트를 반환합니다.

## GET /recommendations/crawl
- 설명: Gemini 없이 검색어를 기준으로 DeviceMart/11번가 검색 결과 상위 상품 몇 개를 크롤링해 반환합니다.
- URL: `/recommendations/crawl?q=<검색어>&limit=6&source=devicemart|11st`
- 의존성: `beautifulsoup4` (설치: `pip install beautifulsoup4`)
- 검색어 토큰(공백/하이픈 기준)이 상품명에 모두 포함되는 결과만 반환해 엉뚱한 추천을 줄였습니다. 11번가 이미지는 `data-original/data-src/src` 순으로 가져옵니다.
- `source`: `devicemart` 또는 `11st`로 제한 검색 (없으면 둘 다 조회)
 - `category`: 11번가 카테고리 번호를 넘기면 검색 시 같이 사용합니다. `q`가 11번가 상품 URL이면 `trCtgrNo`와 productId를 추출해 자동 적용하며, 필요 시 `http://api.11st.co.kr/rest/cateservice/category/<dispCtgrNo>`를 호출해 하위 카테고리 정보를 가져옵니다.

### 응답 예시
```json
{
  "query": "라즈베리파이",
  "count": 5,
  "items": [
    {
      "productName": "라즈베리파이 4 Model B",
      "price": 79000,
      "productUrl": "https://www.devicemart.co.kr/goods/view?no=...",
      "imageUrl": "https://www.devicemart.co.kr/data/item/....jpg",
      "source": "DeviceMart",
      "deliveryPrice": null,
      "estimatedDelivery": null,
      "reason": "크롤링 결과(디바이스마트) 상위 노출"
    }
  ]
}
```

주의: 각 쇼핑몰의 마크업 변경 시 파싱이 실패할 수 있으므로, 문제가 생기면 셀렉터를 조정하거나 크롤링 실패 로그를 확인하세요.
