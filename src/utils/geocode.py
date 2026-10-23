"""주소 → 좌표 변환(지오코딩). 실데이터 분석 경로 전용.

두 가지 자격증명 주입 방식을 모두 지원합니다.
  (A) 네트워크 시크릿(권장): 환경의 'Network secrets' 로 dapi.kakao.com 에
      Authorization: KakaoAK <키> 헤더를 자동 주입. 코드는 키를 몰라도 됨.
      → 이 방식은 해당 호스트의 네트워크 접근도 함께 허용해 줍니다.
  (B) 환경변수: KAKAO_REST_KEY 가 있으면 코드가 직접 헤더를 붙입니다.

VWorld(VWORLD_KEY)는 (B) 방식 폴백으로만 사용합니다.
키 발급: 카카오 https://developers.kakao.com/console/app  ·  VWorld https://www.vworld.kr/dev/v4api.do
"""
from __future__ import annotations

import os

import requests

KAKAO_URL = "https://dapi.kakao.com/v2/local/search/address.json"
VWORLD_URL = "https://api.vworld.kr/req/address"

# 에이전트 프록시가 TLS 를 가로채므로 해당 CA 를 신뢰해야 함(있으면)
_CA = "/root/.ccr/ca-bundle.crt"


def _verify():
    return _CA if os.path.exists(_CA) else True


def geocode(address: str) -> tuple[float, float]:
    """주소 문자열 → (lat, lon). 실패 시 ValueError. (x=경도, y=위도)"""
    # (A)/(B) 카카오: env 키가 있으면 헤더 직접 부착, 없으면 네트워크 시크릿이 주입
    key = os.getenv("KAKAO_REST_KEY")
    headers = {"Authorization": f"KakaoAK {key}"} if key else {}
    try:
        r = requests.get(KAKAO_URL, params={"query": address},
                         headers=headers, timeout=10, verify=_verify())
        if r.ok:
            docs = r.json().get("documents", [])
            if docs:
                return float(docs[0]["y"]), float(docs[0]["x"])
    except requests.RequestException:
        pass  # 네트워크 차단/인증 실패 → 폴백으로

    # (B) VWorld 폴백 (키 직접)
    vworld = os.getenv("VWORLD_KEY")
    if vworld:
        for addr_type in ("road", "parcel"):  # 도로명 → 지번 순
            r = requests.get(
                VWORLD_URL,
                params={
                    "service": "address", "request": "getcoord", "version": "2.0",
                    "crs": "epsg:4326", "type": addr_type, "address": address,
                    "key": vworld,
                },
                timeout=10, verify=_verify(),
            )
            if r.ok and r.json().get("response", {}).get("status") == "OK":
                pt = r.json()["response"]["result"]["point"]
                return float(pt["y"]), float(pt["x"])

    raise ValueError(
        "지오코딩 실패 — 네트워크 시크릿(dapi.kakao.com, Authorization: KakaoAK) 또는 "
        "KAKAO_REST_KEY/VWORLD_KEY 설정과 주소를 확인하세요."
    )
