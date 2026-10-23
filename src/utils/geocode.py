"""주소 → 좌표 변환(지오코딩). 실데이터 분석 경로 전용.

브라우저 데모는 합성 좌표라 주소검색이 무의미하므로, 지오코딩은 파이썬(실데이터)
경로에만 둡니다. 카카오 로컬 API(권장) 또는 국토부 VWorld 를 사용합니다.

키 발급:
  - 카카오: https://developers.kakao.com  (REST API 키) → KAKAO_REST_KEY
  - VWorld: https://www.vworld.kr/dev/v4api.do              → VWORLD_KEY
"""
from __future__ import annotations

import os

import requests


def geocode(address: str) -> tuple[float, float]:
    """주소 문자열 → (lat, lon). 실패 시 ValueError."""
    kakao = os.getenv("KAKAO_REST_KEY")
    if kakao:
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            params={"query": address},
            headers={"Authorization": f"KakaoAK {kakao}"},
            timeout=10,
        )
        r.raise_for_status()
        docs = r.json().get("documents", [])
        if docs:  # x=경도, y=위도
            return float(docs[0]["y"]), float(docs[0]["x"])

    vworld = os.getenv("VWORLD_KEY")
    if vworld:
        for addr_type in ("road", "parcel"):  # 도로명 → 지번 순으로 시도
            r = requests.get(
                "https://api.vworld.kr/req/address",
                params={
                    "service": "address", "request": "getcoord", "version": "2.0",
                    "crs": "epsg:4326", "type": addr_type, "address": address,
                    "key": vworld,
                },
                timeout=10,
            )
            r.raise_for_status()
            res = r.json().get("response", {})
            if res.get("status") == "OK":
                pt = res["result"]["point"]
                return float(pt["y"]), float(pt["x"])

    raise ValueError(
        "지오코딩 실패 — KAKAO_REST_KEY 또는 VWORLD_KEY(.env)와 주소를 확인하세요."
    )
