"""MVP 전역 설정.

실무 튜닝 포인트는 대부분 여기 모아 둡니다 — 반경, 추천 개수, 피처 가중치.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv 미설치 시에도 동작
    pass

# --- 경로 ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- 분석 파라미터 ------------------------------------------------------
# 입지 특성을 뽑을 반경(미터). 도보 상권 기준 300~500m 가 일반적.
ANALYSIS_RADIUS_M = 500

# 유사입지 추천 시 참고할 이웃 입지 수(KNN 의 k)
DEFAULT_K_NEIGHBORS = 10

# 최종 추천 업종 개수
DEFAULT_TOP_N = 10

# --- API 키 (없으면 None → 수집기는 synthetic 로 폴백) ------------------
SANGGA_SERVICE_KEY = os.getenv("SANGGA_SERVICE_KEY") or None
LOCALDATA_SERVICE_KEY = os.getenv("LOCALDATA_SERVICE_KEY") or None
SGIS_CONSUMER_KEY = os.getenv("SGIS_CONSUMER_KEY") or None
SGIS_CONSUMER_SECRET = os.getenv("SGIS_CONSUMER_SECRET") or None

# --- 공공데이터 API 베이스 URL ------------------------------------------
SANGGA_BASE = "http://apis.data.go.kr/B553077/api/open/sdsc2"
LOCALDATA_BASE = "http://www.localdata.go.kr/platform/rest/GR0/openDataApi"
SGIS_BASE = "https://sgisapi.kostat.go.kr/OpenAPI3"
