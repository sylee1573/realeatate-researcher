"""실데이터 수집기 — 공공데이터 API 연동부.

현재는 '실 API 호출 → 미구현 시 synthetic 폴백' 구조입니다.
키를 .env 에 넣고 use_synthetic=False 로 호출하면 TODO 부분만 채우면 됩니다.
하류 모듈이 기대하는 컬럼 스키마는 synthetic.py 와 동일하게 유지하세요.
"""
from __future__ import annotations

import pandas as pd
import requests

import config
from . import synthetic


# --- 1) 상가 POI --------------------------------------------------------
def load_stores(center_lat, center_lon, radius_m, use_synthetic=None) -> pd.DataFrame:
    """소상공인 상가(상권)정보: 반경 내 상가 목록.

    실 엔드포인트 예: {SANGGA_BASE}/storeListInRadius
      params: serviceKey, radius(m), cx(경도), cy(위도), type=json, pageNo, numOfRows
    반환 스키마는 synthetic.make_stores 와 동일하게 매핑하세요.
    """
    if use_synthetic is None:
        use_synthetic = config.SANGGA_SERVICE_KEY is None
    if use_synthetic:
        return synthetic.make_stores()

    # TODO: 실제 호출 — 페이지네이션 돌며 전부 수집 후 DataFrame 으로 정규화
    url = f"{config.SANGGA_BASE}/storeListInRadius"
    params = {
        "serviceKey": config.SANGGA_SERVICE_KEY,
        "radius": int(radius_m),
        "cx": center_lon,
        "cy": center_lat,
        "type": "json",
        "numOfRows": 1000,
        "pageNo": 1,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("body", {}).get("items", [])
    # 공공 API 컬럼명 → 내부 스키마 매핑
    df = pd.DataFrame([{
        "store_id": it.get("bizesId"),
        "biz_lcls": it.get("indsLclsNm"),
        "biz_mcls": it.get("indsMclsNm"),
        "lat": float(it["lat"]),
        "lon": float(it["lon"]),
        "open_date": pd.NaT,   # 상가정보엔 없음 → load_permits 에서 보강
        "close_date": pd.NaT,
    } for it in items])
    return df


# --- 2) 인허가(개·폐업 → 생존) -----------------------------------------
def load_permits(region_code=None, use_synthetic=None) -> pd.DataFrame:
    """지방행정 인허가데이터: 업종별 영업개시/폐업일.

    상가 POI 와 사업자 단위로 join 하여 open_date/close_date 를 채우는 데 사용.
    데모에선 make_stores 가 이미 open/close 를 포함하므로 별도 불필요.
    """
    if use_synthetic is None:
        use_synthetic = config.LOCALDATA_SERVICE_KEY is None
    if use_synthetic:
        return pd.DataFrame()  # 데모는 stores 에 포함됨
    # TODO: LOCALDATA openDataApi 호출 및 정규화
    raise NotImplementedError("localdata 연동은 키 발급 후 구현")


# --- 3) 인구/소득 -------------------------------------------------------
def load_population(use_synthetic=None) -> pd.DataFrame:
    """통계청 SGIS: 집계구 인구/가구/소득.

    실 연동: OAuth 로 accessToken 발급 → 인구/가구 통계 API 호출.
    """
    if use_synthetic is None:
        use_synthetic = config.SGIS_CONSUMER_KEY is None
    if use_synthetic:
        return synthetic.make_population()
    # TODO: SGIS accessToken 발급 후 집계구 통계 호출
    raise NotImplementedError("SGIS 연동은 키 발급 후 구현")


# --- 4) 교통 ------------------------------------------------------------
def load_transit(use_synthetic=True) -> pd.DataFrame:
    """지하철 승하차/교통량 결절점. (공개 CSV 또는 열린데이터광장 API)"""
    if use_synthetic:
        return synthetic.make_transit()
    # TODO: 교통공사/열린데이터광장 승하차 데이터 로드
    raise NotImplementedError("교통 데이터 연동은 소스 확정 후 구현")
