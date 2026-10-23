"""합성 데이터 생성기 — API 키 없이 파이프라인 전체를 검증하기 위한 용도.

실제 데이터와 '컬럼 스키마'를 동일하게 맞춰 둡니다.
따라서 나중에 실제 수집기로 바꿔도 하류(피처/모델/추천)는 수정 불필요합니다.

※ 모든 상호명은 익명 토큰("S00001")입니다. 실존 업체명·실주소를 넣지 마세요.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 업종 분류 (소상공인 상가정보의 대/중분류를 단순화한 예시)
BIZ_TAXONOMY = {
    "음식": ["한식", "분식", "카페", "치킨", "주점", "양식"],
    "소매": ["편의점", "의류", "화장품", "식료품", "문구"],
    "서비스": ["미용", "학원", "병원", "세탁", "부동산중개"],
    "여가": ["피트니스", "노래방", "스터디카페"],
}
_FLAT = [(lc, mc) for lc, mcs in BIZ_TAXONOMY.items() for mc in mcs]

# 데모용 중심좌표 (서울시청 부근). 실분석 시엔 대상 상가 좌표를 쓰세요.
SEOUL_CENTER = (37.5665, 126.9780)


def make_stores(n=2000, n_clusters=12, seed=42) -> pd.DataFrame:
    """상가 POI. 실제: 소상공인 상가(상권)정보 API 와 동일 스키마.

    컬럼: store_id, biz_lcls, biz_mcls, lat, lon, open_date, close_date
    (open/close 는 실제로는 localdata 인허가에서 join; 데모는 함께 생성)
    """
    rng = np.random.default_rng(seed)
    clat, clon = SEOUL_CENTER
    # 상권 클러스터 중심 (유동이 몰리는 '목' 모사)
    centers = np.column_stack([
        clat + rng.normal(0, 0.03, n_clusters),
        clon + rng.normal(0, 0.03, n_clusters),
    ])
    which = rng.integers(0, n_clusters, n)
    lat = centers[which, 0] + rng.normal(0, 0.004, n)
    lon = centers[which, 1] + rng.normal(0, 0.004, n)

    biz_idx = rng.integers(0, len(_FLAT), n)
    lcls = [_FLAT[i][0] for i in biz_idx]
    mcls = [_FLAT[i][1] for i in biz_idx]

    # 영업개시일 / 폐업일(생존기간의 프록시). 폐업 없으면 NaT.
    start = pd.Timestamp("2015-01-01")
    open_offset = rng.integers(0, 365 * 9, n)
    open_date = start + pd.to_timedelta(open_offset, unit="D")
    survive_days = rng.gamma(shape=2.0, scale=500, size=n).astype(int)
    closed = rng.random(n) < 0.35  # 35% 는 폐업한 과거 점포
    close_date = pd.Series(pd.NaT, index=range(n))
    close_date[closed] = (open_date[closed] + pd.to_timedelta(survive_days[closed], unit="D"))

    return pd.DataFrame({
        "store_id": [f"S{i:05d}" for i in range(n)],
        "biz_lcls": lcls,
        "biz_mcls": mcls,
        "lat": lat,
        "lon": lon,
        "open_date": open_date,
        "close_date": close_date.values,
    })


def make_population(grid=40, seed=7) -> pd.DataFrame:
    """집계구 단위 인구(격자로 근사). 실제: 통계청 SGIS.

    컬럼: lat, lon, pop_total, pop_young_ratio, pop_senior_ratio, income_idx
    """
    rng = np.random.default_rng(seed)
    clat, clon = SEOUL_CENTER
    lats = np.linspace(clat - 0.06, clat + 0.06, grid)
    lons = np.linspace(clon - 0.06, clon + 0.06, grid)
    mesh = [(a, b) for a in lats for b in lons]
    la = np.array([m[0] for m in mesh])
    lo = np.array([m[1] for m in mesh])
    # 중심에 가까울수록 인구 집중
    d = np.sqrt((la - clat) ** 2 + (lo - clon) ** 2)
    pop = (8000 * np.exp(-d / 0.03) + rng.normal(0, 400, len(mesh))).clip(50)
    return pd.DataFrame({
        "lat": la,
        "lon": lo,
        "pop_total": pop.astype(int),
        "pop_young_ratio": rng.uniform(0.15, 0.45, len(mesh)),
        "pop_senior_ratio": rng.uniform(0.10, 0.35, len(mesh)),
        "income_idx": rng.uniform(70, 130, len(mesh)),  # 100=전국평균 가정
    })


def make_transit(n=25, seed=11) -> pd.DataFrame:
    """지하철역 등 대중교통 결절점. 실제: 교통공사 승하차 / 열린데이터광장."""
    rng = np.random.default_rng(seed)
    clat, clon = SEOUL_CENTER
    return pd.DataFrame({
        "station_id": [f"T{i:03d}" for i in range(n)],
        "lat": clat + rng.normal(0, 0.035, n),
        "lon": clon + rng.normal(0, 0.035, n),
        "daily_riders": rng.integers(3000, 120000, n),  # 교통량 프록시
    })
