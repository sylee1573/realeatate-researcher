"""입지 특성 벡터화 — MVP 의 심장부.

하나의 '입지(site)'를 (lat, lon) 으로 받아, 반경 내 인구·교통·경쟁·업종구성을
요약한 숫자 벡터로 변환합니다. 이 벡터가 유사도 계산의 입력이 됩니다.

실무 확장 포인트(2단계):
  - 유동인구 시간대 패턴(통신사), 카드 매출, 주거/업무/상업 비율(건축물대장)
  - 업종별 경쟁밀도를 '관심 업종'으로 세분화
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import geo

# 모델에 투입할 피처 순서(고정) — 표준화·KNN 이 이 순서를 공유
FEATURE_COLS = [
    "pop_total",         # 반경 내 추정 인구
    "pop_young_ratio",   # 청년 비중(가중평균)
    "pop_senior_ratio",  # 고령 비중
    "income_idx",        # 소득 지수(100=평균)
    "store_density",     # 반경 내 상가 수(경쟁/집적)
    "biz_diversity",     # 업종 다양성(섀넌 엔트로피)
    "transit_access",    # 대중교통 접근성(가까울수록 큼)
    "traffic_proxy",     # 인근 최대 교통량(승하차)
]


def _weighted(series, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        return float(np.mean(series)) if len(series) else 0.0
    return float(np.average(np.asarray(series, dtype=float), weights=w))


def site_features(lat, lon, stores, population, transit, radius_m) -> dict:
    """단일 입지의 피처 dict 를 계산."""
    # 인구: 반경 내 집계구를 인구가중 평균
    pop_near = geo.within_radius(lat, lon, population, radius_m)
    if pop_near.empty:
        pop_total = 0.0
        young = senior = 0.0
        income = 100.0
    else:
        pop_total = float(pop_near["pop_total"].sum())
        w = pop_near["pop_total"]
        young = _weighted(pop_near["pop_young_ratio"], w)
        senior = _weighted(pop_near["pop_senior_ratio"], w)
        income = _weighted(pop_near["income_idx"], w)

    # 상가: 밀도 + 업종 다양성 (현존 점포만 — close_date 가 비어 있음)
    alive = stores[stores["close_date"].isna()]
    s_near = geo.within_radius(lat, lon, alive, radius_m)
    store_density = float(len(s_near))
    diversity = geo.shannon_entropy(s_near["biz_mcls"].value_counts().values) \
        if not s_near.empty else 0.0

    # 교통: 최근접 결절점 거리(역) → 접근성으로 변환 + 인근 최대 교통량
    nearest_km = geo.nearest_dist_km(lat, lon, transit)
    transit_access = 1.0 / (1.0 + nearest_km)  # 0~1, 가까울수록 1 에 근접
    t_near = geo.within_radius(lat, lon, transit, radius_m * 3)  # 교통은 더 넓게
    traffic_proxy = float(t_near["daily_riders"].max()) if not t_near.empty else 0.0

    return {
        "pop_total": pop_total,
        "pop_young_ratio": young,
        "pop_senior_ratio": senior,
        "income_idx": income,
        "store_density": store_density,
        "biz_diversity": diversity,
        "transit_access": transit_access,
        "traffic_proxy": traffic_proxy,
    }


def build_site_matrix(sites, stores, population, transit, radius_m) -> pd.DataFrame:
    """여러 입지(sites: lat/lon/site_id) → 피처 행렬(DataFrame, site_id 인덱스).

    sites 의 각 행이 '후보 입지' 입니다.
    MVP 에선 '기존 상가 위치' 자체를 후보 입지로 써서, 유사한 자리에서
    실제로 뭐가 성업 중인지 역추적합니다.
    """
    rows = []
    for r in sites.itertuples(index=False):
        feat = site_features(r.lat, r.lon, stores, population, transit, radius_m)
        feat["site_id"] = r.site_id
        feat["lat"] = r.lat
        feat["lon"] = r.lon
        rows.append(feat)
    df = pd.DataFrame(rows).set_index("site_id")
    return df
