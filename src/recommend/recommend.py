"""유사입지 → 업종 추천.

로직: 유사도 높은 이웃 입지들의 반경 내 점포를 모아 업종별로 집계하되,
'생존(오래 버틴·현존) 업종'에 가중치를 줘서 점수화합니다.
그리고 '내 입지에 이미 포화된 업종'은 감점해 과밀을 피합니다.

생존 프록시: 폐업 안 한 점포 + 영업기간(월). 실제 성패는 매출이지만,
MVP 는 공공데이터만으로 돌리기 위해 생존기간을 대리지표로 씁니다(2단계에서 매출로 보강).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import geo


def _survival_months(row, today=None):
    today = today or pd.Timestamp.today()
    end = row["close_date"] if pd.notna(row["close_date"]) else today
    start = row["open_date"]
    if pd.isna(start):
        return 0.0
    return max((end - start).days / 30.0, 0.0)


def recommend_biz(target_feats, neighbors, stores, radius_m,
                  sites, top_n=10, today=None):
    """
    target_feats: 내 입지 피처(dict/Series) — 포화 업종 감점에 사용
    neighbors   : SimilarityModel.query() 결과 (site_id, similarity)
    stores      : 전체 상가 DataFrame
    sites       : site_id → lat/lon (이웃 입지 좌표 조회용)
    반환        : 업종 추천 랭킹 DataFrame
    """
    today = today or pd.Timestamp.today()
    site_xy = sites.set_index("site_id")[["lat", "lon"]]

    # 1) 이웃 입지들의 반경 내 점포 수집 (유사도로 가중)
    records = []
    for nb in neighbors.itertuples(index=False):
        if nb.site_id not in site_xy.index:
            continue
        lat, lon = site_xy.loc[nb.site_id, ["lat", "lon"]]
        near = geo.within_radius(lat, lon, stores, radius_m)
        if near.empty:
            continue
        near = near.copy()
        near["sim"] = nb.similarity
        records.append(near)
    if not records:
        return pd.DataFrame(columns=["biz_mcls", "score", "n_stores",
                                     "survival_rate", "avg_months"])
    pool = pd.concat(records, ignore_index=True)

    # 2) 점포별 생존지표
    pool["months"] = pool.apply(lambda r: _survival_months(r, today), axis=1)
    pool["alive"] = pool["close_date"].isna().astype(float)

    # 3) 업종(중분류)별 집계 — 유사도 × 생존월 가중
    pool["weight"] = pool["sim"] * (1.0 + np.log1p(pool["months"])) * (0.5 + pool["alive"])
    grp = pool.groupby("biz_mcls")
    agg = pd.DataFrame({
        "n_stores": grp.size(),
        "survival_rate": grp["alive"].mean(),
        "avg_months": grp["months"].mean(),
        "raw_score": grp["weight"].sum(),
    })

    # 4) 내 입지 포화도 감점: 이미 내 반경에 많은 업종은 신규 진입 매력 ↓
    my_lat, my_lon = target_feats["lat"], target_feats["lon"]
    my_near = geo.within_radius(my_lat, my_lon, stores[stores["close_date"].isna()], radius_m)
    my_counts = my_near["biz_mcls"].value_counts()
    saturation = agg.index.map(lambda b: my_counts.get(b, 0)).astype(float)
    agg["my_count"] = saturation
    agg["score"] = agg["raw_score"] / (1.0 + saturation)

    out = (agg.sort_values("score", ascending=False)
           .reset_index()
           .rename(columns={"biz_mcls": "업종"}))
    out["survival_rate"] = (out["survival_rate"] * 100).round(1)
    out["avg_months"] = out["avg_months"].round(1)
    out["score"] = out["score"].round(2)
    return out.head(top_n)[["업종", "score", "n_stores", "survival_rate",
                            "avg_months", "my_count"]]
