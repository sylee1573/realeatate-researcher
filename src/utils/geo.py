"""공간 계산 유틸 — 반경 필터의 핵심. 외부 의존성 없음(numpy 만)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표(또는 배열) 사이의 대원거리(km). 브로드캐스팅 지원."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def within_radius(center_lat, center_lon, df, radius_m,
                  lat_col="lat", lon_col="lon"):
    """center 기준 radius_m(미터) 안의 행만 반환. 'dist_km' 컬럼 추가.

    MVP 규모(수만 건)에선 벡터화 haversine 으로 충분히 빠릅니다.
    데이터가 수백만 건으로 커지면 PostGIS 공간인덱스나 scipy cKDTree 로 교체.
    """
    if df.empty:
        return df.assign(dist_km=[])
    d = haversine_km(center_lat, center_lon,
                     df[lat_col].to_numpy(), df[lon_col].to_numpy())
    out = df.copy()
    out["dist_km"] = d
    return out[out["dist_km"] <= radius_m / 1000.0].reset_index(drop=True)


def shannon_entropy(counts) -> float:
    """업종 다양성 지수(0=단일업종, 클수록 다양). counts: 범주별 개수."""
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def nearest_dist_km(lat, lon, points_df, lat_col="lat", lon_col="lon"):
    """가장 가까운 지점까지의 거리(km). points 비면 inf."""
    if points_df is None or points_df.empty:
        return float("inf")
    d = haversine_km(lat, lon,
                     points_df[lat_col].to_numpy(), points_df[lon_col].to_numpy())
    return float(np.min(d))
