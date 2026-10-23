"""유사입지 탐색 — 표준화 + KNN.

왜 이 방식인가(실무 판단):
  - 피처 스케일이 제각각(인구 수천 vs 비율 0~1)이라 표준화는 필수.
  - MVP 는 해석가능성이 중요 → 블랙박스 임베딩보다 거리기반 KNN 이 낫다.
  - 데이터가 커지고 비선형 패턴이 보이면 그때 임베딩/오토인코더로 교체.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLS


class SimilarityModel:
    def __init__(self, feature_cols=None, weights=None):
        self.feature_cols = feature_cols or FEATURE_COLS
        # 피처별 중요도 가중(선택). 예: 유동인구를 더 중시.
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None
        self.scaler = StandardScaler()
        self.nn: NearestNeighbors | None = None
        self.index_ = None

    def fit(self, feature_df: pd.DataFrame):
        X = feature_df[self.feature_cols].to_numpy(dtype=float)
        Xs = self.scaler.fit_transform(X)
        if self.weights is not None:
            Xs = Xs * self.weights
        self.nn = NearestNeighbors(metric="euclidean")
        self.nn.fit(Xs)
        self.index_ = feature_df.index
        self._Xs = Xs
        return self

    def _vec(self, feats: dict | pd.Series) -> np.ndarray:
        x = np.array([[float(feats[c]) for c in self.feature_cols]])
        xs = self.scaler.transform(x)
        if self.weights is not None:
            xs = xs * self.weights
        return xs

    def query(self, feats, k=10, exclude_self=True) -> pd.DataFrame:
        """입지 피처 → 유사 입지 top-k.

        반환: site_id, distance, similarity(0~1, 클수록 유사).
        """
        if self.nn is None:
            raise RuntimeError("fit() 먼저 호출")
        xs = self._vec(feats)
        n = min(k + (1 if exclude_self else 0), len(self.index_))
        dist, idx = self.nn.kneighbors(xs, n_neighbors=n)
        dist, idx = dist[0], idx[0]
        site_ids = self.index_[idx]
        out = pd.DataFrame({"site_id": site_ids, "distance": dist})
        if exclude_self and len(out):
            # 거리 0(자기 자신) 한 건 제거
            out = out.iloc[1:] if out["distance"].iloc[0] < 1e-9 else out
        out = out.head(k).reset_index(drop=True)
        # 거리 → 유사도(0~1): 지수감쇠. 임계값은 데이터 분포 보고 튜닝.
        out["similarity"] = np.exp(-out["distance"] / (out["distance"].mean() + 1e-9))
        return out
