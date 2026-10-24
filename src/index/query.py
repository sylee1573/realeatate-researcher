"""전국 인덱스 쿼리/추천 레이어.

build_index 가 만든 소형 인덱스(data/index/)를 읽어:
  1) 주소 좌표 → 가장 가까운 행정동
  2) 업종 구성(업종중분류 비중)이 비슷한 전국 행정동 KNN(코사인 유사도)
  3) 유사 행정동에서 성업하는데 내 동엔 적은 '기회 업종' 추천

행정동 경계 폴리곤이 없으므로 대상 행정동은 중심좌표 최근접으로 근사.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from src.utils.geo import haversine_km

INDEX_DIR = config.DATA_DIR / "index"


class NationalIndex:
    def __init__(self, index_dir=INDEX_DIR):
        d = Path(index_dir)
        self.features = pd.read_csv(d / "adong_features.csv.gz", dtype={"adong_cd": str})
        self.counts = pd.read_csv(d / "adong_biz_counts.csv.gz", dtype={"adong_cd": str})
        self.feat = self.features.set_index("adong_cd")
        self._build_matrix()

    def _build_matrix(self):
        piv = self.counts.pivot_table(index="adong_cd", columns="biz_mcls",
                                      values="count", fill_value=0)
        self.piv = piv
        self.adong_ids = list(piv.index)
        self._pos = {cd: i for i, cd in enumerate(self.adong_ids)}
        mat = piv.to_numpy(dtype=float)
        self.totals = mat.sum(axis=1)
        self.share = mat / np.clip(self.totals[:, None], 1, None)   # 업종 구성 비중
        self.nn = NearestNeighbors(metric="cosine").fit(self.share)

    # --- 1) 대상 행정동 ---------------------------------------------------
    def nearest_adong(self, lat, lon):
        f = self.features
        d = haversine_km(lat, lon, f["lat"].to_numpy(), f["lon"].to_numpy())
        i = int(np.argmin(d))
        r = f.iloc[i]
        return r["adong_cd"], r["adong_nm"], f"{r['sido']} {r['sigungu']}", float(d[i])

    # --- 2) 유사 행정동 ---------------------------------------------------
    def similar(self, adong_cd, k=10) -> pd.DataFrame:
        if adong_cd not in self._pos:
            return pd.DataFrame()
        row = self._pos[adong_cd]
        n = min(k + 1, len(self.adong_ids))
        dist, idx = self.nn.kneighbors(self.share[row:row + 1], n_neighbors=n)
        has_pop = "tot_ppltn" in self.feat.columns
        out = []
        for d, j in zip(dist[0], idx[0]):
            cd = self.adong_ids[j]
            if cd == adong_cd:
                continue
            fr = self.feat.loc[cd]
            row = {
                "adong_cd": cd, "행정동": fr["adong_nm"],
                "지역": f"{fr['sido']} {fr['sigungu']}",
                "유사도": round(1 - d, 3), "점포수": int(self.totals[j]),
            }
            if has_pop:
                row["인구"] = (int(fr["tot_ppltn"]) if pd.notna(fr["tot_ppltn"]) else None)
                row["인구밀도"] = (round(fr["ppltn_dnsty"], 0)
                                 if pd.notna(fr.get("ppltn_dnsty")) else None)
            out.append(row)
        return pd.DataFrame(out).head(k)

    def adong_info(self, adong_cd) -> dict:
        """행정동 요약(이름·지역·점포수·인구 등). 추천 상단 표시용."""
        if adong_cd not in self.feat.index:
            return {}
        fr = self.feat.loc[adong_cd]
        d = {"행정동": fr["adong_nm"], "지역": f"{fr['sido']} {fr['sigungu']}"}
        for c, lab in [("tot_ppltn", "인구"), ("ppltn_dnsty", "인구밀도"),
                       ("avg_age", "평균연령"), ("employee_cnt", "종사자수"),
                       ("aged_child_idx", "노령화지수")]:
            if c in self.feat.columns and pd.notna(fr.get(c)):
                d[lab] = fr[c]
        return d

    # --- 3) 기회 업종 추천 -----------------------------------------------
    def recommend(self, adong_cd, similar_df, top_n=12) -> pd.DataFrame:
        sim_cds = [c for c in similar_df["adong_cd"] if c in self.piv.index]
        if not sim_cds:
            return pd.DataFrame()
        sim_share = self.piv.loc[sim_cds].div(
            self.piv.loc[sim_cds].sum(axis=1), axis=0).mean(axis=0)  # 유사동 평균 비중
        if adong_cd in self.piv.index:
            t = self.piv.loc[adong_cd]
            my_share = t / max(t.sum(), 1)
        else:
            my_share = pd.Series(0.0, index=sim_share.index)
        df = pd.DataFrame({
            "업종": sim_share.index,
            "유사동평균비중": sim_share.values * 100,
            "내동비중": [float(my_share.get(b, 0.0)) * 100 for b in sim_share.index],
        })
        df["기회점수"] = (df["유사동평균비중"] - df["내동비중"]).round(2)  # 유사동엔 많고 내 동엔 적음
        df["유사동평균비중"] = df["유사동평균비중"].round(2)
        df["내동비중"] = df["내동비중"].round(2)
        return df.sort_values("기회점수", ascending=False).head(top_n).reset_index(drop=True)
