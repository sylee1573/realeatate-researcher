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

import config
from src.utils.geo import haversine_km

INDEX_DIR = config.DATA_DIR / "index"

# 유사도에 반영할 인구 특성(있을 때). 규모형은 로그변환 후 표준화.
POP_SIM_FIELDS = ["tot_ppltn", "ppltn_dnsty", "employee_cnt",
                  "avg_age", "aged_child_idx", "avg_fmember_cnt"]
POP_LOG_FIELDS = {"tot_ppltn", "ppltn_dnsty", "employee_cnt"}


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
        self._share_norm = np.linalg.norm(self.share, axis=1)

        # 인구 특성 행렬(표준화) — 모든 피처가 있는 동만 유효(pop_avail)
        self.has_pop = all(c in self.feat.columns for c in POP_SIM_FIELDS)
        if self.has_pop:
            P = self.feat.reindex(self.adong_ids)[POP_SIM_FIELDS].astype(float).copy()
            for c in POP_LOG_FIELDS:
                P[c] = np.log1p(P[c])
            mean, std = P.mean(), P.std(ddof=0).replace(0, 1)
            self._popZ = ((P - mean) / std).to_numpy()
            self.pop_avail = ~np.isnan(self._popZ).any(axis=1)

    # --- 1) 대상 행정동 ---------------------------------------------------
    def nearest_adong(self, lat, lon):
        f = self.features
        d = haversine_km(lat, lon, f["lat"].to_numpy(), f["lon"].to_numpy())
        i = int(np.argmin(d))
        r = f.iloc[i]
        return r["adong_cd"], r["adong_nm"], f"{r['sido']} {r['sigungu']}", float(d[i])

    # --- 2) 유사 행정동 (업종 구성 + 인구 특성 블렌드) --------------------
    def similar(self, adong_cd, k=10, alpha=0.6) -> pd.DataFrame:
        """alpha: 업종 구성 가중(0~1). 나머지(1-alpha)는 인구 특성 유사도."""
        if adong_cd not in self._pos:
            return pd.DataFrame()
        i = self._pos[adong_cd]

        # 업종 구성 코사인 유사도 (전 행정동 대비)
        t = self.share[i]
        denom = self._share_norm * (np.linalg.norm(t) or 1.0)
        biz_sim = np.divide(self.share @ t, denom,
                            out=np.zeros(len(self.adong_ids)), where=denom > 0)
        sim = biz_sim

        # 인구 특성 유사도 블렌드 (대상·후보 모두 인구 데이터 있을 때만).
        # 대상에 인구가 있으면, 인구 없는 후보는 공정 비교 불가라 제외(-inf).
        if self.has_pop and self.pop_avail[i]:
            d = np.sqrt(np.nansum((self._popZ - self._popZ[i]) ** 2, axis=1))
            pop_sim = 1.0 / (1.0 + d)
            sim = np.where(self.pop_avail, alpha * biz_sim + (1 - alpha) * pop_sim,
                           -np.inf)

        has_pop = "tot_ppltn" in self.feat.columns
        out = []
        for j in np.argsort(-sim):
            cd = self.adong_ids[j]
            if cd == adong_cd:
                continue
            fr = self.feat.loc[cd]
            r = {"adong_cd": cd, "행정동": fr["adong_nm"],
                 "지역": f"{fr['sido']} {fr['sigungu']}",
                 "유사도": round(float(sim[j]), 3), "점포수": int(self.totals[j])}
            if has_pop:
                r["인구"] = int(fr["tot_ppltn"]) if pd.notna(fr["tot_ppltn"]) else None
                r["인구밀도"] = (round(fr["ppltn_dnsty"], 0)
                               if pd.notna(fr.get("ppltn_dnsty")) else None)
            out.append(r)
            if len(out) >= k:
                break
        return pd.DataFrame(out)

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
