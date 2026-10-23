"""전국 상가 인덱스 빌더 (API 기반, 1회 구축 + 증분 갱신).

enumeration 전략: storeListByDate(수정일자기준) + 업종대분류 루프.
  - 전체 빌드 : since 를 과거 날짜로 → 사실상 전국 전수
  - 증분 갱신 : since 를 직전 갱신일로 → 바뀐 것만 (분기 자동갱신에 재사용)
행정동 코드 목록이 불필요(응답에 adongCd 포함).

산출물(기본 data/index/):
  stores_all.csv.gz       전국 상가(정규화)
  adong_features.csv.gz   행정동별 피처(유사입지 KNN 입력)
  meta.json               마지막 갱신일 등
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.collect.sources import _get_with_key, _parse_items, _verify
from src.utils.geo import shannon_entropy

INDEX_DIR = config.DATA_DIR / "index"


# --- API ----------------------------------------------------------------
def fetch_large_upjong(service_key) -> list[tuple[str, str]]:
    """업종 대분류 코드·명 목록."""
    url = f"{config.SANGGA_BASE}/largeUpjongList"
    r = _get_with_key(url, service_key, {"type": "json", "numOfRows": 100, "pageNo": 1},
                      timeout=30, verify=_verify())
    r.raise_for_status()
    items = _parse_items(r.json().get("body", {}) or {})
    return [(it.get("indsLclsCd"), it.get("indsLclsNm")) for it in items]


def _paginate(url, service_key, base_params, max_pages=100000, sleep=0.1,
              verbose=False, label="") -> list[dict]:
    """공통 페이지네이션: totalCount 소진 또는 빈 페이지까지."""
    rows, page = [], 1
    while page <= max_pages:
        params = dict(base_params, type="json", numOfRows=1000, pageNo=page)
        r = _get_with_key(url, service_key, params, timeout=60, verify=_verify())
        r.raise_for_status()
        body = r.json().get("body", {}) or {}
        items = _parse_items(body)
        if not items:
            break
        rows.extend(items)
        total = int(body.get("totalCount", 0) or 0)
        if verbose:
            print(f"    {label} page{page}: +{len(items)} (누적 {len(rows):,}/{total:,})")
        if page * 1000 >= total:
            break
        page += 1
        time.sleep(sleep)
    return rows


def fetch_by_upjong(service_key, inds_lcls_cd, **kw) -> list[dict]:
    """업종별 상가업소 조회(storeListInUpjong): 한 업종대분류의 전국 상가.

    전체 빌드의 enumeration 수단. divId=indsLclsCd, key=대분류코드(예 'M1').
    """
    url = f"{config.SANGGA_BASE}/storeListInUpjong"
    return _paginate(url, service_key,
                     {"divId": "indsLclsCd", "key": inds_lcls_cd},
                     label=inds_lcls_cd, **kw)


def fetch_by_date(service_key, since, inds_lcls=None, **kw) -> list[dict]:
    """수정일자기준 조회(storeListByDate): 증분 갱신용(최근 날짜 since 권장)."""
    url = f"{config.SANGGA_BASE}/storeListByDate"
    base = {"key": since}
    if inds_lcls:
        base["indsLclsCd"] = inds_lcls
    return _paginate(url, service_key, base, label="date", **kw)


# --- 정규화 & 피처 ------------------------------------------------------
def normalize(rows) -> pd.DataFrame:
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    df = pd.DataFrame([{
        "store_id": it.get("bizesId"),
        "biz_lcls": it.get("indsLclsNm"),
        "biz_mcls": it.get("indsMclsNm"),
        "adong_cd": it.get("adongCd"),
        "adong_nm": it.get("adongNm"),
        "sido": it.get("ctprvnNm"),
        "sigungu": it.get("signguNm"),
        "lat": f(it.get("lat")),
        "lon": f(it.get("lon")),
    } for it in rows])
    return df.dropna(subset=["lat", "lon"])


def build_adong_biz_counts(stores: pd.DataFrame) -> pd.DataFrame:
    """행정동 × 업종중분류 점포수(롱 포맷) — 추천 집계용 소형 인덱스."""
    g = (stores.groupby(["adong_cd", "biz_mcls"]).size()
         .reset_index(name="count"))
    return g[g["adong_cd"].astype(str).str.len() > 0]


def build_adong_features(stores: pd.DataFrame) -> pd.DataFrame:
    """행정동별 피처: 점포수·업종다양성·대표업종·중심좌표 + 업종중분류 구성비."""
    out = []
    for cd, g in stores.groupby("adong_cd"):
        if not cd:
            continue
        mc = g["biz_mcls"].value_counts()
        out.append({
            "adong_cd": cd,
            "adong_nm": g["adong_nm"].iloc[0],
            "sido": g["sido"].iloc[0],
            "sigungu": g["sigungu"].iloc[0],
            "store_count": len(g),
            "biz_diversity": shannon_entropy(mc.values),
            "top_biz": mc.index[0] if len(mc) else None,
            "lat": g["lat"].mean(),
            "lon": g["lon"].mean(),
        })
    return pd.DataFrame(out)


# --- 빌드 ---------------------------------------------------------------
def build_full(service_key, out_dir=INDEX_DIR, resume=True, verbose=True):
    """전국 전체 빌드: 업종대분류별 전국 수집 → 소형 쿼리 인덱스 저장.

    업종별 중간 저장(stores_<code>.csv.gz)으로 재개 가능. 끊겨 재실행하면
    이미 받은 업종은 건너뜀. 최종 산출은 git 에 올릴 소형 파일 2개 + meta.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = fetch_large_upjong(service_key)
    frames = []
    for code, name in cats:
        cp = out / f"stores_{code}.csv.gz"
        if resume and cp.exists():
            df = pd.read_csv(cp, dtype={"adong_cd": str})
            if verbose:
                print(f"  [{code}] {name}: 체크포인트 재사용 {len(df):,}")
        else:
            rows = fetch_by_upjong(service_key, code, verbose=verbose)
            df = normalize(rows)
            df.to_csv(cp, index=False, compression="gzip")
            if verbose:
                print(f"  [{code}] {name}: {len(df):,}")
        frames.append(df)

    allstores = pd.concat(frames, ignore_index=True).drop_duplicates("store_id")
    feats = build_adong_features(allstores)
    counts = build_adong_biz_counts(allstores)
    feats.to_csv(out / "adong_features.csv.gz", index=False, compression="gzip")
    counts.to_csv(out / "adong_biz_counts.csv.gz", index=False, compression="gzip")
    (out / "meta.json").write_text(json.dumps({
        "built_at": pd.Timestamp.now().isoformat(),
        "n_stores": int(len(allstores)),
        "n_adong": int(feats.shape[0]),
        "source": "소상공인 상가(상권)정보 storeListInUpjong",
    }, ensure_ascii=False, indent=2))
    if verbose:
        print(f"\n완료: 상가 {len(allstores):,} / 행정동 {len(feats):,}")
    return allstores, feats
