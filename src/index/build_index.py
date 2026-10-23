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


def fetch_by_date(service_key, since="20100101", inds_lcls=None,
                  max_pages=100000, sleep=0.1, verbose=False) -> list[dict]:
    """수정일자 >= since 인 상가를 페이지네이션으로 수집(업종대분류 필터 가능)."""
    url = f"{config.SANGGA_BASE}/storeListByDate"
    rows, page = [], 1
    while page <= max_pages:
        params = {"key": since, "type": "json", "numOfRows": 1000, "pageNo": page}
        if inds_lcls:
            params["indsLclsCd"] = inds_lcls
        r = _get_with_key(url, service_key, params, timeout=60, verify=_verify())
        r.raise_for_status()
        body = r.json().get("body", {}) or {}
        items = _parse_items(body)
        if not items:
            break
        rows.extend(items)
        total = int(body.get("totalCount", 0) or 0)
        if verbose:
            print(f"    page {page}: +{len(items)} (누적 {len(rows):,}/{total:,})")
        if page * 1000 >= total:
            break
        page += 1
        time.sleep(sleep)
    return rows


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
def build_full(service_key, since="20100101", out_dir=INDEX_DIR, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = fetch_large_upjong(service_key)
    frames = []
    for code, name in cats:
        rows = fetch_by_date(service_key, since=since, inds_lcls=code, verbose=verbose)
        df = normalize(rows)
        df.to_csv(out / f"stores_{code}.csv.gz", index=False, compression="gzip")  # 체크포인트
        frames.append(df)
        if verbose:
            print(f"  [{code}] {name}: {len(df):,}")
    allstores = pd.concat(frames, ignore_index=True).drop_duplicates("store_id")
    allstores.to_csv(out / "stores_all.csv.gz", index=False, compression="gzip")
    feats = build_adong_features(allstores)
    feats.to_csv(out / "adong_features.csv.gz", index=False, compression="gzip")
    (out / "meta.json").write_text(json.dumps({
        "built_at": pd.Timestamp.now().isoformat(),
        "since": since,
        "n_stores": int(len(allstores)),
        "n_adong": int(feats.shape[0]),
    }, ensure_ascii=False, indent=2))
    if verbose:
        print(f"\n완료: 상가 {len(allstores):,} / 행정동 {len(feats):,}")
    return allstores, feats
