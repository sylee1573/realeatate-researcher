"""통계청 SGIS 인구 데이터 수집 (전국 행정동 단위).

SGIS 코드 체계가 상가 인덱스(행정표준코드)와 달라, 결합은 '이름'(시도·시군구·동)으로 한다.
순회: 시도(17) → 시군구(stage) → 동별 인구(population low_search=1).
필드명 확정: stage=cd/addr_name, population=adm_cd/adm_nm + 수치필드(문자열).
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

import config

_CA = "/root/.ccr/ca-bundle.crt"
BASE = config.SGIS_BASE

# 인덱스에 붙일 인구 피처(동 단위)
NUM_FIELDS = [
    "tot_ppltn",        # 총인구
    "avg_age",          # 평균연령
    "ppltn_dnsty",      # 인구밀도
    "tot_family",       # 총가구
    "tot_house",        # 총주택
    "avg_fmember_cnt",  # 평균가구원수
    "corp_cnt",         # 사업체수
    "employee_cnt",     # 종사자수
    "oldage_suprt_per", # 노년부양비
    "juv_suprt_per",    # 유년부양비
    "aged_child_idx",   # 노령화지수
]


def _verify():
    return _CA if os.path.exists(_CA) else True


def _get(url, retries=3, **params):
    # 빈/None 파라미터 제거 — mods.go.kr 는 cd='' 같은 빈값을 422로 거부
    params = {k: v for k, v in params.items() if v not in (None, "")}
    last = None
    for _ in range(retries):  # SGIS 간헐적 TLS 끊김 대비
        try:
            r = requests.get(url, params=params, timeout=25, verify=_verify())
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"SGIS 요청 실패: {last}")


def get_token():
    ck = config.SGIS_CONSUMER_KEY or os.getenv("SGIS_CONSUMER_KEY")
    cs = config.SGIS_CONSUMER_SECRET or os.getenv("SGIS_CONSUMER_SECRET")
    if not (ck and cs):
        raise RuntimeError("SGIS_CONSUMER_KEY / SGIS_CONSUMER_SECRET 미설정")
    j = _get(f"{BASE}/auth/authentication.json", consumer_key=ck, consumer_secret=cs)
    tok = (j.get("result") or {}).get("accessToken")
    if not tok:
        raise RuntimeError(f"SGIS 토큰 발급 실패: {j}")
    return tok


def _stage(token, cd=""):
    return _get(f"{BASE}/addr/stage.json", accessToken=token, cd=cd).get("result") or []


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_population_nationwide(year="2023", token=None, verbose=True, sleep=0.1):
    """전국 행정동 인구 DataFrame (이름 키: sido/sigungu/adong_nm)."""
    token = token or get_token()
    rows = []
    for s in _stage(token):
        sido_nm = s["addr_name"]
        for g in _stage(token, s["cd"]):
            gnm = g["addr_name"]
            j = _get(f"{BASE}/stats/population.json", accessToken=token,
                     adm_cd=g["cd"], year=year, low_search="1")
            dongs = j.get("result") or []
            for d in dongs:
                rec = {"sido": sido_nm, "sigungu": gnm,
                       "adong_nm": d.get("adm_nm"), "sgis_cd": d.get("adm_cd")}
                for f in NUM_FIELDS:
                    rec[f] = _num(d.get(f))
                rows.append(rec)
            if verbose:
                print(f"  {sido_nm} {gnm}: 동 {len(dongs)}")
            time.sleep(sleep)
    return pd.DataFrame(rows)
