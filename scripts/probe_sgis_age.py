"""SGIS 인구 API의 연령대별 분포 지원 여부 탐침.

population.json 에 age_type/gender 등 파라미터를 바꿔가며, 연령대별 인구가
내려오는지(필드/행 구조) 확인한다. 강남구(11230) 기준.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401
import requests

_CA = "/root/.ccr/ca-bundle.crt"
VERIFY = _CA if os.path.exists(_CA) else True
BASE = config.SGIS_BASE


def get(url, **params):
    params = {k: v for k, v in params.items() if v not in (None, "")}
    for _ in range(4):
        try:
            return requests.get(url, params=params, timeout=20, verify=VERIFY).json()
        except Exception:  # noqa: BLE001
            pass
    return {"_err": "fail"}


ck, cs = os.getenv("SGIS_CONSUMER_KEY"), os.getenv("SGIS_CONSUMER_SECRET")
tok = (get(f"{BASE}/auth/authentication.json", consumer_key=ck, consumer_secret=cs)
       .get("result") or {}).get("accessToken")
print("token?", bool(tok))
if not tok:
    sys.exit(1)

GU = "11230"  # 강남구(SGIS)
trials = [
    {"year": "2023", "adm_cd": GU, "low_search": "0"},
    {"year": "2023", "adm_cd": GU, "low_search": "0", "age_type": "1"},
    {"year": "2023", "adm_cd": GU, "low_search": "0", "age_type": "2"},
    {"year": "2023", "adm_cd": GU, "low_search": "0", "age_type": "31"},
    {"year": "2023", "adm_cd": GU, "low_search": "0", "gender": "1"},
    {"year": "2023", "adm_cd": GU, "low_search": "0", "age_from": "0", "age_to": "9"},
]
for t in trials:
    j = get(f"{BASE}/stats/population.json", accessToken=tok, **t)
    res = j.get("result") if isinstance(j, dict) else None
    extra = {k: v for k, v in t.items() if k not in ("year", "adm_cd", "low_search")}
    print(f"\n■ extra={extra or '없음'}  errCd={j.get('errCd')} errMsg={j.get('errMsg')}")
    if res:
        print(f"  n={len(res)}  첫행키={sorted(res[0].keys())}")
        # 연령 관련 값이 보이면 출력
        r0 = res[0]
        age_keys = [k for k in r0 if any(s in k.lower() for s in ("age", "ppltn"))]
        print("  연령/인구 관련:", {k: r0[k] for k in age_keys})
        if len(res) > 1:
            print("  (여러 행) 2~3행 요약:",
                  [{k: r[k] for k in ("adm_cd", "adm_nm") if k in r} for r in res[1:3]])
