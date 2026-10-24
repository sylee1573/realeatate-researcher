"""SGIS 전국 순회(시도→시군구→동) 필드명 확정용 탐침.

addr/stage.json 의 응답 필드와, 시군구 단위 인구조회(동 목록) 필드를 확인한다.
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
    last = None
    for _ in range(3):  # 간헐적 TLS 끊김 대비 재시도
        try:
            r = requests.get(url, params=params, timeout=25, verify=VERIFY)
            return r.status_code, r.json()
        except Exception as e:  # noqa: BLE001
            last = e
    return None, {"_err": str(last)}


ck = os.getenv("SGIS_CONSUMER_KEY")
cs = os.getenv("SGIS_CONSUMER_SECRET")
_, j = get(f"{BASE}/auth/authentication.json", consumer_key=ck, consumer_secret=cs)
tok = (j.get("result") or {}).get("accessToken")
print("token?", bool(tok))
if not tok:
    print(j)
    sys.exit(1)

print("\n■ stage.json 시도(cd 생략)")
_, j = get(f"{BASE}/addr/stage.json", accessToken=tok)
res = j.get("result") or []
print("  errCd", j.get("errCd"), "n", len(res))
if res:
    print("  첫 항목 키:", sorted(res[0].keys()))
    print("  시도 목록:", [(r.get("cd"), r.get("addr_name")) for r in res])

if res:
    sido_cd = res[0].get("cd")
    print(f"\n■ stage.json 시군구 (시도 {sido_cd})")
    _, j2 = get(f"{BASE}/addr/stage.json", accessToken=tok, cd=sido_cd)
    r2 = j2.get("result") or []
    print("  n", len(r2), "예시", [(x.get("cd"), x.get("addr_name")) for x in r2[:5]])

    if r2:
        gu_cd = r2[0].get("cd")
        print(f"\n■ population.json 동목록 (시군구 {gu_cd}, low_search=1)")
        _, j3 = get(f"{BASE}/stats/population.json", accessToken=tok,
                    adm_cd=gu_cd, year="2023", low_search="1")
        r3 = j3.get("result") or []
        print("  errCd", j3.get("errCd"), "n", len(r3))
        if r3:
            print("  첫 동 전체:", r3[0])
