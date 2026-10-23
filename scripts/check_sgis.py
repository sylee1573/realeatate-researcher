"""SGIS 키·연결 점검 + 인구 API 사양 탐침.

- 환경변수(SGIS_CONSUMER_KEY/SECRET) 존재 확인 (값 미노출)
- 'SGIS' 포함 환경변수 '이름'만 나열(변수명 불일치 진단용)
- 인증토큰 발급 → 인구 조회 몇 가지 파라미터 조합 시도
- 우리 상가 인덱스의 행정동코드(예: 역삼1동 11680640)로 SGIS가 응답하는지 확인
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401
import requests

_CA = "/root/.ccr/ca-bundle.crt"
VERIFY = _CA if os.path.exists(_CA) else True
BASE = "https://sgisapi.mods.go.kr/OpenAPI3"  # 구 sgisapi.kostat.go.kr 에서 이전됨

names = sorted([k for k in os.environ if "SGIS" in k.upper()])
print("■ 환경변수 중 'SGIS' 포함 이름:", names or "(없음)")

ck = os.getenv("SGIS_CONSUMER_KEY")
cs = os.getenv("SGIS_CONSUMER_SECRET")
print(f"  SGIS_CONSUMER_KEY set={bool(ck)} / SGIS_CONSUMER_SECRET set={bool(cs)}")
if not (ck and cs):
    print("❌ SGIS_CONSUMER_KEY / SGIS_CONSUMER_SECRET 둘 다 필요. 위 이름 목록 참고.")
    sys.exit(1)


def get(url, **params):
    r = requests.get(url, params=params, timeout=20, verify=VERIFY)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"_raw": r.text[:300]}


print("\n■ 인증토큰 발급")
sc, j = get(f"{BASE}/auth/authentication.json", consumer_key=ck, consumer_secret=cs)
print("  status", sc, "errCd", j.get("errCd"), "errMsg", j.get("errMsg"))
tok = (j.get("result") or {}).get("accessToken")
print("  accessToken?", bool(tok))
if not tok:
    print("  전체:", j)
    sys.exit(1)

print("\n■ 인구 조회 탐침 (여러 파라미터 조합)")
trials = [
    {"year": "2023"},
    {"year": "2022"},
    {"year": "2023", "adm_cd": "11", "low_search": "1"},
    {"year": "2023", "adm_cd": "11680640"},            # 우리 인덱스 역삼1동 코드
    {"year": "2022", "adm_cd": "11680640"},
]
for t in trials:
    sc, j = get(f"{BASE}/stats/population.json", accessToken=tok, **t)
    res = j.get("result") if isinstance(j, dict) else None
    print(f"\n  params={t}")
    print(f"    status={sc} errCd={j.get('errCd')} errMsg={j.get('errMsg')}")
    if res:
        print(f"    n={len(res)}  첫행={res[0]}")
