"""전국 인덱스 빌더 소규모 검증 — 큰 빌드 전에 응답 필드/동작 확인.

- 업종대분류 목록 조회
- 한 업종, 2페이지만 storeListByDate 로 받아 raw 키와 정규화 결과 확인
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401  (.env 로드)
from src.index import build_index as bi

key = config.SANGGA_SERVICE_KEY
if not key:
    print("❌ SANGGA_SERVICE_KEY 없음")
    sys.exit(1)

print("■ 업종대분류 목록")
cats = bi.fetch_large_upjong(key)
print(f"  개수={len(cats)}  예시={cats[:5]}")

if not cats:
    print("❌ 업종대분류 응답 비어있음 — 엔드포인트/키 확인 필요")
    sys.exit(1)

code, name = cats[0]
print(f"\n■ storeListInUpjong 소량 테스트 (업종 {code}/{name}, 2페이지)")
rows = bi.fetch_by_upjong(key, code, max_pages=2, verbose=True)
print(f"  수집 rows={len(rows)}")
if rows:
    print(f"  첫 레코드 키: {sorted(rows[0].keys())}")
    df = bi.normalize(rows)
    print(f"  정규화 후 rows={len(df)} (좌표 결측 제거)")
    print(df.head(5).to_string())
    print("\n  행정동 분포 상위5:")
    print(df["adong_nm"].value_counts().head(5).to_string())
else:
    print("  ⚠ rows 0 — since 날짜/파라미터 점검 필요")
