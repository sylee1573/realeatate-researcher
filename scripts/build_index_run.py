"""전국 인덱스 전체 빌드 실행기.

python scripts/build_index_run.py
  → 업종대분류별 전국 수집(재개 가능) → 소형 쿼리 인덱스 저장
  산출(data/index/): adong_features.csv.gz, adong_biz_counts.csv.gz, meta.json
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401
from src.index import build_index as bi

if not config.SANGGA_SERVICE_KEY:
    print("❌ SANGGA_SERVICE_KEY 없음")
    sys.exit(1)

t0 = time.time()
stores, feats = bi.build_full(config.SANGGA_SERVICE_KEY, verbose=True)
dt = time.time() - t0
print(f"\n⏱ 소요 {dt/60:.1f}분")
print(f"상가 {len(stores):,} / 행정동 {len(feats):,}")
print("산출:", config.DATA_DIR / "index")
