"""전국 인구 수집 → 이름 매칭으로 상가 인덱스(adong_features)에 결합.

산출(data/index/):
  adong_population.csv.gz   SGIS 원자료(이름 키)
  adong_features.csv.gz     기존 피처 + 인구 컬럼(덮어씀)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config  # noqa: F401
from src.collect import sgis

IDX = config.DATA_DIR / "index"


def _norm(s):
    return s.astype(str).str.strip() if hasattr(s, "str") else s


def main():
    t0 = time.time()
    print("■ SGIS 전국 인구 수집 (체크포인트·건너뛰기·재시도) ...")
    cp = IDX / "_pop_checkpoint.csv"
    token = sgis.get_token()
    pop, failed = None, [1]
    for attempt in range(4):  # 실패 시군구가 남으면 토큰 갱신 후 재시도(이어받기)
        pop, failed = sgis.load_population_nationwide(
            year="2023", token=token, verbose=True, checkpoint=str(cp), resume=True)
        print(f"  [시도 {attempt+1}] 누적 동 {len(pop):,} / 실패 시군구 {len(failed)}")
        if not failed:
            break
        token = sgis.get_token()
    if failed:
        print(f"  ⚠ 끝내 실패한 시군구 {len(failed)}곳(부분 수집으로 진행):")
        for f in failed[:10]:
            print("    -", f[0], f[1])
    print(f"  수집 행정동 {len(pop):,}")
    pop.to_csv(IDX / "adong_population.csv.gz", index=False, compression="gzip")
    if cp.exists():
        cp.unlink()

    feats = pd.read_csv(IDX / "adong_features.csv.gz", dtype={"adong_cd": str})
    # 이름 정규화 후 결합
    for df in (feats, pop):
        for c in ("sido", "sigungu", "adong_nm"):
            df[c] = df[c].astype(str).str.strip()
    popcols = ["sido", "sigungu", "adong_nm"] + sgis.NUM_FIELDS
    merged = feats.merge(pop[popcols].drop_duplicates(["sido", "sigungu", "adong_nm"]),
                         on=["sido", "sigungu", "adong_nm"], how="left")

    matched = merged["tot_ppltn"].notna().sum()
    print(f"\n■ 결합 결과: {matched:,}/{len(merged):,} 매칭 "
          f"({matched/len(merged)*100:.1f}%)")
    miss = merged[merged["tot_ppltn"].isna()][["sido", "sigungu", "adong_nm"]].head(15)
    if len(miss):
        print("  미매칭 예시(상위 15):")
        print(miss.to_string(index=False))

    merged.to_csv(IDX / "adong_features.csv.gz", index=False, compression="gzip")
    print(f"\n완료: adong_features 에 인구 {len(sgis.NUM_FIELDS)}개 컬럼 추가. "
          f"소요 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
