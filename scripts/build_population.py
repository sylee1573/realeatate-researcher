"""전국 인구 수집 → 이름 정규화 매칭으로 상가 인덱스(adong_features)에 결합.

  python scripts/build_population.py              # SGIS 수집 후 결합
  python scripts/build_population.py --merge-only  # 수집 생략, 저장된 인구로 재매칭만

이름 매칭 규칙(85%→99.6%):
  - 동명 구두점/공백 정규화: '종로1.2.3.4가동' == '종로1·2·3·4가동'
  - (시군구,동명) 우선 매칭 → 상가 데이터의 잘못된 시도명('전남광주통합특별시') 무시
  - (시군구) 불일치 시 (시도,동명) 폴백 → 인천 자치구 개편(제물포·서해구 등) 흡수
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config  # noqa: F401
from src.collect import sgis

IDX = config.DATA_DIR / "index"


def _nd(s):
    """동명 정규화: 마침표·가운뎃점·공백 제거."""
    return re.sub(r"[.·\s]", "", str(s))


def match_population(feats: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """이름 정규화로 pop 의 인구 컬럼을 feats 에 결합(기존 인구컬럼은 덮어씀)."""
    feats = feats.drop(columns=[c for c in sgis.NUM_FIELDS if c in feats.columns])
    for df in (feats, pop):
        for c in ("sido", "sigungu", "adong_nm"):
            df[c] = df[c].astype(str).str.strip()
    pop = pop.copy()
    pop["ndkey"] = pop["adong_nm"].map(_nd)

    by_gu, by_sido = {}, {}
    for r in pop.itertuples(index=False):
        by_gu.setdefault((r.sigungu, r.ndkey), []).append(r)
        by_sido.setdefault((r.sido, r.ndkey), []).append(r)

    def pick(sido, sigungu, adong):
        k = _nd(adong)
        c = by_gu.get((sigungu, k))
        if c:
            if len(c) == 1:
                return c[0]
            same = [x for x in c if x.sido == sido]  # 중복 시 시도로 구분
            return same[0] if len(same) == 1 else c[0]
        c = by_sido.get((sido, k))      # 시군구 개편 등 → 시도+동명 폴백
        return c[0] if (c and len(c) == 1) else None

    recs = []
    for fr in feats.itertuples(index=False):
        m = pick(fr.sido, fr.sigungu, fr.adong_nm)
        recs.append({col: (getattr(m, col) if m else None) for col in sgis.NUM_FIELDS})
    return pd.concat([feats.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true",
                    help="SGIS 수집 생략, 저장된 adong_population.csv.gz 로 재매칭만")
    args = ap.parse_args()
    t0 = time.time()

    if args.merge_only:
        pop = pd.read_csv(IDX / "adong_population.csv.gz", dtype={"sgis_cd": str})
        print(f"■ 저장된 인구 재사용: {len(pop):,} 행정동")
    else:
        print("■ SGIS 전국 인구 수집 (체크포인트·건너뛰기·재시도) ...")
        cp = IDX / "_pop_checkpoint.csv"
        token = sgis.get_token()
        pop, failed = None, [1]
        for attempt in range(4):
            pop, failed = sgis.load_population_nationwide(
                year="2023", token=token, verbose=True, checkpoint=str(cp), resume=True)
            print(f"  [시도 {attempt+1}] 누적 동 {len(pop):,} / 실패 시군구 {len(failed)}")
            if not failed:
                break
            token = sgis.get_token()
        if failed:
            print(f"  ⚠ 끝내 실패한 시군구 {len(failed)}곳(부분 수집으로 진행)")
        pop.to_csv(IDX / "adong_population.csv.gz", index=False, compression="gzip")
        if cp.exists():
            cp.unlink()

    feats = pd.read_csv(IDX / "adong_features.csv.gz", dtype={"adong_cd": str})
    merged = match_population(feats, pop)
    matched = merged["tot_ppltn"].notna().sum()
    print(f"\n■ 결합: {matched:,}/{len(merged):,} 매칭 ({matched/len(merged)*100:.1f}%)")
    miss = merged[merged["tot_ppltn"].isna()][["sido", "sigungu", "adong_nm"]]
    if len(miss):
        print(f"  미매칭 {len(miss)}건(대개 SGIS 미수록 최근 분동):")
        print(miss.head(10).to_string(index=False))

    merged.to_csv(IDX / "adong_features.csv.gz", index=False, compression="gzip")
    print(f"\n완료. 소요 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
