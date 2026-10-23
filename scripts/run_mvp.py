"""엔드투엔드 MVP 실행기.

사용 예:
  # API 키 없이 합성 데이터로 전 과정 데모 (서울시청 부근 좌표)
  python scripts/run_mvp.py --demo

  # 특정 좌표를 '내 상가'로 지정해 추천
  python scripts/run_mvp.py --demo --lat 37.5700 --lon 126.9820 --k 12 --top 8

파이프라인: 수집 → 피처화 → 유사입지 KNN → 유사입지 기반 업종 추천
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.collect import sources
from src.features.build_features import build_site_matrix
from src.model.similarity import SimilarityModel
from src.recommend.recommend import recommend_biz
from src.features.build_features import site_features


def parse_args():
    p = argparse.ArgumentParser(description="유사입지 기반 상가 활용 추천 MVP")
    p.add_argument("--demo", action="store_true", help="합성 데이터로 실행(키 불필요)")
    p.add_argument("--lat", type=float, default=37.5665, help="내 상가 위도")
    p.add_argument("--lon", type=float, default=126.9780, help="내 상가 경도")
    p.add_argument("--radius", type=int, default=config.ANALYSIS_RADIUS_M, help="분석 반경(m)")
    p.add_argument("--k", type=int, default=config.DEFAULT_K_NEIGHBORS, help="유사입지 수")
    p.add_argument("--top", type=int, default=config.DEFAULT_TOP_N, help="추천 업종 수")
    p.add_argument("--max-sites", type=int, default=600,
                   help="후보 입지 상한(피처 계산 속도용 샘플링)")
    return p.parse_args()


def main():
    args = parse_args()
    use_syn = True if args.demo else None

    print(f"[1/4] 데이터 수집 (radius={args.radius}m) ...")
    stores = sources.load_stores(args.lat, args.lon, args.radius, use_synthetic=use_syn)
    population = sources.load_population(use_synthetic=use_syn)
    transit = sources.load_transit(use_synthetic=True)
    print(f"      상가 {len(stores):,} / 인구격자 {len(population):,} / 교통결절 {len(transit):,}")

    # 후보 입지 = 기존 상가 위치(샘플링). 실무에선 격자/필지로 확장 가능.
    sample = stores.sample(min(args.max_sites, len(stores)), random_state=0)
    sites = sample[["store_id", "lat", "lon"]].rename(columns={"store_id": "site_id"})

    print(f"[2/4] 입지 피처 벡터화 (후보 {len(sites):,}곳) ...")
    feat_mat = build_site_matrix(sites, stores, population, transit, args.radius)

    print("[3/4] 유사입지 모델 학습(KNN) 및 조회 ...")
    model = SimilarityModel().fit(feat_mat)
    my_feats = site_features(args.lat, args.lon, stores, population, transit, args.radius)
    my_feats["lat"], my_feats["lon"] = args.lat, args.lon
    neighbors = model.query(my_feats, k=args.k)

    print("\n=== 내 입지 특성 요약 ===")
    for c in ["pop_total", "store_density", "biz_diversity", "transit_access", "income_idx"]:
        print(f"  - {c:16s}: {my_feats[c]:,.2f}")

    print(f"\n=== 유사 입지 Top {len(neighbors)} ===")
    print(neighbors.to_string(index=False))

    print("[4/4] 유사입지 기반 업종 추천 ...")
    rec = recommend_biz(my_feats, neighbors, stores, args.radius, sites, top_n=args.top)

    print(f"\n=== 추천 업종 Top {len(rec)} (유사입지에서 성업 & 내 입지 과밀 보정) ===")
    print(rec.to_string(index=False))
    print("\n컬럼: score=종합점수 / n_stores=이웃입지 표본 점포수 /",
          "survival_rate=현존비율(%) / avg_months=평균영업개월 / my_count=내 반경 기존점포수")
    print("\n※ 데모는 합성 데이터입니다. 실데이터는 .env 키 설정 후 --demo 없이 실행하세요.")


if __name__ == "__main__":
    main()
