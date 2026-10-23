"""전국 유사입지 추천 — 전국 인덱스 기반(즉시 조회).

사용:
  python scripts/recommend.py --address "서울 강남구 테헤란로 152"
  python scripts/recommend.py --lat 37.5 --lon 127.03 --k 12 --top 12

인덱스(data/index/)가 있어야 하며, 주소 입력 시 카카오 지오코딩 키 필요.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401
from src.index.query import NationalIndex


def main():
    p = argparse.ArgumentParser(description="전국 유사입지 업종 추천")
    p.add_argument("--address", type=str, default=None, help="내 상가 주소(지오코딩)")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--k", type=int, default=10, help="유사 행정동 수")
    p.add_argument("--top", type=int, default=12, help="추천 업종 수")
    args = p.parse_args()

    if args.address:
        from src.utils.geocode import geocode
        lat, lon = geocode(args.address)
        print(f"[지오코딩] '{args.address}' → ({lat:.5f}, {lon:.5f})")
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        print("주소(--address) 또는 좌표(--lat/--lon)를 주세요.")
        sys.exit(1)

    idx = NationalIndex()
    cd, nm, region, dist = idx.nearest_adong(lat, lon)
    print(f"\n내 입지 행정동: {region} {nm} (중심 {dist:.2f}km) [{cd}]")

    sim = idx.similar(cd, k=args.k)
    print(f"\n=== 업종 구성이 비슷한 전국 행정동 Top {len(sim)} ===")
    print(sim.to_string(index=False) if not sim.empty else "  (유사 행정동 없음)")

    rec = idx.recommend(cd, sim, top_n=args.top)
    print(f"\n=== 추천 업종(기회) Top {len(rec)} ===")
    print(rec.to_string(index=False) if not rec.empty else "  (추천 없음)")
    print("\n해석: 유사동평균비중=비슷한 동들의 업종 비중(%), 내동비중=내 동의 비중(%),",
          "기회점수=유사동엔 많은데 내 동엔 적은 정도(진입 기회).")


if __name__ == "__main__":
    main()
