"""카카오(또는 VWorld) 지오코딩 키 점검.

사용: python scripts/check_kakao.py "서울 중구 세종대로 110"
키가 정상이면 주소 → 좌표가 출력됩니다. 전체 파이프라인과 분리해 키만 빠르게 검증.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401  (.env 를 os.environ 으로 로드)
from src.utils.geocode import geocode

addr = " ".join(sys.argv[1:]) or "서울 중구 세종대로 110"
try:
    lat, lon = geocode(addr)
    print(f"✅ 성공: '{addr}'\n   → 위도 {lat:.6f}, 경도 {lon:.6f}")
    print(f"\n이제 다음으로 실행하세요:\n   python scripts/run_mvp.py --address \"{addr}\"")
except Exception as e:  # noqa: BLE001
    print(f"❌ 실패: {e}")
    print("   .env 에 KAKAO_REST_KEY(또는 VWORLD_KEY) 가 들어갔는지 확인하세요.")
    sys.exit(1)
