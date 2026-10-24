"""전국 인덱스(CSV) → 웹앱 내장용 압축 JSON.

산출: web/national_index.json
  { biz:[업종중분류...], adong:[ {cd,nm,sido,sgg,lat,lon,sc, pop:[6값|null], bi:[...], bc:[...]} ] }
  - biz: 업종중분류 이름 배열(인덱스로 참조)
  - bi/bc: 그 행정동의 업종 인덱스/점포수 (희소)
  - pop: [tot_ppltn, ppltn_dnsty, avg_age, employee_cnt, aged_child_idx, avg_fmember_cnt] 또는 null
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

IDX = config.DATA_DIR / "index"
# pop 배열 순서(웹에서 참조): 인구·밀도·평균연령·종사자수·가구원수·유소년부양비·노년부양비
POP = ["tot_ppltn", "ppltn_dnsty", "avg_age", "employee_cnt", "avg_fmember_cnt",
       "juv_suprt_per", "oldage_suprt_per"]

feat = pd.read_csv(IDX / "adong_features.csv.gz", dtype={"adong_cd": str})
cnt = pd.read_csv(IDX / "adong_biz_counts.csv.gz", dtype={"adong_cd": str})

# 업종중분류 인덱스(총 점포수 내림차순)
order = cnt.groupby("biz_mcls")["count"].sum().sort_values(ascending=False)
biz = list(order.index)
bidx = {b: i for i, b in enumerate(biz)}

by_cd = {cd: g for cd, g in cnt.groupby("adong_cd")}

def num(v):
    return None if pd.isna(v) else (int(v) if float(v).is_integer() else round(float(v), 1))

adong = []
for r in feat.itertuples(index=False):
    g = by_cd.get(r.adong_cd)
    bi, bc = [], []
    if g is not None:
        for b, c in zip(g["biz_mcls"], g["count"]):
            bi.append(bidx[b]); bc.append(int(c))
    pop = [num(getattr(r, c)) for c in POP]
    if all(v is None for v in pop):
        pop = None
    adong.append({
        "cd": r.adong_cd, "nm": r.adong_nm, "sido": r.sido, "sgg": r.sigungu,
        "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
        "sc": int(r.store_count), "pop": pop, "bi": bi, "bc": bc,
    })

out = {"biz": biz, "adong": adong}
dst = config.ROOT / "web" / "national_index.json"
dst.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
mb = dst.stat().st_size / 1e6
print(f"완료: {dst}  ({mb:.2f} MB)  행정동 {len(adong):,} · 업종 {len(biz)}")
