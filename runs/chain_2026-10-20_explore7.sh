#!/bin/bash
# 20 Oct 2026, exploratory simulations, part 7: best case for the classical pipeline, its filter tuned for each hard
# condition on that condition's selection throws (seeds 100000+), then tested (seeds 500000+): varied throws, camera
# with 2 cm of noise, degraded realistic camera.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
WORST='"cam_calib_m": 0.02, "cam_calib_deg": 2.0, "cam_depth_k": 0.005, "cam_latency_s": 0.066, "cam_dropout": 0.1'
mkdir -p runs/explore/kf_cond
ev() {  # name, n, overrides, seed0
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods classical --n $2 --seed0 ${4:-500000} --procs $(procs) --overrides "{$3}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
tune() {  # tag, n selection, n test, overrides, candidate values
  local tag=$1 ns=$2 nt=$3 ov=$4; shift 4
  for M in "$@"; do ev kf_cond/${tag}_sel_m$M $ns "$ov, \"kf_meas_std\": $M" 100000; done
  local best=$($PY -c "
import json,sys
v=sys.argv[1:]; r={m: json.load(open(f'runs/explore/kf_cond/${tag}_sel_m{m}.json'))['table']['classical']['success'] for m in v}
print(max(v, key=lambda m: (r[m], -float(m))))" "$@")
  echo "$tag: best $best"
  ev kf_cond/${tag}_test_m$best $nt "$ov, \"kf_meas_std\": $best"
}
echo "start $(date +%H:%M)"
tune wide 1000 3000 "$D, $W" 0.005 0.01 0.02 0.04
tune camnoise020 500 500 "$D, \"cam_noise_m\": 0.02" 0.06 0.1 0.15 0.2 0.3
tune pinhole_worst 500 500 "$D, $P, $WORST" 0.03 0.06 0.1 0.15
echo "all done $(date +%H:%M)"
