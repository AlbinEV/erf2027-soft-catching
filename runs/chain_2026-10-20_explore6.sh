#!/bin/bash
# 20 Oct 2026, exploratory simulations, part 6: a fair classical pipeline. Its Kalman filter measurement noise is tuned
# on the selection throws (seeds 100000+, nominal throws, ideal camera, designed hand), then the tuned pipeline is
# tested in every condition of the paper (test seeds 500000+), also on the rigid hand.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
RPOS='"cam_pos_random": [-0.6, 1.0, 0.5, 1.6, 1.4, 2.3]'
WORST='"cam_calib_m": 0.02, "cam_calib_deg": 2.0, "cam_depth_k": 0.005, "cam_latency_s": 0.066, "cam_dropout": 0.1'
mkdir -p runs/explore/kf_tuned
ev() {  # name, n, overrides (complete, without braces), seed0
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods classical --n $2 --seed0 ${4:-500000} --procs $(procs) --overrides "{$3}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
for M in 0.005 0.02 0.04 0.06 0.08 0.12 0.16; do ev kf_tuned/selection_m$M 500 "$D, \"kf_meas_std\": $M" 100000; done
BEST=$($PY -c "
import json
r={m: json.load(open(f'runs/explore/kf_tuned/selection_m{m}.json'))['table']['classical']['success'] for m in ('0.005','0.02','0.04','0.06','0.08','0.12','0.16')}
print(max(r, key=lambda m: (r[m], -float(m))))")
echo "best measurement noise on the selection throws: $BEST"
for M in $BEST 0.04; do
  ev kf_tuned/m${M}_rigid_nominal 500 "\"kf_meas_std\": $M"
  ev kf_tuned/m${M}_rigid_camnoise020 500 "\"cam_noise_m\": 0.02, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_nominal 500 "$D, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_camnoise020 500 "$D, \"cam_noise_m\": 0.02, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_pinhole 500 "$D, $P, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_pinhole_worst 500 "$D, $P, $WORST, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_side 500 "$D, $P, \"cam_pos_m\": [0.6, -1.4, 1.7], \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_random_worst 1000 "$D, $P, $RPOS, $WORST, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_wide 3000 "$D, $W, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_wide_randompos 3000 "$D, $P, $W, $RPOS, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_speed12 500 "$D, \"close_speed_mps\": 0.03, \"kf_meas_std\": $M"
  ev kf_tuned/m${M}_drag_light 500 "$D, \"air_cd\": 1.05, \"box_mass_kg\": [0.05, 0.15], \"kf_meas_std\": $M"
done
echo "all done $(date +%H:%M)"
