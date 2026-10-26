#!/bin/bash
# 20 Oct 2026, exploratory simulations, part 5 (after part 4): is the classical pipeline under-tuned? Its Kalman filter
# uses the camera noise as measurement noise (5 mm, ideal camera) and 0.5 m/s^2 of process noise; with two realistic
# cameras (filter tuned for 15 mm) it reached 98.6 %. Classical pipeline with other filter settings.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-explore4; do sleep 20; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
mkdir -p runs/explore/kf
ev() {
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
for M in 0.01 0.02 0.04 0.08; do
  for A in 0.5 2.0; do ev kf/ideal_m${M}_a${A} "classical" 500 ", \"kf_meas_std\": $M, \"kf_acc_std\": $A"; done
done
ev kf/ideal_m0.02_a0.1 "classical" 500 ', "kf_meas_std": 0.02, "kf_acc_std": 0.1'
for M in 0.03 0.06; do
  ev kf/pinhole_m$M "classical" 500 ", $P, \"kf_meas_std\": $M"
  ev kf/wide_m$M "classical" 3000 ", $W, \"kf_meas_std\": $M"
  ev kf/camnoise020_m$M "classical" 500 ", \"cam_noise_m\": 0.02, \"kf_meas_std\": $M"
done
echo "all done $(date +%H:%M)"
