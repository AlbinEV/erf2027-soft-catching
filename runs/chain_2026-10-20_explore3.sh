#!/bin/bash
# 20 Oct 2026, exploratory simulations, part 3 (after part 2): does a second camera close the gap between the classical
# pipeline and the students also with a degraded camera and with varied throws; can a stiffer wrist keep parcels of
# 1.2-2.0 kg off the robot (they touch it in every throw with the 150 N/m spring).
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-explore2; do sleep 20; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
C2='"cam2_pos_m": [-0.3, 0.6, 2.0]'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
WORST='"cam_calib_m": 0.02, "cam_calib_deg": 2.0, "cam_depth_k": 0.005, "cam_latency_s": 0.066, "cam_dropout": 0.1'
ev() {
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
ev twocam_worst "classical dagger_proto_cam dagger_proto_campos" 500 ", $P, $C2, $WORST"
ev onecam_worst_campos "dagger_proto_campos" 500 ", $P, $WORST"
ev twocam_wide_all "dagger_proto_all_long" 3000 ", $P, $C2, $W"
for K in 300 600; do
  for CW in 80 160; do
    ev heavy_wrist_k${K}_c${CW} "teacher dagger_proto" 500 ", \"box_mass_kg\": [1.2, 2.0], \"wrist_k\": $K, \"wrist_c\": $CW"
  done
done
ev heavy_wrist_k600_c160_nominal "teacher dagger_proto" 500 ", \"wrist_k\": 600, \"wrist_c\": 160"
echo "all done $(date +%H:%M)"
