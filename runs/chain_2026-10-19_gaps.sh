#!/bin/bash
# 19 Oct 2026, one hour of simulations on the open gaps (12 processes, memory-capped service).
# 1) dorsal stops at 0 deg (prototype) instead of -30 deg; 2) hand contact stiffness; 3) compliant hand with heavy
# parcels; 4) camera frames dropped; 5) parcel shapes with the compliant hand; 6) two more seeds of the camera-only
# supervisor and their evaluation. Outputs in runs/gaps/ and runs/eval_v2_cam_seed23.json.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
P=12
COMPL='"pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 8, "wrist_stroke_m": 0.15, "wrist_c": 80, "wrist_k": 150'
mkdir -p runs/gaps/boxes_compliant
ev() {  # name, methods, n, overrides
  [ -s runs/gaps/$1.json ] && return
  echo "== $1 $(date +%H:%M)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $P --overrides "$4" \
    --out runs/gaps/$1.json 2>&1 | grep -v -i -E "warn|info"
}
echo "start $(date +%H:%M)"
ev dorsal0_rigid "teacher classical dagger_v1" 500 '{"outward_limit_deg": 0}'
ev dorsal0_compliant "teacher classical dagger_v1" 500 "{\"outward_limit_deg\": 0, $COMPL}"
for tc in 0.0025 0.01 0.02; do
  ev contact_${tc}_rigid "teacher dagger_v1" 500 "{\"hand_contact_timeconst\": $tc}"
  ev contact_${tc}_compliant "teacher dagger_v1" 500 "{\"hand_contact_timeconst\": $tc, $COMPL}"
done
for T in 20 40; do
  ev compliant_heavy_T$T "teacher dagger_v1" 500 "{\"box_mass_kg\": [0.4, 0.8], \"t_max_n\": $T, $COMPL}"
done
for dp in 0.1 0.3; do
  ev dropout_$dp "classical dagger_v1" 500 "{\"cam_dropout\": $dp}"
done
for dims in "0.06 0.06 0.06" "0.08 0.08 0.08" "0.10 0.10 0.10" "0.12 0.12 0.12" "0.14 0.14 0.14" \
            "0.15 0.10 0.10" "0.20 0.10 0.10" "0.10 0.10 0.05" "0.15 0.10 0.05" "0.20 0.10 0.05" "0.15 0.15 0.05"; do
  set -- $dims
  name=$(printf "box_%02.0fx%02.0fx%02.0f" $(echo "$1*100" | bc) $(echo "$2*100" | bc) $(echo "$3*100" | bc))
  ev boxes_compliant/$name "teacher dagger_v1" 300 "{\"box_dims_m\": [$1, $2, $3], $COMPL}"
done
echo "evaluations done $(date +%H:%M)"
$PY scripts/train_high2.py --name hi2_cam_s2 --envs 6 --seed 2 > runs/hi2_cam_s2.log 2>&1 &
$PY scripts/train_high2.py --name hi2_cam_s3 --envs 6 --seed 3 > runs/hi2_cam_s3.log 2>&1 &
wait
echo "camera-only seeds done $(date +%H:%M)"
$PY scripts/eval_high2.py --methods rl:hi2_cam_s2 rl:hi2_cam_s3 \
  --conditions 'nominal={}' \
  'broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}' \
  'heavy={"box_mass_kg": [0.4, 0.8]}' 'slow_motor={"close_speed_mps": 0.13}' 'fast={"flight_s": [0.45, 0.55]}' \
  'very_slow={"close_speed_mps": 0.10}' \
  --n 500 --seed0 500000 --procs $P --out runs/eval_v2_cam_seed23.json 2>&1 | grep -v -i -E "warn|info"
echo "all done $(date +%H:%M)"
