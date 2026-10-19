#!/bin/bash
# 19 Oct 2026, day chain (max 12 processes, run as a memory-capped user service):
# 1) compliant palm + wrist on the 500 test throws: teacher, classical pipeline, rigid-hand student (no retraining);
# 2) second training seed of the three catch-parameter RL supervisors (hi2_cse, hi2_cam, hi2_all);
# 3) matched evaluation of the seed-1 supervisors on the same conditions as runs/eval_v2_final.json.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
COMPL='{"pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 8, "wrist_stroke_m": 0.15, "wrist_c": 80, "wrist_k": 150}'
echo "start $(date +%H:%M)"
$PY scripts/eval_policies.py --methods teacher classical dagger_v1 --n 500 --seed0 500000 --procs 12 \
  --overrides "$COMPL" --out runs/eval_compliant_500000.json 2>&1 | grep -v -i -E "warn|info"
echo "compliance done $(date +%H:%M)"
$PY scripts/train_high2.py --name hi2_cse_s1 --sensors size,encoder --envs 6 --seed 1 > runs/hi2_cse_s1.log 2>&1 &
$PY scripts/train_high2.py --name hi2_cam_s1 --envs 6 --seed 1 > runs/hi2_cam_s1.log 2>&1 &
wait
echo "seed-1 cse/cam done $(date +%H:%M)"
$PY scripts/train_high2.py --name hi2_all_s1 --sensors size,encoder,ft --two-step --envs 6 --seed 1 \
  > runs/hi2_all_s1.log 2>&1
echo "seed-1 all done $(date +%H:%M)"
$PY scripts/eval_high2.py --methods rl:hi2_cam_s1 rl:hi2_cse_s1 rl:hi2_all_s1 \
  --conditions 'nominal={}' \
  'broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}' \
  'heavy={"box_mass_kg": [0.4, 0.8]}' 'slow_motor={"close_speed_mps": 0.13}' 'fast={"flight_s": [0.45, 0.55]}' \
  'very_slow={"close_speed_mps": 0.10}' \
  --n 500 --seed0 500000 --procs 12 --out runs/eval_v2_seed1.json 2>&1 | grep -v -i -E "warn|info"
echo "all done $(date +%H:%M)"
