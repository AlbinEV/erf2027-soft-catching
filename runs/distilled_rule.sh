#!/bin/bash
# Rule distilled from the RL supervisor (size + encoder): its mean catch parameters on the 300 selection throws
# (seeds 100000-100299, nominal conditions), fixed or with the closing ratio scaled by the measured motor speed,
# on the same test conditions and throws as runs/eval_v2_final.json. Output: runs/eval_v2_distilled.json
cd "$(dirname "$0")/.."
P='{"ratio":0.56,"absorb":0.52,"brake_time":0.25,"turn_gain":6.0,"motor_hold":0.83}'
.venv/bin/python scripts/eval_high2.py --methods "fixed:$P" "adaptive:$P" \
  --conditions 'nominal={}' \
  'broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}' \
  'heavy={"box_mass_kg": [0.4, 0.8]}' 'slow_motor={"close_speed_mps": 0.13}' 'fast={"flight_s": [0.45, 0.55]}' \
  'very_slow={"close_speed_mps": 0.10}' \
  --n 500 --seed0 500000 --procs ${1:-4} --out runs/eval_v2_distilled.json
