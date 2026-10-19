#!/bin/bash
# Final matched comparison of the catch-parameter policies v2 on the test throws (seeds 500000-500499).
cd "$(dirname "$0")/.."
NOM='fixed:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}'
BEST='fixed:{"ratio":0.65,"absorb":0.4,"brake_time":0.4,"turn_gain":4.0,"motor_hold":1.0}'
ADAPT='adaptive:{"ratio":0.55,"absorb":0.4,"brake_time":0.4,"turn_gain":4.0,"motor_hold":1.0}'
nice -n 19 .venv/bin/python scripts/eval_high2.py --methods "$NOM" "$BEST" "$ADAPT" rl:hi2_cam rl:hi2_cse rl:hi2_all \
  --conditions 'nominal={}' \
  'broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}' \
  'heavy={"box_mass_kg": [0.4, 0.8]}' 'slow_motor={"close_speed_mps": 0.13}' 'fast={"flight_s": [0.45, 0.55]}' \
  'very_slow={"close_speed_mps": 0.10}' \
  --n 500 --seed0 500000 --procs ${1:-21} --out runs/eval_v2_final.json 2>&1 | grep -v -i -E "warn|Info"
