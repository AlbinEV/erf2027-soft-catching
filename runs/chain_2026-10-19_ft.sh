#!/bin/bash
# 19 Oct 2026: seeds 2 and 3 of the supervisor with size, encoder and wrist force sensor (hi2_all), to settle whether
# the wrist force sensor helps (seed 0: no; seed 1: +4-5 points with heavy parcels), then the matched evaluation.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
echo "start $(date +%H:%M)"
$PY scripts/train_high2.py --name hi2_all_s2 --sensors size,encoder,ft --two-step --envs 6 --seed 2 > runs/hi2_all_s2.log 2>&1 &
$PY scripts/train_high2.py --name hi2_all_s3 --sensors size,encoder,ft --two-step --envs 6 --seed 3 > runs/hi2_all_s3.log 2>&1 &
wait
echo "training done $(date +%H:%M)"
$PY scripts/eval_high2.py --methods rl:hi2_all_s2 rl:hi2_all_s3 \
  --conditions 'nominal={}' \
  'broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}' \
  'heavy={"box_mass_kg": [0.4, 0.8]}' 'slow_motor={"close_speed_mps": 0.13}' 'fast={"flight_s": [0.45, 0.55]}' \
  'very_slow={"close_speed_mps": 0.10}' \
  --n 500 --seed0 500000 --procs 12 --out runs/eval_v2_seed23.json 2>&1 | grep -v -i -E "warn|info"
echo "all done $(date +%H:%M)"
