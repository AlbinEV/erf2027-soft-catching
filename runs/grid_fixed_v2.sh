#!/bin/bash
# Best fixed catch parameters for the v2 student under the broad training conditions, on the selection throws
# (seeds 100000-100299, never the test throws).
cd "$(dirname "$0")/.."
M=""
for r in 0.45 0.55 0.65 0.75; do for ab in 0.2 0.4; do for tg in 0.0 4.0; do
  M="$M fixed:{\"ratio\":$r,\"absorb\":$ab,\"brake_time\":0.4,\"turn_gain\":$tg,\"motor_hold\":1.0}"
done; done; done
BROAD='broad={"overrides": {"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}, "randomize": {"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}}'
nice -n 19 .venv/bin/python scripts/eval_high2.py --methods $M --conditions "$BROAD" --n 300 --seed0 100000 \
    --procs ${1:-21} --out runs/theta/grid_fixed_v2.json 2>&1 | grep -v -i -E "warn|Info"
