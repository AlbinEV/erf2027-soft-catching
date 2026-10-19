#!/bin/bash
cd "$(dirname "$0")/.."
while ps -eo comm,args | awk '$1 ~ /^python/' | grep -q "dagger_theta.py --name dagger_theta "; do sleep 10; done
OV='{"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}'
RND='{"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}'
nice -n 19 .venv/bin/python scripts/dagger_theta2.py --name dagger_theta2 --procs 21 --iters 16 \
    --teacher-episodes 2000 --episodes 1000 --overrides "$OV" --randomize "$RND" > runs/dagger_theta2.log 2>&1
