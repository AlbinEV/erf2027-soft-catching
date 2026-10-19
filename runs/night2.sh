#!/bin/bash
# Stronger hand (40 N per arm) with the palm-up teacher, parcels 0.1-0.8 kg: student by DAgger, then matched
# evaluation (teacher, classical, student) on 500 test throws, plus the 20 N reference with the same masses.
cd "$(dirname "$0")/.."
OV='{"t_max_n": 40.0, "box_mass_kg": [0.1, 0.8]}'
nice -n 19 .venv/bin/python scripts/dagger.py --name dagger_T40 --procs 20 --iters 12 --overrides "$OV" \
    --teacher '{"turn_up": true}' > runs/dagger_T40.log 2>&1
nice -n 19 .venv/bin/python scripts/eval_policies.py --methods teacher classical dagger_T40 --n 500 --seed0 500000 \
    --procs 20 --overrides "$OV" --teacher '{"turn_up": true}' --out runs/eval_T40_mass.json 2>&1 | grep -v -i -E "warn|Info"
nice -n 19 .venv/bin/python scripts/eval_policies.py --methods teacher classical dagger_v1 --n 500 --seed0 500000 \
    --procs 20 --overrides '{"box_mass_kg": [0.1, 0.8]}' --out runs/eval_T20_mass.json 2>&1 | grep -v -i -E "warn|Info"
nice -n 19 .venv/bin/python scripts/eval_policies.py --methods dagger_T40 --n 500 --seed0 500000 \
    --procs 20 --overrides '{"t_max_n": 40.0}' --out runs/eval_T40_nominal.json 2>&1 | grep -v -i -E "warn|Info"
