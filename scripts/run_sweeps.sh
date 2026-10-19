#!/bin/bash
# Robustness sweeps of 17 Oct 2026: one condition changed at a time, same 500 test throws (seeds 500000+),
# methods teacher (true state), classical (Kalman filter + teacher logic), dagger_v1 (student, sensors only).
# Usage: scripts/run_sweeps.sh PROCS [TAG] [METHODS...]
cd "$(dirname "$0")/.."
P=${1:-3}
TAG=${2:-}
shift 2 2>/dev/null
METHODS=${@:-teacher classical dagger_v1}
run() { [ -s runs/sweeps/$1$TAG.json ] && { echo "[$1$TAG] already done"; return; }; nice -n 19 .venv/bin/python scripts/eval_policies.py --methods $METHODS --n 500 \
        --seed0 500000 --procs $P --overrides "$2" --out runs/sweeps/$1$TAG.json 2>&1 | grep -v -i -E "warn|Info" \
        | sed "s/^/[$1$TAG] /"; }
run aim006 '{"aim_noise_m": 0.06}'
run aim018 '{"aim_noise_m": 0.18}'
run aim024 '{"aim_noise_m": 0.24}'
run camnoise010 '{"cam_noise_m": 0.010}'
run camnoise020 '{"cam_noise_m": 0.020}'
run latency066 '{"cam_latency_s": 0.066}'
run latency100 '{"cam_latency_s": 0.100}'
run camrate30 '{"cam_rate_hz": 30.0}'
run small_parcels '{"box_side_m": [0.05, 0.08]}'
run heavy_parcels '{"box_mass_kg": [0.4, 0.8]}'
run spin15 '{"spin_max": 15.0}'
run fast_throws '{"flight_s": [0.40, 0.50]}'
