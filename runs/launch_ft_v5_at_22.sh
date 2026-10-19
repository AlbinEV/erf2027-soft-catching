#!/bin/bash
# Night variant: waits until 22:00, then PPO fine-tuning with a higher learning rate
# and slightly more exploration, 9 environments next to ft_v4 (12), deadline 07:30.
cd "$(dirname "$0")/.."
while [ "$(date +%H%M)" -lt 2200 ]; do sleep 60; done
RND='{"aim_noise_m": [0.06, 0.24], "cam_noise_m": [0.005, 0.02], "cam_latency_s": [0.033, 0.1], "cam_rate_hz": [30, 60], "spin_max": [0, 15]}'
OV='{"box_mass_kg": [0.1, 0.8], "flight_s": [0.4, 0.8]}'
nice -n 19 .venv/bin/python scripts/train_ft.py --init dagger_dr --name ft_v5 --envs 9 --warmup 100000 --lr 5e-5 \
    --std 0.03 --eval-every 500000 --eval-episodes 200 --reward '{}' --overrides "$OV" --randomize "$RND" \
    --deadline 07:30 > runs/ft_v5.log 2>&1
