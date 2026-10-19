#!/bin/bash
# Waits for the night fine-tuning runs (deadline 07:30), then evaluates their best checkpoints against the
# randomized-condition student on the same 500 test throws: nominal and the hardest conditions.
cd "$(dirname "$0")/.."
sleep 120
while ps -eo comm,args | awk '$1 ~ /^python/' | grep -q "train_ft.py"; do sleep 60; done
M="dagger_dr"
[ -s runs/ft_v4/model_best.zip ] && M="$M ft_v4:best"
[ -s runs/ft_v5/model_best.zip ] && M="$M ft_v5:best"
for c in "nominal|{}" "heavy|{\"box_mass_kg\": [0.4, 0.8]}" "aim024|{\"aim_noise_m\": 0.24}" "fast|{\"flight_s\": [0.40, 0.50]}"; do
  name=${c%%|*}; ov=${c#*|}
  nice -n 19 .venv/bin/python scripts/eval_policies.py --methods $M --n 500 --seed0 500000 --procs 13 \
      --overrides "$ov" --out runs/eval_ft_$name.json 2>&1 | grep -v -i -E "warn|Info" | sed "s/^/[$name] /"
done
