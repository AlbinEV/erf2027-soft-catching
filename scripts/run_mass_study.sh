#!/bin/bash
# Holding capacity: teacher success versus parcel mass (0.1-0.8 kg) for three tendon tensions, 1000 throws each.
cd "$(dirname "$0")/.."
P=${1:-2}
for T in 20 40 80; do
  [ -s runs/sweeps/mass_T$T.json ] && continue
  nice -n 19 .venv/bin/python scripts/eval_teacher.py --n 1000 --seed0 600000 --procs $P \
      --kw "{\"overrides\": {\"box_mass_kg\": [0.1, 0.8], \"t_max_n\": $T}}" --out runs/sweeps/mass_T$T.json 2>&1 \
      | grep -v -i warn | sed "s/^/[T $T] /"
done
