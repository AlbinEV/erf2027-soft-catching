#!/bin/bash
# Damping tuning of the compliant palm plate (5 cm, housing 10 cm) and wrist (15 or 25 cm), teacher, 200 throws.
cd "$(dirname "$0")/.."
P=${1:-4}
for ws in 0.15 0.25; do for pc in 8 20 40; do for wc in 20 40 80; do
  name=cg_w${ws}_pc${pc}_wc${wc}
  [ -s runs/theta/$name.json ] && continue
  ov="{\"pad_stroke_m\": 0.05, \"housing_m\": 0.10, \"pad_c\": $pc, \"wrist_stroke_m\": $ws, \"wrist_c\": $wc, \"wrist_k\": 150}"
  echo "== $name"
  nice -n 19 .venv/bin/python scripts/eval_teacher.py --n 200 --seed0 500000 --procs $P --kw "{\"overrides\": $ov}" \
      --out runs/theta/$name.json 2>&1 | grep -v -i warn | sed 's/ | not triggered.*touched UR/ UR/'
done; done; done
