#!/bin/bash
# Teacher, closing ratio 0.45, fraction of the parcel velocity followed by the palm 0, 0.2, 0.4, 0.6, on the 300
# selection throws (seeds 100000-100299). Output: runs/absorb/teacher_absorb_<a>.json
cd "$(dirname "$0")/.."
P=${1:-4}
mkdir -p runs/absorb
for a in 0.0 0.2 0.4 0.6; do
  echo "== absorb $a"
  .venv/bin/python scripts/eval_teacher.py --n 300 --seed0 100000 --procs $P \
    --kw "{\"teacher\": {\"ratio\": 0.45, \"absorb\": $a}}" --out runs/absorb/teacher_absorb_$a.json
done
