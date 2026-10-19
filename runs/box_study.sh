#!/bin/bash
# Box-shape study (author request, 19 Oct 2026): catch success versus parcel size and shape, rigid nominal hand,
# teacher and student (dagger_v1, trained on 8-10 cm parcels), the same 300 test throws for every shape
# (seeds 500000-500299; sides fixed with box_dims_m, all other throw parameters unchanged, mass 0.10-0.40 kg).
# Usage: bash runs/box_study.sh [procs]
cd "$(dirname "$0")/.."
P=${1:-2}
mkdir -p runs/boxes
for dims in "0.06 0.06 0.06" "0.08 0.08 0.08" "0.10 0.10 0.10" "0.12 0.12 0.12" "0.14 0.14 0.14" \
            "0.15 0.10 0.10" "0.20 0.10 0.10" "0.10 0.10 0.05" "0.15 0.10 0.05" "0.20 0.10 0.05" "0.15 0.15 0.05"; do
  set -- $dims
  name=$(printf "box_%02.0fx%02.0fx%02.0f" $(echo "$1*100" | bc) $(echo "$2*100" | bc) $(echo "$3*100" | bc))
  [ -s runs/boxes/$name.json ] && continue
  echo "== $name $(date +%H:%M)"
  .venv/bin/python scripts/eval_policies.py --methods teacher dagger_v1 --n 300 --seed0 500000 --procs $P \
    --overrides "{\"box_dims_m\": [$1, $2, $3]}" --out runs/boxes/$name.json 2>&1 | grep -v -i -E "warn|info"
done
echo "done $(date +%H:%M)"
