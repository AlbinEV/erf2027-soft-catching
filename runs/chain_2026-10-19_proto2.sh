#!/bin/bash
# 19 Oct 2026, second part of the hour on the prototype configuration:
# 1) intermediate dorsal stops (outward limit 10 and 20 deg): nominal throws, 14 cm cube, 20 x 10 x 10 cm box;
# 2) harder conditions: fast throws, aim error 24 cm, spin 15 rad/s, small parcels, slower motors (closing 0.41 and
#    0.78 s, teacher with the matching closing time and ratio 0.45 or 0.6). Student: dagger_proto.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
P=12
base() { echo "\"pad_stroke_m\": 0.05, \"housing_m\": 0.10, \"pad_c\": 40, \"wrist_stroke_m\": 0.10, \"wrist_c\": 80, \"wrist_k\": 150, \"outward_limit_deg\": $1"; }
mkdir -p runs/proto2
ev() {  # name, methods, n, overrides, [teacher json]
  [ -s runs/proto2/$1.json ] && return
  echo "== $1 $(date +%H:%M)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $P --overrides "{$4}" \
    --teacher "${5:-{\}}" --out runs/proto2/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
for L in 10 20; do
  ev dorsal${L}_nominal "teacher dagger_proto" 500 "$(base $L)"
  ev dorsal${L}_cube14 "teacher dagger_proto" 300 "$(base $L), \"box_dims_m\": [0.14, 0.14, 0.14]"
  ev dorsal${L}_box20x10x10 "teacher dagger_proto" 300 "$(base $L), \"box_dims_m\": [0.20, 0.10, 0.10]"
done
B=$(base 0)
ev fast "teacher dagger_proto" 500 "$B, \"flight_s\": [0.40, 0.50]"
ev aim024 "teacher dagger_proto" 500 "$B, \"aim_noise_m\": 0.24"
ev spin15 "teacher dagger_proto" 500 "$B, \"spin_max\": 15.0"
ev small "teacher dagger_proto" 500 "$B, \"box_side_m\": [0.05, 0.08]"
for R in 0.45 0.6; do
  ev motor0.0625_r$R "teacher dagger_proto" 500 "$B, \"close_speed_mps\": 0.0625" "{\"ratio\": $R, \"t_close\": 0.409}"
  ev motor0.03_r$R "teacher dagger_proto" 500 "$B, \"close_speed_mps\": 0.03" "{\"ratio\": $R, \"t_close\": 0.78}"
done
echo "all done $(date +%H:%M)"
