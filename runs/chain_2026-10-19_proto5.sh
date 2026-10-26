#!/bin/bash
# 19 Oct 2026, end of the hour: check of the prototype hand with dorsal stops at 5 deg, the best compromise in proto3,
# under the conditions of proto and proto2. Outputs: runs/proto5/
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
P=12
B='"outward_limit_deg": 5, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
mkdir -p runs/proto5
ev() {  # name, methods, n, extra overrides, [teacher json]
  [ -s runs/proto5/$1.json ] && return
  echo "== $1 $(date +%H:%M)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $P --overrides "{$B$4}" \
    --teacher "${5:-{\}}" --out runs/proto5/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
ev nominal "teacher classical dagger_proto" 500 ""
ev camnoise020 "classical dagger_proto" 500 ', "cam_noise_m": 0.02'
ev latency100 "classical dagger_proto" 500 ', "cam_latency_s": 0.1'
ev motor0.0625 "teacher dagger_proto" 500 ', "close_speed_mps": 0.0625' '{"ratio": 0.45, "t_close": 0.409}'
ev motor0.03 "teacher dagger_proto" 500 ', "close_speed_mps": 0.03' '{"ratio": 0.45, "t_close": 0.78}'
ev fast "teacher dagger_proto" 500 ', "flight_s": [0.40, 0.50]'
ev aim024 "teacher dagger_proto" 500 ', "aim_noise_m": 0.24'
ev small "teacher dagger_proto" 500 ', "box_side_m": [0.05, 0.08]'
ev heavy_T20 "classical" 500 ', "box_mass_kg": [0.4, 0.8]'
echo "all done $(date +%H:%M)"
