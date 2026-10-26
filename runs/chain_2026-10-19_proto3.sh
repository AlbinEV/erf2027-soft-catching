#!/bin/bash
# 19 Oct 2026, third part of the hour on the prototype configuration: dorsal stops at 5, 10 and 15 deg, which help the
# 14 cm cube and the 20 x 10 x 10 cm box (proto2). For each angle: heavy parcels (0.4-0.8 kg, 20 N), slower motor
# (0.125 m/s), all box shapes with the student dagger_proto; then rules and supervisor at 10 deg.
# Starts after runs/chain_2026-10-19_proto2.sh. Outputs: runs/proto3/
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-proto2; do sleep 20; done
PY=".venv/bin/python"
P=12
mkdir -p runs/proto3
pr() { echo "\"pad_stroke_m\": 0.05, \"housing_m\": 0.10, \"pad_c\": 40, \"wrist_stroke_m\": 0.10, \"wrist_c\": 80, \"wrist_k\": 150, \"outward_limit_deg\": $1"; }
ev() {  # name, methods, n, overrides, [teacher json]
  [ -s runs/proto3/$1.json ] && return
  echo "== $1 $(date +%H:%M)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $P --overrides "{$4}" \
    --teacher "${5:-{\}}" --out runs/proto3/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
SHAPES=("0.06 0.06 0.06" "0.08 0.08 0.08" "0.10 0.10 0.10" "0.12 0.12 0.12" "0.14 0.14 0.14" "0.15 0.10 0.10"
        "0.20 0.10 0.10" "0.10 0.10 0.05" "0.15 0.10 0.05" "0.20 0.10 0.05" "0.15 0.15 0.05")
echo "start $(date +%H:%M)"
for L in 10 5 15; do
  ev d${L}_heavy_T20 "teacher dagger_proto" 500 "$(pr $L), \"box_mass_kg\": [0.4, 0.8]"
  ev d${L}_motor_0.125 "teacher dagger_proto" 500 "$(pr $L), \"close_speed_mps\": 0.125" '{"ratio": 0.45, "t_close": 0.265}'
  for dims in "${SHAPES[@]}"; do
    set -- $dims
    name=$(printf "box_%02.0fx%02.0fx%02.0f" $(echo "$1*100" | bc) $(echo "$2*100" | bc) $(echo "$3*100" | bc))
    ev d${L}_$name "dagger_proto" 300 "$(pr $L), \"box_dims_m\": [$1, $2, $3]"
  done
done
echo "== rules and supervisor, dorsal stops at 10 deg $(date +%H:%M)"
PR=$(pr 10)
D='{"ratio":0.56,"absorb":0.52,"brake_time":0.25,"turn_gain":6.0,"motor_hold":0.83}'
[ -s runs/proto3/eval_v2_d10.json ] || $PY scripts/eval_high2.py \
  --methods 'fixed:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}' "adaptive:$D" rl:hi2_cse \
  --conditions "nominal={$PR}" "heavy={$PR, \"box_mass_kg\": [0.4, 0.8]}" "slow_motor={$PR, \"close_speed_mps\": 0.13}" \
  "very_slow={$PR, \"close_speed_mps\": 0.10}" \
  --n 500 --seed0 500000 --procs $P --out runs/proto3/eval_v2_d10.json 2>&1 | grep -v -i -E "warn|info"
echo "all done $(date +%H:%M)"
