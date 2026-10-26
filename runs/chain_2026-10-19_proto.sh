#!/bin/bash
# 19 Oct 2026, one hour of simulations on the prototype configuration of the hand:
# dorsal stops at 0 deg, palm 5 cm stroke with 40 N s/m, wrist 10 cm stroke with 80 N s/m and 150 N/m.
# 1) teacher, classical pipeline and current student (dagger_v1) under several conditions; 2) a student trained with
# DAgger on the prototype hand (dagger_proto) and its evaluation; 3) catch-parameter rules and supervisor on it.
# Outputs: runs/proto/*.json, runs/dagger_proto/, runs/proto/eval_v2_proto.json
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
P=12
PR='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
mkdir -p runs/proto/boxes runs/proto/boxes_proto
ev() {  # name, methods, n, extra overrides, [teacher json]
  [ -s runs/proto/$1.json ] && return
  echo "== $1 $(date +%H:%M)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $P --overrides "{$PR$4}" \
    --teacher "${5:-{\}}" --out runs/proto/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
SHAPES=("0.06 0.06 0.06" "0.08 0.08 0.08" "0.10 0.10 0.10" "0.12 0.12 0.12" "0.14 0.14 0.14" "0.15 0.10 0.10"
        "0.20 0.10 0.10" "0.10 0.10 0.05" "0.15 0.10 0.05" "0.20 0.10 0.05" "0.15 0.15 0.05")
boxes() {  # dir, methods
  for dims in "${SHAPES[@]}"; do
    set -- $dims
    name=$(printf "box_%02.0fx%02.0fx%02.0f" $(echo "$1*100" | bc) $(echo "$2*100" | bc) $(echo "$3*100" | bc))
    ev $BOXDIR/$name "$METH" 300 ", \"box_dims_m\": [$1, $2, $3]"
  done
}
echo "start $(date +%H:%M)"
ev nominal "teacher classical dagger_v1" 500 ""
ev heavy_T20 "teacher dagger_v1" 500 ', "box_mass_kg": [0.4, 0.8]'
ev heavy_T40 "teacher dagger_v1" 500 ', "box_mass_kg": [0.4, 0.8], "t_max_n": 40'
ev camnoise020 "classical dagger_v1" 500 ', "cam_noise_m": 0.02'
ev latency100 "classical dagger_v1" 500 ', "cam_latency_s": 0.1'
ev motor_0.125 "teacher dagger_v1" 500 ', "close_speed_mps": 0.125' '{"ratio": 0.45, "t_close": 0.265}'
BOXDIR=boxes METH="teacher dagger_v1" boxes
echo "== DAgger on the prototype hand $(date +%H:%M)"
[ -s runs/dagger_proto/model.zip ] || $PY scripts/dagger.py --name dagger_proto --procs $P --overrides "{$PR}" \
  > runs/dagger_proto.log 2>&1
ev nominal_dagger_proto "dagger_proto" 500 ""
ev heavy_T40_dagger_proto "dagger_proto" 500 ', "box_mass_kg": [0.4, 0.8], "t_max_n": 40'
ev camnoise020_dagger_proto "dagger_proto" 500 ', "cam_noise_m": 0.02'
ev latency100_dagger_proto "dagger_proto" 500 ', "cam_latency_s": 0.1'
BOXDIR=boxes_proto METH="dagger_proto" boxes
echo "== rules and supervisor on the prototype hand $(date +%H:%M)"
D='{"ratio":0.56,"absorb":0.52,"brake_time":0.25,"turn_gain":6.0,"motor_hold":0.83}'
[ -s runs/proto/eval_v2_proto.json ] || $PY scripts/eval_high2.py \
  --methods 'fixed:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}' "adaptive:$D" rl:hi2_cse \
  --conditions "nominal={$PR}" "heavy={$PR, \"box_mass_kg\": [0.4, 0.8]}" "slow_motor={$PR, \"close_speed_mps\": 0.13}" \
  "very_slow={$PR, \"close_speed_mps\": 0.10}" \
  --n 500 --seed0 500000 --procs $P --out runs/proto/eval_v2_proto.json 2>&1 | grep -v -i -E "warn|info"
echo "all done $(date +%H:%M)"
