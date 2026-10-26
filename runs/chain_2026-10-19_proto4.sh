#!/bin/bash
# 19 Oct 2026, last part of the hour: why the prototype hand tolerates slow closing (proto2: 99.8 % at 0.41 s against
# 47.8 % of the rigid hand in runs/sweeps). Teacher only, ratio 0.45, closing at 95 % of the stops in 0.41, 0.78 and
# 2.19 s, five hands: rigid with dorsal stops at 30 deg (paper) or 0 deg, compliant hand of the paper (palm 8 N s/m,
# wrist 15 cm), prototype dampers with dorsal stops at 30 deg, prototype. Starts after proto3. Outputs: runs/proto4/
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-proto3; do sleep 20; done
PY=".venv/bin/python"
P=12
mkdir -p runs/proto4
declare -A HAND=(
  [rigid30]=''
  [rigid0]='"outward_limit_deg": 0'
  [compl30]='"pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 8, "wrist_stroke_m": 0.15, "wrist_c": 80, "wrist_k": 150'
  [protodamp30]='"pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
  [proto]='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
)
declare -A TCLOSE=([0.0625]=0.409 [0.03]=0.78 [0.01]=2.189)
echo "start $(date +%H:%M)"
for h in rigid30 rigid0 compl30 protodamp30 proto; do
  for v in 0.0625 0.03 0.01; do
    out=runs/proto4/${h}_motor$v.json
    [ -s $out ] && continue
    ov=${HAND[$h]}
    [ -n "$ov" ] && ov="$ov, "
    echo "== $h motor $v $(date +%H:%M)"
    $PY scripts/eval_policies.py --methods teacher --n 500 --seed0 500000 --procs $P \
      --overrides "{$ov\"close_speed_mps\": $v}" --teacher "{\"ratio\": 0.45, \"t_close\": ${TCLOSE[$v]}}" \
      --out $out 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
  done
done
echo "all done $(date +%H:%M)"
