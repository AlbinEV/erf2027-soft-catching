#!/bin/bash
# 20 Oct 2026, exploratory simulations, part 4 (after part 3): is the robot the limit on the hardest throws? Teacher on
# the varied throws with a faster robot (TCP 2 and 3 m/s instead of 1.5), and a 24 cm aiming error.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-explore3; do sleep 20; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
ev() {
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
ev tcp_2.0_wide "teacher" 1000 ", \"tcp_v_max\": 2.0, $W"
ev tcp_3.0_wide "teacher" 1000 ", \"tcp_v_max\": 3.0, \"joint_v_max\": 6.28, $W"
ev aim024 "teacher classical dagger_proto dagger_proto_wide" 500 ', "aim_noise_m": 0.24'
ev aim024_tcp3 "teacher" 500 ', "aim_noise_m": 0.24, "tcp_v_max": 3.0, "joint_v_max": 6.28'
echo "all done $(date +%H:%M)"
