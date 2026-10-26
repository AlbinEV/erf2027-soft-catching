#!/bin/bash
# 20 Oct 2026, 06:00-08:00, exploratory simulations on the designed hand: heaviest parcels the hand can
# hold with 20 and 40 N per arm; slower robot (TCP speed limit); lower friction of parcel and hand pads.
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
mkdir -p runs/explore
ev() {
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
for T in 20 40; do
  ev mass_0.8-1.2_T$T "teacher dagger_proto" 500 ", \"box_mass_kg\": [0.8, 1.2], \"t_max_n\": $T"
  ev mass_1.2-1.6_T$T "teacher dagger_proto" 500 ", \"box_mass_kg\": [1.2, 1.6], \"t_max_n\": $T"
  ev mass_1.6-2.0_T$T "teacher dagger_proto" 500 ", \"box_mass_kg\": [1.6, 2.0], \"t_max_n\": $T"
done
for V in 1.0 0.75 0.5; do
  ev tcp_$V "teacher classical dagger_proto" 500 ", \"tcp_v_max\": $V"
  ev tcp_${V}_wide "teacher" 1000 ", \"tcp_v_max\": $V, $W"
done
ev box_friction_0.3 "teacher dagger_proto" 500 ", \"box_friction\": 0.3"
ev hand_friction_0.4 "teacher dagger_proto" 500 ", \"hand_friction\": 0.4"
ev friction_both_low "teacher dagger_proto" 500 ", \"box_friction\": 0.3, \"hand_friction\": 0.4"
echo "all done $(date +%H:%M)"
