#!/bin/bash
# 20 Oct 2026, 06:30-08:00, exploratory simulations, part 2 (after chain_2026-10-20_explore.sh): a second camera to cover
# the throwing area (behind the robot, left and right), and air drag on the parcel (the operator aims allowing for it).
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-explore; do sleep 20; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
C2='"cam2_pos_m": [-0.3, 0.6, 2.0]'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
CAMR='{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}'
mkdir -p runs/explore
ev() {
  [ -s runs/explore/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/explore/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
ev twocam_nominal "classical dagger_proto_cam" 500 ", $P, $C2"
ev twocam_wide "classical dagger_proto_widecam" 3000 ", $P, $C2, $W"
ev drag_nominal "teacher classical dagger_proto" 500 ', "air_cd": 1.05'
ev drag_light "teacher classical dagger_proto" 500 ', "air_cd": 1.05, "box_mass_kg": [0.05, 0.15]'
ev drag_wide "teacher classical dagger_proto_wide" 1000 ", \"air_cd\": 1.05, $W"
if ! [ -s runs/dagger_proto_wide2cam/model.zip ]; then
  echo "== DAgger dagger_proto_wide2cam $(date +%H:%M)"
  $PY scripts/dagger.py --name dagger_proto_wide2cam --procs $(procs) --teacher-episodes 1500 --episodes 600 --iters 15 \
    --randomize "$CAMR" --overrides "{$D, $P, $C2, $W}" > runs/dagger_proto_wide2cam.log 2>&1
  tail -1 runs/dagger_proto_wide2cam/iterations.jsonl | cut -c1-200
fi
ev twocam_wide_student2cam "dagger_proto_wide2cam" 3000 ", $P, $C2, $W"
echo "all done $(date +%H:%M)"
