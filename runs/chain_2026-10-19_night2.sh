#!/bin/bash
# Night of 19-20 Oct 2026, part 2 (after runs/chain_2026-10-19_night.sh): learned policies drop when the camera is
# moved (runs/night/cam/side_*). A student trained with the camera placement drawn per throw (dagger_proto_campos),
# compared with the classical pipeline and the student trained at one placement, at the training placement, at the
# side placement, at its mirror image, at a placement outside the training range and at random placements.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-night; do sleep 30; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
RPOS='"cam_pos_random": [-0.6, 1.0, 0.5, 1.6, 1.4, 2.3]'
mkdir -p runs/night/campos
ev() {
  [ -s runs/night/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/night/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
if ! [ -s runs/dagger_proto_campos/model.zip ]; then
  echo "== DAgger dagger_proto_campos $(date +%H:%M)"
  $PY scripts/dagger.py --name dagger_proto_campos --procs $(procs) --teacher-episodes 1000 --episodes 500 --iters 14 \
    --randomize '{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}' \
    --overrides "{$D, $P, $RPOS}" > runs/dagger_proto_campos.log 2>&1
  tail -1 runs/dagger_proto_campos/iterations.jsonl | cut -c1-200
fi
M="classical dagger_proto_cam dagger_proto_campos"
ev campos/behind "$M" 500 ", $P"
ev campos/side "$M" 500 ", $P, \"cam_pos_m\": [0.6, -1.4, 1.7]"
ev campos/mirror "$M" 500 ", $P, \"cam_pos_m\": [0.6, 1.4, 1.7]"
ev campos/far_low "$M" 500 ", $P, \"cam_pos_m\": [1.5, -2.0, 1.2]"
ev campos/random "$M" 1000 ", $P, $RPOS"
ev campos/random_worst "$M" 1000 ", $P, $RPOS, \"cam_calib_m\": 0.02, \"cam_calib_deg\": 2.0, \"cam_depth_k\": 0.005, \"cam_latency_s\": 0.066, \"cam_dropout\": 0.1"
ev campos/ideal "dagger_proto_campos" 500 ""
echo "all done $(date +%H:%M)"
