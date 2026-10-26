#!/bin/bash
# Night of 19-20 Oct 2026, part 3 (after part 2): wide throws seen by the realistic camera. The student trained on wide
# throws with the ideal camera drops to 77.6 % with the realistic camera (runs/night/wide/wide_pinhole.json). Students
# trained on wide throws with the realistic camera (fixed placement; random placement) against the classical pipeline.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet erf-night2 || systemctl --user is-active --quiet erf-night; do sleep 30; done
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
RPOS='"cam_pos_random": [-0.6, 1.0, 0.5, 1.6, 1.4, 2.3]'
CAMR='{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}'
mkdir -p runs/night/widecam
ev() {
  [ -s runs/night/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/night/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
dag() {  # name, randomize, overrides
  [ -s runs/$1/model.zip ] && [ -s runs/$1/dataset.npz ] && return
  echo "== DAgger $1 $(date +%H:%M)"
  $PY scripts/dagger.py --name $1 --procs $(procs) --teacher-episodes 1500 --episodes 600 --iters 15 \
    --randomize "$2" --overrides "{$D$3}" > runs/$1.log 2>&1
  tail -1 runs/$1/iterations.jsonl | cut -c1-200
}
echo "start $(date +%H:%M)"
dag dagger_proto_widecam "$CAMR" ", $P, $W"
dag dagger_proto_all "$CAMR" ", $P, $W, $RPOS"
ev widecam/wide_pinhole "dagger_proto_widecam dagger_proto_all" 3000 ", $P, $W"
ev widecam/wide_randompos "classical dagger_proto_widecam dagger_proto_all" 3000 ", $P, $W, $RPOS"
ev widecam/nominal_pinhole "dagger_proto_widecam dagger_proto_all" 500 ", $P"
ev widecam/nominal_ideal "dagger_proto_all" 500 ""
ev widecam/wide_randompos_worst "classical dagger_proto_all" 3000 ", $P, $W, $RPOS, \"cam_calib_m\": 0.02, \"cam_calib_deg\": 2.0, \"cam_depth_k\": 0.005, \"cam_latency_s\": 0.066, \"cam_dropout\": 0.1"
echo "all done $(date +%H:%M)"
