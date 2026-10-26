#!/bin/bash
# Night of 19-20 Oct 2026, part 4 (the three studies are complete): reproducibility of the
# camera-placement result (second seed of dagger_proto_campos) and a longer training of the student that sees wide
# throws through a camera at random placements (dagger_proto_all_long: 3000 teacher throws, 1200 per iteration, 20).
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
P='"cam_model": "pinhole"'
W='"release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
RPOS='"cam_pos_random": [-0.6, 1.0, 0.5, 1.6, 1.4, 2.3]'
CAMR='{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}'
WORST='"cam_calib_m": 0.02, "cam_calib_deg": 2.0, "cam_depth_k": 0.005, "cam_latency_s": 0.066, "cam_dropout": 0.1'
ev() {
  [ -s runs/night/$1.json ] && return
  echo "== $1 $(date +%H:%M) procs $(procs)"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --out runs/night/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
echo "start $(date +%H:%M)"
if ! [ -s runs/dagger_proto_campos_s1/model.zip ]; then
  echo "== DAgger dagger_proto_campos_s1 $(date +%H:%M)"
  $PY scripts/dagger.py --name dagger_proto_campos_s1 --seed 1 --procs $(procs) --teacher-episodes 1000 --episodes 500 \
    --iters 14 --randomize "$CAMR" --overrides "{$D, $P, $RPOS}" > runs/dagger_proto_campos_s1.log 2>&1
  tail -1 runs/dagger_proto_campos_s1/iterations.jsonl | cut -c1-200
fi
ev campos/side_s1 "dagger_proto_campos_s1" 500 ", $P, \"cam_pos_m\": [0.6, -1.4, 1.7]"
ev campos/mirror_s1 "dagger_proto_campos_s1" 500 ", $P, \"cam_pos_m\": [0.6, 1.4, 1.7]"
ev campos/random_s1 "dagger_proto_campos_s1" 1000 ", $P, $RPOS"
ev campos/random_worst_s1 "dagger_proto_campos_s1" 1000 ", $P, $RPOS, $WORST"
if ! [ -s runs/dagger_proto_all_long/model.zip ]; then
  echo "== DAgger dagger_proto_all_long $(date +%H:%M)"
  $PY scripts/dagger.py --name dagger_proto_all_long --procs $(procs) --teacher-episodes 3000 --episodes 1200 --iters 20 \
    --randomize "$CAMR" --overrides "{$D, $P, $W, $RPOS}" > runs/dagger_proto_all_long.log 2>&1
  tail -1 runs/dagger_proto_all_long/iterations.jsonl | cut -c1-200
fi
ev widecam/wide_randompos_long "dagger_proto_all_long" 3000 ", $P, $W, $RPOS"
ev widecam/wide_randompos_worst_long "dagger_proto_all_long" 3000 ", $P, $W, $RPOS, $WORST"
ev widecam/nominal_pinhole_long "dagger_proto_all_long" 500 ", $P"
echo "all done $(date +%H:%M)"
