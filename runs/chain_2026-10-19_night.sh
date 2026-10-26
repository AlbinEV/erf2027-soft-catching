#!/bin/bash
# Night of 19-20 Oct 2026: three studies on the
# designed hand (dorsal stops at 0 deg, palm 5 cm 40 N s/m, wrist 10 cm 80 N s/m 150 N/m).
# 1) degraded actuation (motor slower, supply voltage lower, tendon elongated): teacher with the true closing time,
#    classical pipeline (nominal and true closing time), students, fixed rule, encoder rule and an RL supervisor, the
#    last three with an executor retrained on the degraded designed hand (closing-ratio range widened to 2.5);
# 2) realistic camera (position, field of view, occlusion, depth noise growing with distance, calibration error):
#    classical pipeline against students trained with the ideal and with the realistic camera;
# 3) wider throws (azimuth +-45 deg, 1-4 m, flight 0.3-1.0 s): teacher, classical, students (nominal and wide).
# 4) second training seeds, if time allows. Outputs: runs/night/<study>/, runs/<new runs>/
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
procs() { h=$(date +%-H); if [ $h -ge 20 ] || [ $h -lt 7 ]; then echo 18; else echo 14; fi; }
D='"outward_limit_deg": 0, "pad_stroke_m": 0.05, "housing_m": 0.10, "pad_c": 40, "wrist_stroke_m": 0.10, "wrist_c": 80, "wrist_k": 150'
mkdir -p runs/night/deg runs/night/cam runs/night/wide
log() { echo "== $* $(date +%H:%M) procs $(procs)"; }
ev() {  # dir/name, methods, n, extra overrides, [teacher json]
  [ -s runs/night/$1.json ] && return
  log "$1"
  $PY scripts/eval_policies.py --methods $2 --n $3 --seed0 500000 --procs $(procs) --overrides "{$D$4}" \
    --teacher "${5:-{\}}" --out runs/night/$1.json 2>&1 | grep -v -i -E "warn|info" | sed -E 's/ \| palm acc.*//'
}
dag() {  # name, randomize, extra overrides, teacher, teacher episodes, episodes, iters, seed
  [ -s runs/$1/model.zip ] && [ -s runs/$1/dataset.npz ] && return
  log "DAgger $1"
  $PY scripts/dagger.py --name $1 --procs $(procs) --randomize "$2" --overrides "{$D$3}" --teacher "$4" \
    --teacher-episodes $5 --episodes $6 --iters $7 --seed ${8:-0} > runs/$1.log 2>&1
  tail -2 runs/$1/iterations.jsonl | cut -c1-200
}
echo "start $(date +%H:%M)"

# ---------------- 1. degraded actuation ----------------
DEGR='"close_speed_mps": [0.03, 0.25], "motor_force_scale": [0.5, 1.0], "cable_stretch_m": [0.0, 0.004]'
declare -A C=(
  [nominal]=''
  [speed50]=', "close_speed_mps": 0.125'
  [speed25]=', "close_speed_mps": 0.0625'
  [speed12]=', "close_speed_mps": 0.03'
  [volt75]=', "motor_force_scale": 0.75'
  [volt50]=', "motor_force_scale": 0.5'
  [stretch2]=', "cable_stretch_m": 0.002'
  [stretch4]=', "cable_stretch_m": 0.004'
  [combo1]=', "close_speed_mps": 0.125, "motor_force_scale": 0.75, "cable_stretch_m": 0.002'
  [combo2]=', "close_speed_mps": 0.0625, "motor_force_scale": 0.75, "cable_stretch_m": 0.002'
)
CONDS="nominal speed50 speed25 speed12 volt75 volt50 stretch2 stretch4 combo1 combo2"
# methods that need no training first
for c in $CONDS; do
  ev deg/${c}_base "teacher dagger_proto" 500 "${C[$c]}" '{"t_close": null}'
  ev deg/${c}_classical "classical" 500 "${C[$c]}"
  ev deg/${c}_classical_true "classical" 500 "${C[$c]}" '{"t_close": null}'
done
# executor for the catch-parameter policies, retrained on the degraded designed hand, ratio range 0.35-2.5
export CATCH_RATIO_MAX=2.5
if ! [ -s runs/dagger_theta_deg/student_best.pt ]; then
  log "executor dagger_theta_deg"
  $PY scripts/dagger_theta2.py --name dagger_theta_deg --procs $(procs) --teacher-episodes 2000 --episodes 1000 \
    --iters 16 --randomize "{$DEGR, \"aim_noise_m\": [0.06, 0.2]}" \
    --overrides "{$D, \"box_mass_kg\": [0.1, 0.8], \"flight_s\": [0.45, 0.8]}" > runs/dagger_theta_deg.log 2>&1
  tail -2 runs/dagger_theta_deg/iterations.jsonl | cut -c1-200
fi
# RL supervisor (camera, parcel size, encoder) on the degraded designed hand
if ! [ -s runs/hi2_deg/model_best.zip ]; then
  log "PPO supervisor hi2_deg"
  $PY scripts/train_high2.py --name hi2_deg --sensors size,encoder --envs 16 --steps 100000 --eval-every 10000 \
    --student dagger_theta_deg --overrides "{$D, \"box_mass_kg\": [0.1, 0.8], \"flight_s\": [0.45, 0.8]}" \
    --randomize "{$DEGR, \"aim_noise_m\": [0.06, 0.2]}" > runs/hi2_deg.log 2>&1
  tail -2 runs/hi2_deg/evaluations.jsonl | cut -c1-200
fi
# student (sensors only) trained on the degraded hand with a teacher that knows the true closing time
dag dagger_proto_deg "{$DEGR}" "" '{"t_close": null}' 600 300 12
# evaluation: rules and supervisor with the new executor
export CATCH_EXECUTOR=dagger_theta_deg
CL=()
for c in $CONDS; do CL+=("$c={$D${C[$c]}}"); done
if ! [ -s runs/night/deg/eval_rules_rl.json ]; then
  log "rules and supervisor"
  $PY scripts/eval_high2.py --methods 'fixed:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}' \
    'adaptive:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}' rl:hi2_deg \
    --conditions "${CL[@]}" --n 500 --seed0 500000 --procs $(procs) --out runs/night/deg/eval_rules_rl.json 2>&1 | grep -v -i -E "warn|info"
fi
unset CATCH_EXECUTOR CATCH_RATIO_MAX
for c in $CONDS; do ev deg/${c}_student_deg "dagger_proto_deg" 500 "${C[$c]}"; done

# ---------------- 2. realistic camera ----------------
P='"cam_model": "pinhole"'
declare -A K=(
  [ideal]=''
  [pinhole]=", $P"
  [calib1]=", $P, \"cam_calib_m\": 0.01, \"cam_calib_deg\": 1.0"
  [calib2]=", $P, \"cam_calib_m\": 0.02, \"cam_calib_deg\": 2.0, \"cam_depth_k\": 0.005"
  [worst]=", $P, \"cam_calib_m\": 0.02, \"cam_calib_deg\": 2.0, \"cam_depth_k\": 0.005, \"cam_latency_s\": 0.066, \"cam_dropout\": 0.1"
  [side]=", $P, \"cam_pos_m\": [0.6, -1.4, 1.7]"
)
KCONDS="ideal pinhole calib1 calib2 worst side"
for k in $KCONDS; do ev cam/${k}_base "teacher classical dagger_proto" 500 "${K[$k]}"; done
dag dagger_proto_cam '{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}' \
  ", $P" '{}' 600 300 12
for k in $KCONDS; do ev cam/${k}_student_cam "dagger_proto_cam" 500 "${K[$k]}"; done

# ---------------- 3. wider throws ----------------
W=', "release_polar": [1.0, 4.0, -45.0, 45.0], "flight_s": [0.3, 1.0]'
ev wide/wide_base "teacher classical dagger_proto" 3000 "$W"
dag dagger_proto_wide '{}' "$W" '{}' 1500 600 15
ev wide/wide_student_wide "dagger_proto_wide" 3000 "$W"
ev wide/nominal_student_wide "dagger_proto_wide" 500 ""
ev wide/wide_pinhole "teacher classical dagger_proto dagger_proto_wide" 3000 "$W, $P"

# ---------------- 4. second seeds ----------------
dag dagger_proto_cam_s1 '{"cam_depth_k": [0.001, 0.006], "cam_calib_m": [0.0, 0.025], "cam_calib_deg": [0.0, 2.5], "cam_latency_s": [0.033, 0.07], "cam_dropout": [0.0, 0.12]}' \
  ", $P" '{}' 600 300 12 1
for k in pinhole calib2 worst; do ev cam/${k}_student_cam_s1 "dagger_proto_cam_s1" 500 "${K[$k]}"; done
dag dagger_proto_wide_s1 '{}' "$W" '{}' 1500 600 15 1
ev wide/wide_student_wide_s1 "dagger_proto_wide_s1" 3000 "$W"
export CATCH_RATIO_MAX=2.5
if ! [ -s runs/hi2_deg_s1/model_best.zip ]; then
  log "PPO supervisor hi2_deg_s1"
  $PY scripts/train_high2.py --name hi2_deg_s1 --seed 1 --sensors size,encoder --envs 16 --steps 100000 --eval-every 10000 \
    --student dagger_theta_deg --overrides "{$D, \"box_mass_kg\": [0.1, 0.8], \"flight_s\": [0.45, 0.8]}" \
    --randomize "{$DEGR, \"aim_noise_m\": [0.06, 0.2]}" > runs/hi2_deg_s1.log 2>&1
fi
export CATCH_EXECUTOR=dagger_theta_deg
if ! [ -s runs/night/deg/eval_rl_s1.json ]; then
  log "supervisor seed 1"
  $PY scripts/eval_high2.py --methods rl:hi2_deg_s1 --conditions "${CL[@]}" --n 500 --seed0 500000 --procs $(procs) \
    --out runs/night/deg/eval_rl_s1.json 2>&1 | grep -v -i -E "warn|info"
fi
echo "all done $(date +%H:%M)"
