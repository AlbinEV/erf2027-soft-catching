#!/bin/bash
# Night batch of 17-18 Oct 2026 (author: full CPU at night). Runs sequentially with P processes.
cd "$(dirname "$0")/.."
P=${1:-20}
log() { echo "$(date +%H:%M) $*"; }
# 1. training variability of the student: two more DAgger seeds
for s in 1 2; do
  log "dagger seed $s"
  nice -n 19 .venv/bin/python scripts/dagger.py --name dagger_s$s --seed $s --procs $P --iters 12 > runs/dagger_s$s.log 2>&1
done
# 2. student trained with randomized conditions (domain randomization)
RND='{"aim_noise_m": [0.06, 0.24], "cam_noise_m": [0.005, 0.02], "cam_latency_s": [0.033, 0.1], "cam_rate_hz": [30, 60], "spin_max": [0, 15]}'
OV='{"box_mass_kg": [0.1, 0.8], "flight_s": [0.4, 0.8]}'
log "dagger randomized"
nice -n 19 .venv/bin/python scripts/dagger.py --name dagger_dr --procs $P --iters 15 --randomize "$RND" --overrides "$OV" > runs/dagger_dr.log 2>&1
# 3. base evaluation of the new students on the same 500 test throws
log "eval base"
nice -n 19 .venv/bin/python scripts/eval_policies.py --methods dagger_s1 dagger_s2 dagger_dr --n 500 --seed0 500000 \
    --procs $P --out runs/eval_students_500000.json 2>&1 | grep -v -i -E "warn|Info"
# 4. robustness sweeps for the randomized student
log "sweeps dagger_dr"
scripts/run_sweeps.sh $P _dr dagger_dr
# 5. closing ratio and motor speed (teacher, 500 throws)
log "ratio and motor"
for r in 0.1 0.2 0.3 0.45 0.6 0.71 0.9; do
  nice -n 19 .venv/bin/python scripts/eval_teacher.py --n 500 --seed0 500000 --procs $P \
      --kw "{\"teacher\": {\"ratio\": $r}}" --out runs/sweeps/ratio_$r.json 2>&1 | grep -v -i warn | sed "s/^/[ratio $r] /"
done
for vt in "0.125 0.265" "0.0625 0.409" "0.03 0.78" "0.01 2.189"; do
  set -- $vt
  for r in 0.3 0.45 0.6; do
    nice -n 19 .venv/bin/python scripts/eval_teacher.py --n 500 --seed0 500000 --procs $P \
        --kw "{\"teacher\": {\"ratio\": $r, \"t_close\": $2}, \"overrides\": {\"close_speed_mps\": $1}}" \
        --out runs/sweeps/motor_$1_ratio_$r.json 2>&1 | grep -v -i warn | sed "s/^/[motor $1 ratio $r] /"
  done
done
log "done"
