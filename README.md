# Catching thrown parcels with a compliant underactuated hand (simulation)

Code, trained policies and simulation results of a preliminary study submitted to the ERF 2027 Scientific Track:
a UR5e robot carrying a four-arm, single-motor, tendon-driven hand with a preset closed shape catches parcels
thrown by an operator. **Everything here is simulation (MuJoCo)**; no hardware results are reported.

The study derives design requirements for the hand (closing time, closing instant, tendon tension, compliance of
palm and wrist, range of parcel sizes), compares a camera-only student policy (DAgger from a privileged teacher)
with a classical Kalman-filter pipeline, and evaluates reinforcement learning (PPO) both for fine-tuning the student's
commands and as a supervisor that chooses the catch parameters for each throw.

## Contents

| Path | What it is |
|---|---|
| `catch/` | Simulation: hand and scene model (`model.py`), Gymnasium environment (`env.py`), parameters (`config.py`), privileged teacher (`teacher.py`), ballistic Kalman filter (`perception.py`), catch-parameter environments for RL (`high_env2.py`, `theta2.py`) |
| `scripts/` | Training (DAgger, PPO), evaluation, statistics and figure scripts |
| `runs/` | Trained policies, training curves and all evaluation results (JSON), plus the shell scripts used to launch the campaigns |
| `figures/` | Figures of the paper (Italian and English labels) |
| `requirements.txt` | Exact package versions used |

Not included: the DAgger training datasets (about 0.5 GB) and TensorBoard logs; they can be regenerated with the
training commands below.

## Installation

Python 3.12 on Linux, CPU only. `ffmpeg` is needed only to render videos.

```bash
git clone https://github.com/AlbinEV/erf2027-soft-catching.git
cd erf2027-soft-catching
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/get_menagerie.sh          # UR5e model of MuJoCo Menagerie (BSD-3-Clause), fixed commit
```

Quick check (10 test throws, teacher and camera-only student; about 1 minute):

```bash
.venv/bin/python scripts/eval_policies.py --methods teacher dagger_v1 --n 10 --seed0 500000 --procs 2 --out /tmp/check.json
```

The simulation is deterministic: every result below can be recomputed exactly from its seeds.

## Evaluation protocol

All methods are compared on the same throws. Test throws use seeds 500000-500499 (500 throws) and are never used
for training or model selection; selection uses seeds 100000+. Percentages have Wilson 95 % intervals; differences
between methods are paired on the same throws (95 % interval and exact McNemar test, `scripts/paired_stats.py`).

## Reproducing the results

`P` is the number of parallel processes. Each run directory stores the exact arguments in `config.json`.

| Result | Command | Output |
|---|---|---|
| Hand closing time versus motor speed | `scripts/closing_times.py --write` | `runs/closing_times.json` |
| Closing ratio and closing time (teacher) | `scripts/eval_teacher.py --n 500 --seed0 500000 --kw '{"teacher": {"ratio": R}}'`; motor speeds with `--kw '{"teacher": {"ratio": R, "t_close": T}, "overrides": {"close_speed_mps": V}}'` | `runs/sweeps/ratio_*.json`, `runs/sweeps/motor_*_ratio_*.json` |
| Tendon tension versus parcel mass (teacher) | `bash scripts/run_mass_study.sh P` | `runs/sweeps/mass_T{20,40,80}.json` |
| Rigid and compliant hand (teacher, classical pipeline, student) | see `runs/chain_2026-10-19_day.sh` (first command) and `scripts/eval_policies.py --methods teacher classical dagger_v1 --n 500 --seed0 500000` | `runs/eval_rigid_500000.json`, `runs/eval_compliant_500000.json` |
| Damping of the compliant wrist and palm | `bash runs/compliance_grid.sh P` | `runs/theta/cg_*.json` |
| Fraction of the parcel velocity followed by the palm | `bash runs/absorb_study.sh P` | `runs/absorb/` |
| Parcel size and shape | `bash runs/box_study.sh P` | `runs/boxes/` |
| Robustness (one condition at a time) | `bash scripts/run_sweeps.sh P` | `runs/sweeps/{aim*,camnoise*,latency*,camrate30,fast_throws,heavy_parcels,small_parcels,spin15}.json` |
| Camera-only student (DAgger) and training seeds | `scripts/dagger.py --name dagger_v1 --procs P` (seeds: `--seed 1`, `--seed 2`) | `runs/dagger_v1/`, `runs/dagger_s1/`, `runs/dagger_s2/`, `runs/eval_base_500000.json`, `runs/eval_students_500000.json` |
| PPO fine-tuning of the student (negative result) | `scripts/train_ft.py` with the arguments in `runs/ft_v*/config.json` | `runs/ft_v1` … `runs/ft_v5`, `runs/eval_ft_*.json` |
| Parameter-conditioned executor | `scripts/dagger_theta2.py` with the arguments in `runs/dagger_theta2/config.json` | `runs/dagger_theta2/` |
| RL supervisors of the catch parameters | `scripts/train_high2.py --name hi2_cse --sensors size,encoder --envs 6` (camera only: `--sensors ""`; with wrist force sensor: `--sensors size,encoder,ft --two-step`; other seeds: `--seed 1/2/3`) | `runs/hi2_*` |
| Matched comparison of rules and supervisors | `bash runs/eval_v2_final.sh P`; seeds: `runs/chain_2026-10-19_day.sh`, `runs/chain_2026-10-19_ft.sh` | `runs/eval_v2_final.json`, `runs/eval_v2_seed1.json`, `runs/eval_v2_seed23.json` |
| Rule distilled from the supervisor | `bash runs/distilled_rule.sh P` | `runs/eval_v2_distilled.json` |
| Paired differences | `scripts/paired_stats.py runs/eval_v2_final.json --a rl:hi2_cse --b 'fixed:{"ratio":0.45'` | printed table |
| Figures | `scripts/paper_figures.py`, `scripts/fig_braccio.py`, `scripts/fig_funzionamento.py` (needs an OpenGL display, `MUJOCO_GL=glfw`) | `figures/` |

Run directories and results that are not cited above (for example `runs/sa_v1`, `runs/hi2_soft`, `runs/dagger_T40`)
are exploratory runs kept for completeness.

Approximate run times on a 16-core laptop CPU: a 500-throw evaluation of one method takes about 1 minute with 12
processes; the camera-only student about 5 minutes of DAgger with 13 processes; one RL supervisor about 30-50 minutes
with 6 environments.

## Third-party components

The UR5e model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause,
commit `62ab00f`) and is downloaded by `scripts/get_menagerie.sh`, not redistributed here.
