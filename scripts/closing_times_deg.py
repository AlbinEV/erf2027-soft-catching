"""Closing time of the designed hand with degraded actuation (19 Oct 2026), same method as scripts/closing_times.py.

Grid: motor stroke speed x motor force scale (force and no-load speed together, e.g. a supply-voltage drop) x tendon
elongation (taken up by the motor before it pulls). The parcel is parked far away, the model settles for 0.3 s, then the motor pulls at full command for up to 4 s;
the time at which the corner joints of every arm reach 50 % and 95 % of their stops is recorded (null if never), with
the largest corner fraction reached.
Output: runs/closing_times_deg.json (read by catch/env.py closing_time_true)
Usage: .venv/bin/python scripts/closing_times_deg.py [procs]
"""
import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DESIGNED = dict(outward_limit_deg=0, pad_stroke_m=0.05, housing_m=0.10, pad_c=40, wrist_stroke_m=0.10, wrist_c=80,
                wrist_k=150)
SPEEDS = (0.03, 0.0625, 0.125, 0.25)
SCALES = (0.5, 0.75, 1.0)
STRETCHES = (0.0, 0.001, 0.002, 0.003, 0.004)


def measure(job):
    from catch.env import CatchEnv
    v, s, st = job
    env = CatchEnv(overrides=dict(DESIGNED, close_speed_mps=v, motor_force_scale=s, cable_stretch_m=st))
    m, d, cm, p = env.m, env.d, env.cm, env.p
    env.reset(seed=0)
    d.qpos[cm.parcel_q:cm.parcel_q + 3] = [3, 2, 0.1]
    d.qvel[cm.parcel_d:cm.parcel_d + 6] = 0
    for _ in range(300):
        mujoco.mj_step(m, d)
    d.ctrl[cm.motor_act] = 1.0
    t, cmax, hist = {}, 0.0, []
    for i in range(1, 4000):
        mujoco.mj_step(m, d)
        c = float(env.corners().min())
        cmax = max(cmax, c)
        hist.append(c)
        for f in (0.5, 0.95):
            if str(f) not in t and c > f:
                t[str(f)] = round(i * p.dt_s, 3)
        if "0.95" in t:
            break
    # effective closing time: 95 % of the stops, or 95 % of the largest closure when the stops cannot be reached
    t_eff = t.get("0.95") or next(round((i + 1) * p.dt_s, 3) for i, c in enumerate(hist) if c >= 0.95 * cmax)
    return dict(v=v, s=s, stretch=st, t50=t.get("0.5"), t95=t.get("0.95"), t_eff=t_eff, corner_max=round(cmax, 3))


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = list(product(SPEEDS, SCALES, STRETCHES))
    with Pool(procs) as pool:
        rows = pool.map(measure, jobs)
    t95 = [[[None] * len(STRETCHES) for _ in SCALES] for _ in SPEEDS]
    for r in rows:
        t95[SPEEDS.index(r["v"])][SCALES.index(r["s"])][STRETCHES.index(r["stretch"])] = r["t_eff"]
        print(json.dumps(r), flush=True)
    out = dict(hand=DESIGNED, speeds=SPEEDS, scales=SCALES, stretches=STRETCHES, t95=t95, rows=rows)
    (ROOT / "runs" / "closing_times_deg.json").write_text(json.dumps(out, indent=1))
