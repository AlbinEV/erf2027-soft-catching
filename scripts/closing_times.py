"""Hand closing time versus motor stroke speed (method used for runs/closing_times.json, 17 Oct 2026).

For each stroke speed: the parcel is parked far from the robot and stopped, the model settles for 0.3 s, then the
motor is commanded at full force and the physics is stepped at 1 ms; the time at which the corner joints of every
arm reach 50 % and 95 % of their stops is recorded.
Output: printed; with --write also runs/closing_times.json  {speed: {"0.5": t50, "0.95": t95}}
Usage: .venv/bin/python scripts/closing_times.py [--write]
"""
import json
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catch.env import CatchEnv  # noqa: E402

SPEEDS = (0.25, 0.125, 0.0625, 0.03, 0.01)


def closing_time(v):
    env = CatchEnv(overrides=dict(close_speed_mps=v))
    m, d, cm, p = env.m, env.d, env.cm, env.p
    env.reset(seed=0)
    d.qpos[cm.parcel_q:cm.parcel_q + 3] = [3, 2, 0.1]          # parcel parked far from the robot
    d.qvel[cm.parcel_d:cm.parcel_d + 6] = 0
    for _ in range(300):                                        # settle for 0.3 s
        mujoco.mj_step(m, d)
    d.ctrl[cm.motor_act] = 1.0
    t = {}
    for i in range(1, 4000):
        mujoco.mj_step(m, d)
        c = env.corners().min()
        for f in (0.5, 0.95):
            if str(f) not in t and c > f:
                t[str(f)] = round(i * p.dt_s, 3)
        if "0.95" in t:
            break
    return t


if __name__ == "__main__":
    res = {str(v): closing_time(v) for v in SPEEDS}
    print(json.dumps(res, indent=1))
    if "--write" in sys.argv:
        (ROOT / "runs" / "closing_times.json").write_text(json.dumps(res, indent=1))
