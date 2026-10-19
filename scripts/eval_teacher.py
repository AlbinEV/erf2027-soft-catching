"""Evaluate the teacher (true state) on N throws in parallel; prints success and failure categories.
Usage: .venv/bin/python scripts/eval_teacher.py --n 200 --seed0 0 --procs 12
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(job):
    from catch.env import CatchEnv
    from catch.teacher import Teacher
    seed, kw = job
    env = CatchEnv(overrides=kw.get("overrides"))
    env.reset(seed=seed)
    te = Teacher(env, **kw.get("teacher", {}))
    while True:
        _, _, term, trunc, info = env.step(te())
        if term or trunc:
            break
    th = env.throw
    return dict(seed=seed, success=bool(info["success"]), dropped=bool(info["dropped"]), inside=bool(env.was_inside),
                triggered=bool(te.triggered), max_force=float(info["max_force"]), touched_ur=bool(info["touched_ur"]),
                max_acc_g=float(info["max_acc_g"]), max_tcp_acc=float(info["max_tcp_acc"]),
                v_rel_contact=None if info["v_rel_contact"] is None else float(info["v_rel_contact"]),
                speed=float(np.linalg.norm(th["v0"])), T=float(th["T"]), mass=float(th["mass"]),
                side=float(np.max(th["sides"])), min_lateral=float(te.min_lateral),
                aim_err=float(np.linalg.norm(th["aim"] - np.array([0.45, 0.0, env.p.base_height_m + env.p.aim_dz]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--kw", default="{}", help="JSON: {'teacher': {...}, 'overrides': {...}}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kw = json.loads(a.kw)
    with Pool(a.procs) as pool:
        res = pool.map(run, [(a.seed0 + k, kw) for k in range(a.n)], chunksize=4)
    ok = [r for r in res if r["success"]]
    n = len(res)
    print(f"success {len(ok)}/{n} | not triggered {sum(not r['triggered'] for r in res)} | inside but failed "
          f"{sum(r['inside'] and not r['success'] for r in res)} | triggered never inside "
          f"{sum(r['triggered'] and not r['inside'] for r in res)} | touched UR {sum(r['touched_ur'] for r in res)}")
    if ok:
        f = [r["max_force"] for r in ok]
        print(f"max contact force on success: median {np.median(f):.0f} N, p90 {np.percentile(f, 90):.0f} N")
        ag = [r["max_acc_g"] for r in ok]
        at = [r["max_tcp_acc"] for r in ok]
        print(f"parcel acceleration on success: median {np.median(ag):.1f} g, p90 {np.percentile(ag, 90):.1f} g | "
              f"palm acceleration: median {np.median(at):.0f} m/s2, p90 {np.percentile(at, 90):.0f} m/s2")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(kw=kw, results=res), indent=1))


if __name__ == "__main__":
    main()
