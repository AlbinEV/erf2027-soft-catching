"""Matched evaluation on fresh throws: teacher (true state), classical pipeline (teacher logic on the Kalman filter)
and saved policies (sensors only), optionally under modified conditions (parameter overrides).

Policies: "teacher", "classical", or a run folder in runs/ (model.zip + vecnormalize.pkl, or with ":best"
model_best.zip + vecnormalize_best.pkl). Every method plays the same throws (seeds); deterministic actions.
Usage: nice -n 19 .venv/bin/python scripts/eval_policies.py --methods teacher classical dagger_v1 --n 500
       --seed0 500000 --overrides '{"aim_noise_m": 0.18}' --out runs/eval_x.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_POL = {}


def _init(specs):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from catch.env import CatchEnv
    for spec in specs:
        if spec in ("teacher", "classical"):
            continue
        run, _, which = spec.partition(":")
        suf = "_best" if which == "best" else ""
        p = ROOT / "runs" / run
        vn = VecNormalize.load(str(p / f"vecnormalize{suf}.pkl"), DummyVecEnv([lambda: CatchEnv()]))
        vn.training = False
        _POL[spec] = (PPO.load(p / f"model{suf}", device="cpu"), vn)


def episode(job):
    from catch.env import CatchEnv
    from catch.teacher import Teacher
    spec, seed, overrides, tkw = job
    env = CatchEnv(overrides=overrides)
    obs, _ = env.reset(seed=seed)
    te = (Teacher(env, source="kf" if spec == "classical" else "true", **tkw)
          if spec in ("teacher", "classical") else None)
    while True:
        if te is not None:
            a = te()
        else:
            model, vn = _POL[spec]
            a = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    th = env.throw
    aim0 = np.array([0.45, 0.0, env.p.base_height_m + env.p.aim_dz])
    return dict(method=spec, seed=seed, success=bool(info["success"]), dropped=bool(info["dropped"]),
                inside=bool(env.was_inside), max_force=float(info["max_force"]), touched_ur=bool(info["touched_ur"]),
                max_acc_g=float(info["max_acc_g"]), max_tcp_acc=float(info["max_tcp_acc"]),
                t=float(info["t"]), aim_err=float(np.linalg.norm(th["aim"] - aim0)), speed=float(np.linalg.norm(th["v0"])),
                mass=float(th["mass"]), side=float(np.max(th["sides"])), T=float(th["T"]),
                release_r=float(np.linalg.norm(th["p0"][:2] - aim0[:2])),
                release_az=float(np.degrees(np.arctan2(th["p0"][1] - aim0[1], th["p0"][0] - aim0[0]))),
                frames_lost=int(getattr(env, "frames_lost", 0)), frames=int(getattr(env, "frames_total", 0)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--teacher", default="{}", help="JSON options of the teacher and classical pipeline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ov = json.loads(a.overrides)
    jobs = [(m, a.seed0 + k, ov, json.loads(a.teacher)) for m in a.methods for k in range(a.n)]
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.methods,)) as pool:
        res = pool.map(episode, jobs, chunksize=8)
    table = {}
    for m in a.methods:
        r = [x for x in res if x["method"] == m]
        ok = [x for x in r if x["success"]]
        lo, hi = wilson(len(ok), len(r))
        table[m] = dict(n=len(r), success=len(ok), ci95=[round(lo, 3), round(hi, 3)],
                        dropped=sum(x["dropped"] for x in r), touched_ur=sum(x["touched_ur"] for x in r),
                        median_force_ok=float(np.median([x["max_force"] for x in ok])) if ok else None,
                        p90_force_ok=float(np.percentile([x["max_force"] for x in ok], 90)) if ok else None,
                        median_acc_g_ok=float(np.median([x["max_acc_g"] for x in ok])) if ok else None,
                        p90_acc_g_ok=float(np.percentile([x["max_acc_g"] for x in ok], 90)) if ok else None,
                        median_tcp_acc_ok=float(np.median([x["max_tcp_acc"] for x in ok])) if ok else None)
        print(f"{m:22s} success {len(ok):4d}/{len(r)} ({100 * len(ok) / len(r):.1f} %, 95 % CI {100 * lo:.1f}-{100 * hi:.1f}) "
              f"dropped {table[m]['dropped']} touched UR {table[m]['touched_ur']} "
              f"force median {table[m]['median_force_ok']} p90 {table[m]['p90_force_ok']} | parcel acc median "
              f"{table[m]['median_acc_g_ok']} g p90 {table[m]['p90_acc_g_ok']} | palm acc {table[m]['median_tcp_acc_ok']}",
              flush=True)
    Path(a.out).write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), overrides=ov,
                                           seeds=[a.seed0, a.seed0 + a.n - 1], table=table, episodes=res,
                                           wall_min=round((time.time() - t0) / 60, 1)), indent=1))
    print(f"wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
