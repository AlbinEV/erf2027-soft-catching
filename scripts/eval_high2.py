"""Matched evaluation of catch-parameter policies (v2) on fresh throws under several conditions.

Methods: "fixed:<json kwargs>", "adaptive:<json kwargs>" (ratio scaled with the encoder motor speed), "rl:<run>"
(runs/<run>/model_best.zip, sensors and decision mode from its config.json). Conditions: name=<json overrides>,
applied on top of the nominal parameters (no randomisation unless given with "randomize").
Usage: nice -n 19 .venv/bin/python scripts/eval_high2.py --methods 'fixed:{...}' rl:hi2_cam --conditions
       'nominal={}' 'heavy={"box_mass_kg": [0.4, 0.8]}' --n 500 --out runs/eval_high2.json
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
sys.path.insert(0, str(ROOT / "scripts"))
_CACHE = {}


def _policy_and_kw(method):
    import train_high2 as T
    kind, _, arg = method.partition(":")
    kw = dict(student_run="dagger_theta2", sensors=(), two_step=False)
    if kind == "fixed":
        return T.fixed_policy(**json.loads(arg)), kw
    if kind == "adaptive":
        return T.adaptive_rule(**json.loads(arg)), kw
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from catch.high_env2 import CatchParamEnv2
    run = ROOT / "runs" / arg
    cfg = json.loads((run / "config.json").read_text())
    kw = dict(student_run=cfg["student"], sensors=tuple(x for x in cfg["sensors"].split(",") if x),
              two_step=cfg["two_step"])
    model = PPO.load(run / "model_best", device="cpu")
    vn = VecNormalize.load(str(run / "vecnormalize_best.pkl"), DummyVecEnv([lambda: CatchParamEnv2(**kw)]))
    vn.training = False
    return (lambda o, env: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]), kw


def job(j):
    from catch.high_env2 import CatchParamEnv2
    import train_high2 as T
    method, cname, cond, seed = j
    key = (method, cname)
    if key not in _CACHE:
        _CACHE.clear()  # one environment per worker: caching all method-condition pairs exhausted the RAM (19 Oct)
        pol, kw = _policy_and_kw(method)
        _CACHE[key] = (pol, CatchParamEnv2(**kw, overrides=cond.get("overrides"), randomize=cond.get("randomize")))
    pol, env = _CACHE[key]
    info, r = T.episode(env, pol, seed)
    return dict(method=method, condition=cname, seed=seed, r=r, success=bool(info["success"]), acc=info["max_acc_g"],
                F=info["max_force"], tcp=info["max_tcp_acc"], u=info["u"], mass=float(env.env.throw["mass"]),
                speed_motor=float(env.env.p.close_speed_mps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["nominal={}"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--procs", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conds = {}
    for c in a.conditions:
        name, _, js = c.partition("=")
        d = json.loads(js)
        conds[name] = d if ("overrides" in d or "randomize" in d) else dict(overrides=d)
    jobs = [(m, cn, cd, a.seed0 + k) for m in a.methods for cn, cd in conds.items() for k in range(a.n)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs, chunksize=10)
    import train_high2 as T
    table = {}
    for cn in conds:
        for m in a.methods:
            s = T.summarise([r for r in rows if r["method"] == m and r["condition"] == cn])
            table[f"{cn} | {m}"] = s
            f = lambda x, nd=1: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
            print(f"{cn:12s} {m[:60]:60s} success {100 * s['success']:5.1f} %  reward {s['reward']:.3f}  acc "
                  f"{f(s['acc_median'])} g (p90 {f(s['acc_p90'])})  F {f(s['force_median'], 0)} N  palm "
                  f"{f(s['tcp_median'], 0)}", flush=True)
    Path(a.out).write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), conditions=conds, table=table,
                                           rows=rows, wall_min=round((time.time() - t0) / 60, 1)), indent=1))
    print(f"wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
