"""Matched evaluation of semi-active damping policies on fresh throws under several conditions.

Methods: "fixed:<c_pad>,<c_wrist>" (constant damping, N s/m), "classical" (two-phase constant-force law: plate force
for the parcel to reach the hand speed within 80 % of the plate stroke, wrist force to stop parcel and hand within
80 % of the wrist stroke; camera approach speed at contact, encoders for the stroke speeds, nominal mass 0.25 kg), "rl:<run>" (runs/<run>/model_best.zip). Conditions: name=<json overrides of the catching env>.
Usage: nice -n 19 .venv/bin/python scripts/eval_semiactive.py --methods fixed:8,40 classical rl:sa_v1 --n 500
       --conditions 'nominal={}' --out runs/eval_sa.json
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
_C = {}
M_NOM = 0.25


def classical_policy():
    state = {}

    def pol(o, env):
        e, cm = env.env, env.env.cm
        if e.t_contact is None:
            state.clear()
            return 2 * env.default_c / env.cmax - 1
        if "F1" not in state:
            ax, vax, lat, sp = env._axial()
            v0 = max(abs(vax), 0.5)
            m_h = e.m.body_subtreemass[cm.hand_body]
            dv = v0 * m_h / (M_NOM + m_h)               # parcel speed change relative to the hand (pad phase)
            vc = v0 * M_NOM / (M_NOM + m_h)             # common speed of parcel and hand (wrist phase)
            state["F1"] = M_NOM * dv ** 2 / (2 * 0.8 * e.p.pad_stroke_m)
            state["F2"] = (M_NOM + m_h) * vc ** 2 / (2 * 0.8 * e.p.wrist_stroke_m)
        vp = abs(e.d.qvel[cm.comp_d["pad"]])
        vw = abs(e.d.qvel[cm.comp_d["wrist"]])
        c = np.array([state["F1"] / max(vp, 0.05), state["F2"] / max(vw, 0.05)])
        return 2 * np.clip(c, 0, env.cmax) / env.cmax - 1
    return pol


def get(method, cname, ov):
    key = (method, cname)
    if key in _C:
        return _C[key]
    _C.clear()  # one environment per worker: caching all method-condition pairs exhausted the RAM (19 Oct)
    from catch.semiactive_env import SemiActiveEnv
    kind, _, arg = method.partition(":")
    env = SemiActiveEnv(overrides=ov)
    if kind == "fixed":
        c = np.array([float(x) for x in arg.split(",")])
        pol = lambda o, env, u=2 * c / env.cmax - 1: u  # noqa: E731
    elif kind == "classical":
        pol = classical_policy()
    else:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        run = ROOT / "runs" / arg
        model = PPO.load(run / "model_best", device="cpu")
        vn = VecNormalize.load(str(run / "vecnormalize_best.pkl"), DummyVecEnv([lambda: SemiActiveEnv()]))
        vn.training = False
        pol = lambda o, env: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]  # noqa: E731
    _C[key] = (env, pol)
    return _C[key]


def job(j):
    method, cname, ov, seed = j
    env, pol = get(method, cname, ov)
    o, _ = env.reset(seed=seed)
    bottom = 0
    while True:
        o, r, te, tr, info = env.step(pol(o, env))
        bottom += env.bottoming() > 0
        if te or tr:
            break
    return dict(method=method, condition=cname, seed=seed, success=bool(info["success"]), acc=float(info["max_acc_g"]),
                F=float(info["max_force"]), bottom_steps=int(bottom), mass=float(env.env.throw["mass"]),
                v_rel=info.get("v_rel_contact"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["nominal={}"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--procs", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conds = {c.partition("=")[0]: json.loads(c.partition("=")[2]) for c in a.conditions}
    jobs = [(m, cn, cd, a.seed0 + k) for m in a.methods for cn, cd in conds.items() for k in range(a.n)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(job, jobs, chunksize=10)
    table = {}
    for cn in conds:
        for m in a.methods:
            r = [x for x in rows if x["method"] == m and x["condition"] == cn]
            ok = [x for x in r if x["success"]]
            s = dict(n=len(r), success=len(ok) / len(r),
                     acc_median=float(np.median([x["acc"] for x in ok])) if ok else None,
                     acc_p90=float(np.percentile([x["acc"] for x in ok], 90)) if ok else None,
                     force_median=float(np.median([x["F"] for x in ok])) if ok else None,
                     bottom_frac=float(np.mean([x["bottom_steps"] > 0 for x in r])))
            table[f"{cn} | {m}"] = s
            f = lambda x, nd=1: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
            print(f"{cn:10s} {m:22s} success {100 * s['success']:5.1f} %  acc median {f(s['acc_median'])} g "
                  f"(p90 {f(s['acc_p90'])})  F {f(s['force_median'], 0)} N  end of stroke {100 * s['bottom_frac']:.0f} %",
                  flush=True)
    Path(a.out).write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), conditions=conds, table=table,
                                           rows=rows, wall_min=round((time.time() - t0) / 60, 1)), indent=1))


if __name__ == "__main__":
    main()
