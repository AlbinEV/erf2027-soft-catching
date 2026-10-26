"""Damper study for the compliant palm and wrist (19 Oct 2026), student policy dagger_v1.

Part 1 (selection throws, seeds 100000+): palm damping and stiffness sweep, to keep the palm plate off its end stop.
Part 2 (test throws, seeds 500000+): linear dampers of the model versus quadratic dampers F = c2 v|v| (orifice
dampers: hydraulic and RC oil shocks), calibrated to the same force at the typical stroke speed (wrist 80 N at
1 m/s, palm 19 N at 2.4 m/s). The quadratic force is applied as qfrc_applied at every 1 ms sub-step, with the
joint's linear damping set to zero. Per throw: success, parcel acceleration (10 ms mean), contact force, and for
palm and wrist peak stroke speed, peak damper force, stroke used, dissipated energy and end-stop hits.
Output: runs/dampers/<name>.json
Usage: .venv/bin/python scripts/damper_study.py [procs]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPL = dict(pad_stroke_m=0.05, housing_m=0.10, pad_c=8.0, wrist_stroke_m=0.15, wrist_c=80.0, wrist_k=150.0)
_M = {}


def policy():
    if "p" not in _M:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from catch.env import CatchEnv
        run = ROOT / "runs/dagger_v1"
        model = PPO.load(run / "model", device="cpu")
        vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: CatchEnv()]))
        vn.training = False
        _M["p"] = lambda o: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]
    return _M["p"]


def episode(job):
    seed, ov, quad = job                              # quad: {"pad": c2, "wrist": c2} (N s^2/m^2) or {}
    from catch.env import CatchEnv
    env = CatchEnv(overrides=dict(COMPL, **ov))
    m, p = env.m, env.p
    adr = {}
    for n in ("pad", "wrist"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        adr[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
    lin = {"pad": (p.pad_c, p.pad_k, p.pad_preload_m, p.pad_stroke_m),
           "wrist": (p.wrist_c, p.wrist_k, p.wrist_preload_m, p.wrist_stroke_m)}
    st = {n: dict(v=0.0, Fd=0.0, stroke=0.0, E=0.0, endstop=False) for n in adr}

    def hook(e):
        for n, (qa, da) in adr.items():
            c, k, pre, smax = lin[n]
            q, v = float(e.d.qpos[qa]), float(e.d.qvel[da])
            if n in quad:
                f = quad[n] * v * abs(v)
                e.d.qfrc_applied[da] = -f
            else:
                f = c * v
            s = st[n]
            s["v"] = max(s["v"], abs(v))
            s["Fd"] = max(s["Fd"], abs(f))
            s["stroke"] = max(s["stroke"], -q)
            s["E"] += abs(f * v) * p.dt_s
            s["endstop"] = s["endstop"] or (-q > smax - 0.002)

    obs, _ = env.reset(seed=seed)
    env.substep_hook = hook                          # after reset: reset clears the hook
    pol = policy()
    info = {}
    while True:
        obs, r, term, trunc, info = env.step(pol(obs))
        if term or trunc:
            break
    out = dict(seed=seed, success=bool(info["success"]), dropped=bool(info["dropped"]),
               touched_ur=bool(info["touched_ur"]), acc_g=float(info["max_acc_g"]), force=float(info["max_force"]))
    for n, s in st.items():
        out.update({f"{n}_{k}": (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in s.items()})
    return out


def summary(rows):
    ok = [r for r in rows if r["success"]]
    med = lambda k, rs=ok: float(np.median([r[k] for r in rs])) if rs else None  # noqa: E731
    p90 = lambda k, rs=ok: float(np.percentile([r[k] for r in rs], 90)) if rs else None  # noqa: E731
    return dict(n=len(rows), success=len(ok) / len(rows), touched_ur=sum(r["touched_ur"] for r in rows),
                acc_med=med("acc_g"), acc_p90=p90("acc_g"), force_med=med("force"),
                pad_endstop=sum(r["pad_endstop"] for r in rows), pad_stroke_max=max(r["pad_stroke"] for r in rows),
                pad_v_p90=p90("pad_v", rows), pad_Fd_p90=p90("pad_Fd", rows), pad_E_med=med("pad_E", rows),
                wrist_endstop=sum(r["wrist_endstop"] for r in rows), wrist_stroke_max=max(r["wrist_stroke"] for r in rows),
                wrist_v_p90=p90("wrist_v", rows), wrist_Fd_p90=p90("wrist_Fd", rows), wrist_E_med=med("wrist_E", rows))


CONFIGS = [  # name, seed0, n, overrides, quadratic dampers
    ("palm_c8_k100", 100000, 200, {}, {}),
    ("palm_c20_k100", 100000, 200, {"pad_c": 20.0}, {}),
    ("palm_c40_k100", 100000, 200, {"pad_c": 40.0}, {}),
    ("palm_c80_k100", 100000, 200, {"pad_c": 80.0}, {}),
    ("palm_c40_k300", 100000, 200, {"pad_c": 40.0, "pad_k": 300.0}, {}),
    ("test_linear", 500000, 300, {}, {}),
    ("test_quad_wrist", 500000, 300, {"wrist_c": 0.0}, {"wrist": 80.0}),
    ("test_quad_both", 500000, 300, {"wrist_c": 0.0, "pad_c": 0.0}, {"wrist": 80.0, "pad": 19.0 / 2.4 ** 2}),
]

if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = ROOT / "runs/dampers"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as pool:
        for name, seed0, n, ov, quad in CONFIGS:
            f = out / f"{name}.json"
            if f.exists():
                continue
            rows = pool.map(episode, [(seed0 + i, ov, quad) for i in range(n)], chunksize=4)
            s = summary(rows)
            f.write_text(json.dumps(dict(name=name, overrides=dict(COMPL, **ov), quadratic=quad, summary=s, rows=rows)))
            print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
