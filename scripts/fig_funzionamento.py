"""Fig. 1: how the catch works, Italian and English. The arm drawing is in
scripts/fig_braccio.py (Fig. 2).

(a) wide view of the simulated scene at release; (b)-(e) close-ups that follow the palm during a catch by
the student policy: waiting, closing starts, cage closed around the parcel, parcel secured. The phases are found from
the simulation (motor command, corner closure, held flag), not chosen by hand.
Usage: MUJOCO_GL=glfw .venv/bin/python scripts/fig_funzionamento.py --seed 500003 --policy runs/dagger_v1
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catch import kin  # noqa: E402
from catch.config import Params  # noqa: E402
from catch.env import CatchEnv  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
TXT = {"it": dict(scene="scena", arm="un braccio: aperto e in battuta", open="aperto", closed="chiuso",
                  cube="cubo di 10 cm", palm="palmo", phases=["attesa", "chiusura", "gabbia chiusa", "pacco bloccato"]),
       "en": dict(scene="scene", arm="one arm: open and at the stops", open="open", closed="closed",
                  cube="10 cm cube", palm="palm", phases=["waiting", "closing", "cage closed", "parcel secured"])}


def arm_polyline(p, closed):
    """Joint positions of one arm in its plane (x outward from the hand axis, z along it), metres."""
    phi = np.radians(p.mount_deg)                    # open arm leans outwards by the mount angle
    x, z, pts = p.root_radius_m, 0.0, [(p.root_radius_m, 0.0)]
    for L, stop in zip(p.pitch_m, p.stop_deg):
        if closed:
            phi -= np.radians(stop)                  # positive joint angles close the arm towards the axis
        x, z = x + L * np.sin(phi), z + L * np.cos(phi)
        pts.append((x, z))
    return np.array(pts)


def run(env, model, vn, seed, render_at=None, rend=None, cams=None):
    obs, _ = env.reset(seed=seed)
    log, frames, info, k = [], [], {}, 0
    while True:
        if render_at is not None and k < len(render_at) and env.t + 1e-9 >= render_at[k]:
            cam = cams["wide"] if k == 0 else cams["close"]
            if k > 0:
                P, Rh = kin.palm_pose(env.cm)
                cam.lookat[:] = P + 0.10 * Rh[:, 2]
            rend.update_scene(env.d, camera=cam)
            frames.append(rend.render().copy())
            k += 1
            continue
        a = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
        obs, r, term, trunc, info = env.step(a)
        log.append(dict(t=env.t, motor=float(a[6]), corners=float(env.corners().min()), held=bool(info["held"]),
                        dist=float(info["dist"])))
        if term or trunc:
            break
    return log, frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=500003)
    ap.add_argument("--policy", default="runs/dagger_v1")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    run_dir = ROOT / a.policy
    model = PPO.load(run_dir / "model", device="cpu")
    vn = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([lambda: CatchEnv()]))
    vn.training = False
    env = CatchEnv()
    log, _, info = run(env, model, vn, a.seed)
    t_close = next(e["t"] for e in log if e["motor"] > 0.5)
    t_cage = next(e["t"] for e in log if e["t"] > t_close and e["corners"] >= 0.8)
    t_held = next((e["t"] for e in log if e["held"]), log[-1]["t"])
    t_wait = t_close                                   # closing command just issued, arms still open
    t_mid = 0.5 * (t_close + t_cage)
    times = [0.0, t_wait, t_mid, t_cage, t_held + 0.25]
    W, H = 960, 720                                  # offscreen framebuffer of the model
    rend = mujoco.Renderer(env.m, height=H, width=W)
    wide, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    for c in (wide, close):
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
    wide.azimuth, wide.elevation, wide.distance, wide.lookat[:] = 60.0, -10.0, 3.1, [1.15, 0.0, 1.0]
    close.azimuth, close.elevation, close.distance = 150.0, -18.0, 0.85
    _, frames, info2 = run(env, model, vn, a.seed, render_at=times, rend=rend, cams=dict(wide=wide, close=close))
    p = Params()
    meta = dict(seed=a.seed, policy=a.policy, phase_times_s=dict(wait=t_wait, close=t_close, mid=t_mid, cage=t_cage,
                                                                 held=t_held, last=t_held + 0.25),
                success=bool(info.get("success")), same_run=bool(info.get("success")) == bool(info2.get("success")))
    out = ROOT / "figures"
    (out / "fig_funzionamento_manifest.json").write_text(json.dumps(meta, indent=1))
    for lang, T in TXT.items():
        fig = plt.figure(figsize=(12 / 2.54, 6.4 / 2.54))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.08, wspace=0.04)
        ax = fig.add_subplot(gs[0, :])
        ax.imshow(frames[0][140:560, :])
        ax.set_axis_off()
        ax.text(0.01, 0.96, f"(a) {T['scene']}", transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(frames[i + 1][60:660, 180:780])
            ax.set_axis_off()
            ax.text(0.03, 0.97, f"({'bcde'[i]}) {T['phases'][i]}", transform=ax.transAxes, va="top", fontsize=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_funzionamento_{lang}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
