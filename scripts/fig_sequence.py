"""Fig. 1: wide view of the scene at release and close-ups of a catch by the student policy (clean frames, no
overlay), Italian and English (decimal separator only).
Usage: MUJOCO_GL=glfw .venv/bin/python scripts/fig_sequence.py --seed 500003 --policy runs/dagger_v1
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
from catch.env import CatchEnv  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=500003)
    ap.add_argument("--policy", default="runs/dagger_v1")
    ap.add_argument("--times", default="0.0,0.55,0.62,0.68,0.9")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    run = ROOT / a.policy
    model = PPO.load(run / "model", device="cpu")
    vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: CatchEnv()]))
    vn.training = False
    env = CatchEnv()
    obs, _ = env.reset(seed=a.seed)
    times = [float(x) for x in a.times.split(",")]
    W, H = 960, 720
    rend = mujoco.Renderer(env.m, height=H, width=W)
    wide, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    for c in (wide, close):
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
    wide.azimuth, wide.elevation, wide.distance, wide.lookat[:] = 60.0, -10.0, 3.1, [1.15, 0.0, 1.0]
    close.azimuth, close.elevation, close.distance, close.lookat[:] = 150.0, -18.0, 0.95, [0.5, 0.0, 1.12]
    frames, k, info = [], 0, {}
    while k < len(times):
        if env.t + 1e-9 >= times[k]:
            rend.update_scene(env.d, camera=wide if k == 0 else close)
            frames.append((env.t, rend.render().copy()))
            k += 1
            continue
        a_ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
        obs, r, term, trunc, info = env.step(a_)
        if term or trunc:
            break
    while not info.get("success") and not info.get("dropped") and env.t < env.t_max:   # finish the episode
        a_ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)[0][0]
        obs, r, term, trunc, info = env.step(a_)
    meta = dict(seed=a.seed, policy=a.policy, times=[t for t, _ in frames], success=bool(info.get("success")),
                throw=dict(mass=float(env.throw["mass"]), sides_cm=[round(100 * s, 1) for s in env.throw["sides"]],
                           speed=float(np.linalg.norm(env.throw["v0"]))))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    (out / "fig_sequenza_manifest.json").write_text(json.dumps(meta, indent=1))
    for lang in ("it", "en"):
        fig = plt.figure(figsize=(12 / 2.54, 6.4 / 2.54))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.08, wspace=0.04)
        axes = [fig.add_subplot(gs[0, :])] + [fig.add_subplot(gs[1, i]) for i in range(4)]
        for i, (ax, (t, img)) in enumerate(zip(axes, frames)):
            if i == 0:
                img = img[90:630, :]
            else:
                img = img[60:660, 180:780]
            ax.imshow(img)
            ax.set_axis_off()
            ts = f"{t:.2f}".replace(".", ",") if lang == "it" else f"{t:.2f}"
            ax.text(0.02, 0.96, f"({'abcde'[i]}) t = {ts} s", transform=ax.transAxes, va="top", ha="left",
                    fontsize=7, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_sequenza_{lang}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
