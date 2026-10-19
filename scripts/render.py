"""Render a catching episode to MP4 (offscreen GLFW + ffmpeg) with the teacher or a saved policy.

Fixed side camera on the robot and the operator; slow motion factor --slow (frames are rendered every physics
sub-step group, output at 30 fps). Overlay: time, parcel speed, distance, motor command, phase.
Usage: MUJOCO_GL=glfw .venv/bin/python scripts/render.py --seed 0 --out videos/teacher_s0.mp4 [--policy runs/x]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catch.env import CatchEnv  # noqa: E402
from catch.teacher import Teacher  # noqa: E402

W, H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def it(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slow", type=float, default=4.0, help="slow-motion factor")
    ap.add_argument("--close", action="store_true", help="close-up on the hand instead of the whole scene")
    ap.add_argument("--frames-png", default=None, help="also save PNG frames at these times (comma list, s)")
    ap.add_argument("--overrides", default="{}", help="JSON parameter overrides (e.g. stronger hand, heavy parcels)")
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    import json
    env = CatchEnv(overrides=json.loads(a.overrides))
    obs, _ = env.reset(seed=a.seed)
    if a.policy:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        run = ROOT / a.policy
        model = PPO.load(run / "model", device="cpu")
        vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: CatchEnv()]))
        vn.training = False
        ctrl = lambda o: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]  # noqa: E731
        who = f"politica appresa {Path(a.policy).name} (solo sensori)"
    else:
        te = Teacher(env)
        ctrl = lambda o: te()  # noqa: E731
        who = "insegnante (stato vero)"
    m, d = env.m, env.d
    rend = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if a.close:
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 150.0, -20.0, 1.3, [0.5, 0.0, 1.1]
    else:
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 60.0, -10.0, 3.0, [1.1, 0.0, 1.0]
    opt = mujoco.MjvOption()
    fonts = (ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT_B, 22))
    fps = 30
    sim_per_frame = 1.0 / (fps * a.slow)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-movflags", "+faststart", "-map_metadata", "-1", a.out], stdin=subprocess.PIPE)
    pngs = [float(x) for x in a.frames_png.split(",")] if a.frames_png else []
    next_t, done_at, info, act = 0.0, None, {}, np.zeros(7)
    phase = "LANCIO"
    while True:
        act = ctrl(obs)
        obs, r, term, trunc, info = env.step(act)
        if act[6] > 0.5 and phase == "LANCIO":
            phase = "CHIUSURA"
        if info["inside"] and phase == "CHIUSURA":
            phase = "PACCO NELLA MANO"
        if (term or trunc) and done_at is None:
            done_at = env.t
            phase = "PRESO" if info["success"] else ("CADUTO" if info["dropped"] else "TEMPO SCADUTO")
        while next_t <= env.t:
            rend.update_scene(d, camera=cam, scene_option=opt)
            im = Image.fromarray(rend.render())
            dr = ImageDraw.Draw(im)
            pc, vc, _ = env.parcel_state()
            lines = [f"ERF2027 · UR5e + mano a quattro bracci · lancio {a.seed} · {who}",
                     f"t = {it(env.t)} s   (rallentato ×{it(a.slow, 0)})",
                     f"velocità del pacco {it(np.linalg.norm(vc), 1)} m/s · distanza dal palmo {it(info['dist'])} m",
                     f"pacco {it(100 * env.throw['sides'].max(), 1)} cm, {it(env.throw['mass'])} kg · motore {it(act[6])}"]
            y = 14
            for k, s in enumerate(lines):
                dr.text((18, y), s, font=fonts[1] if k == 0 else fonts[0], fill=(20, 20, 20))
                y += 30
            col = (0, 130, 0) if phase == "PRESO" else ((180, 0, 0) if phase in ("CADUTO", "TEMPO SCADUTO") else (30, 60, 160))
            dr.text((18, y + 8), phase, font=fonts[1], fill=col)
            ff.stdin.write(np.asarray(im).tobytes())
            for tp in list(pngs):
                if next_t >= tp:
                    im.save(Path(a.out).with_suffix(f".t{tp:.2f}.png"))
                    pngs.remove(tp)
            next_t += sim_per_frame
        if done_at is not None and env.t >= done_at + 0.4:
            break
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: episode {env.t:.2f} s, success {info['success']}, dropped {info['dropped']}, "
          f"max force {info['max_force']:.0f} N")


if __name__ == "__main__":
    main()
