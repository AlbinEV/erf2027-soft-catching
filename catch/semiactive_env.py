"""Semi-active compliance: an RL policy commands the variable dampers of the palm plate and of the wrist at 100 Hz.

Hand: compliant palm plate (5 cm stroke) and compliant wrist (15 cm), both with a light passive damper plus a variable
damper (MuJoCo damper actuators, dissipative only). The arm is driven by the nominal sensor-only student
(runs/dagger_v1, numpy weights); the RL policy acts from the moment the parcel is within `start_dist` of the palm
(camera estimate) to the end of the catch, before that both dampers stay at their default values.
Observation (14, sensors only): plate and wrist displacement and speed (encoders), parcel distance and approach speed
along the hand axis and lateral offset (camera/Kalman), wrist force sensor, contact flag and time since contact,
corner closure, current damping commands, parcel speed estimate.
Reward per step: -(a/10 g)^2 for the parcel acceleration over the step, -1 per step at the end of a stroke
(bottoming); +5 for a successful catch, -5 otherwise.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import kin
from .env import CatchEnv
from .high_env import student_act

ROOT = Path(__file__).resolve().parents[1]
COMPLIANT = dict(pad_stroke_m=0.05, housing_m=0.10, wrist_stroke_m=0.15, wrist_k=150.0, pad_c=1.0, wrist_c=2.0,
                 semi_active=True, pad_c_max=200.0, wrist_c_max=400.0)
G = np.array([0.0, 0.0, -9.81])


def load_student_v1(run="dagger_v1"):
    import torch
    p = ROOT / "runs" / run
    sd = torch.load(p / "student_best.pt", map_location="cpu")
    ds = np.load(p / "dataset.npz")
    W = {k: sd[n].numpy().astype(np.float64) for k, n in (("w0", "0.weight"), ("b0", "0.bias"), ("w1", "2.weight"),
                                                           ("b1", "2.bias"), ("w2", "4.weight"), ("b2", "4.bias"))}
    W["mean"], W["std"] = ds["mean"], np.sqrt(ds["var"] + 1e-8)
    return W


class SemiActiveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, student_run="dagger_v1", overrides=None, randomize=None, start_dist=0.4, default_c=(8.0, 40.0),
                 w_acc=1.0, w_bottom=1.0, r_success=5.0, r_fail=5.0):
        self.env = CatchEnv(overrides=dict(COMPLIANT, **(overrides or {})), randomize=randomize)
        self.W = load_student_v1(student_run)
        self.start_dist, self.default_c = start_dist, np.asarray(default_c, float)
        self.w_acc, self.w_bottom, self.r_success, self.r_fail = w_acc, w_bottom, r_success, r_fail
        self.cmax = np.array([self.env.p.pad_c_max, self.env.p.wrist_c_max])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (14,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def _axial(self):
        e = self.env
        est = e.kf.estimate(e.t)
        P, Rh = kin.palm_pose(e.cm)
        if est is None:
            return 1.0, 0.0, 0.0, 0.0
        rel, vc = est[0] - P, est[1]
        vp, _ = e.palm_velocity()
        z = Rh[:, 2]
        ax = float(rel @ z)
        lat = float(np.linalg.norm(rel - ax * z))
        return ax, float(-(vc - vp) @ z), lat, float(np.linalg.norm(vc))

    def _obs(self):
        e, cm = self.env, self.env.cm
        d = e.d
        qp, qw = d.qpos[cm.comp_q["pad"]], d.qpos[cm.comp_q["wrist"]]
        vp, vw = d.qvel[cm.comp_d["pad"]], d.qvel[cm.comp_d["wrist"]]
        ax, vax, lat, sp = self._axial()
        f = e.f_hist[-1] if e.f_hist else 0.0
        tc = 0.0 if e.t_contact is None else e.t - e.t_contact
        return np.array([qp / 0.05, vp / 3, qw / 0.15, vw / 3, ax, vax / 4, lat * 10, f / 50,
                         float(e.t_contact is not None), tc * 10, e.corners().min(), *(e.damp_cmd / self.cmax),
                         sp / 4], np.float32)

    def _arm_step(self):
        e = self.env
        vc0 = e.d.qvel[e.cm.parcel_d:e.cm.parcel_d + 3].copy()
        self.obs_arm, _, term, trunc, self.info = e.step(student_act(self.W, self.obs_arm))
        acc = np.linalg.norm((e.d.qvel[e.cm.parcel_d:e.cm.parcel_d + 3] - vc0) / e.dt_c - G) / 9.81
        return acc, term or trunc

    def bottoming(self):
        e, cm = self.env, self.env.cm
        return float(e.d.qpos[cm.comp_q["pad"]] <= -e.p.pad_stroke_m + 1e-3) + \
            float(e.d.qpos[cm.comp_q["wrist"]] <= -e.p.wrist_stroke_m + 1e-3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        e = self.env
        self.obs_arm, _ = e.reset(seed=int(self.np_random.integers(2 ** 31)) if seed is None else seed)
        e.damp_cmd = self.default_c.copy()
        self.done, self.info = False, {}
        while not self.done:
            ax, *_ = self._axial()
            if e.t_contact is not None or (e.kf.estimate(e.t) is not None and ax < self.start_dist):
                break
            _, self.done = self._arm_step()
        return self._obs(), {}

    def step(self, action):
        e = self.env
        e.damp_cmd = (np.clip(np.asarray(action, float), -1, 1) + 1) / 2 * self.cmax
        r = 0.0
        if not self.done:
            acc, self.done = self._arm_step()
            r -= self.w_acc * (acc / 10.0) ** 2 + self.w_bottom * self.bottoming()
        if self.done:
            r += self.r_success if self.info.get("success") else -self.r_fail
        return self._obs(), float(r), bool(self.done), False, dict(self.info)
