"""High-level environment: one step per throw. The RL policy sees the camera-based context shortly after release and
chooses the catch parameters (catch/theta.py); a low-level executor then performs the whole catch.

Context (13 values, sensors only), at decision time t_d after release: Kalman-filter parcel position relative to the
palm and velocity, their standard deviations, predicted arrival (ballistic time to the palm plane), palm position,
t_d. Executors: "student" (parameter-conditioned DAgger student, camera only; numpy weights from runs/<run>/) or
"teacher" (true state, for reference). Reward: catch/theta.reward (success minus acceleration and force penalties,
-0.5 for a failed catch).
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import theta
from .env import CatchEnv
from .teacher import Teacher

ROOT = Path(__file__).resolve().parents[1]


def load_student(run):
    import torch
    p = ROOT / "runs" / run
    sd = torch.load(p / "student_best.pt", map_location="cpu")
    nrm = np.load(p / "norm.npz")
    W = {k: sd[n].numpy().astype(np.float64) for k, n in (("w0", "0.weight"), ("b0", "0.bias"), ("w1", "2.weight"),
                                                           ("b1", "2.bias"), ("w2", "4.weight"), ("b2", "4.bias"))}
    W["mean"], W["std"] = nrm["mean"], np.sqrt(nrm["var"] + 1e-8)
    return W


def student_act(W, obs):
    x = np.clip((obs - W["mean"]) / W["std"], -10, 10)
    h = np.tanh(W["w0"] @ x + W["b0"])
    h = np.tanh(W["w1"] @ h + W["b1"])
    return np.clip(W["w2"] @ h + W["b2"], -1, 1)


class CatchParamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, executor="student", student_run="dagger_theta", t_decide=0.12, overrides=None, randomize=None,
                 u_default=None):
        self.env = CatchEnv(overrides=overrides, randomize=randomize)
        self.executor, self.t_decide = executor, t_decide
        self.W = load_student(student_run) if executor == "student" else None
        self.u_default = np.zeros(3) if u_default is None else np.asarray(u_default, float)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (13,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.last_info = {}

    def _act(self, obs, u):
        if self.executor == "student":
            return student_act(self.W, np.concatenate((obs, u)))
        return self.teacher()

    def context(self):
        e = self.env
        est = e.kf.estimate(e.t)
        from . import kin
        P, Rh = kin.palm_pose(e.cm)
        if est is None:
            return np.zeros(13, np.float32)
        pc, vc, sp, sv = est
        rel = pc - P
        vx = vc[0] if abs(vc[0]) > 0.1 else -0.1
        t_arr = float(np.clip(-rel[0] / vx, 0, 2))      # time to reach the palm plane x (operator in front, +x)
        base = np.array([0.0, 0.0, e.p.base_height_m])
        return np.concatenate((rel, 0.3 * vc, [100 * sp, sv, t_arr], P - base, [e.t])).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        e = self.env
        self.obs, _ = e.reset(seed=int(self.np_random.integers(2 ** 31)) if seed is None else seed)
        self.u = self.u_default.copy()
        self.teacher = Teacher(e, **theta.to_kwargs(self.u)) if self.executor == "teacher" else None
        self.done, self.info = False, {}
        while e.t < self.t_decide - 1e-9 and not self.done:
            self.obs, _, term, trunc, self.info = e.step(self._act(self.obs, self.u))
            self.done = term or trunc
        return self.context(), {}

    def step(self, action):
        e = self.env
        self.u = np.clip(np.asarray(action, float), -1, 1)
        if self.teacher is not None:
            for k, v in theta.to_kwargs(self.u).items():
                setattr(self.teacher, k, v)
        while not self.done:
            self.obs, _, term, trunc, self.info = e.step(self._act(self.obs, self.u))
            self.done = term or trunc
        info = dict(self.info, u=self.u.tolist(), **theta.to_kwargs(self.u))
        self.last_info = info
        return self.context(), float(theta.reward(self.info)), True, False, info
