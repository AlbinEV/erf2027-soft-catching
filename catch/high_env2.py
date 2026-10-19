"""High-level environment v2: up to two decisions per throw, optional extra sensors (18 Oct 2026).

Decision 1, at t_decide after release: the five catch parameters (catch/theta2.py). Decision 2 (two_step=True), 30 ms
after the first contact: only the post-contact parameters (braking, palm-up turn, hold force) are updated.
Low-level executor: the parameter-conditioned student v2 (camera only, numpy weights from runs/<run>/).
Observation (21 values, unused sensors are zero):
  camera (Kalman filter): parcel position relative to the palm, velocity, standard deviations, time to the palm plane,
      palm position, time (13);
  size: largest parcel side from the depth camera, 5 mm noise (1);
  encoder: motor stroke speed measured by the encoder, 3 % noise (1);
  ft: wrist force sensor, mean contact force over the first 30 ms after contact (5 % noise) and the parcel speed
      from the camera at contact (2);
  decision index, contact flag, measured time to contact, spare (4).
Reward (catch/theta2.reward) at the end of the throw.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import kin, theta2 as theta
from .env import CatchEnv
from .high_env import load_student, student_act

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ("size", "encoder", "ft")


class CatchParamEnv2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, student_run="dagger_theta2", sensors=(), two_step=False, t_decide=0.12, overrides=None,
                 randomize=None, ft_window=0.03, weights=None):
        self.env = CatchEnv(overrides=overrides, randomize=randomize)
        self.W = load_student(student_run)
        self.sensors, self.two_step, self.t_decide, self.ft_window = set(sensors), two_step, t_decide, ft_window
        self.weights = weights                          # reward weight overrides (Pareto study)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (21,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.last_info = {}

    def _run_until(self, cond):
        e = self.env
        while not self.done and not cond():
            a = student_act(self.W, np.concatenate((self.obs, self.u)))
            self.obs, _, term, trunc, self.info = e.step(a)
            self.done = term or trunc

    def _obs(self, stage):
        e, rng = self.env, self.rng_s
        out = np.zeros(21, np.float32)
        est = e.kf.estimate(e.t)
        P, _ = kin.palm_pose(e.cm)
        base = np.array([0.0, 0.0, e.p.base_height_m])
        if est is not None:
            pc, vc, sp, sv = est
            rel = pc - P
            vx = vc[0] if abs(vc[0]) > 0.1 else -0.1
            out[:13] = np.concatenate((rel, 0.3 * vc, [100 * sp, sv, np.clip(-rel[0] / vx, 0, 2)], P - base, [e.t]))
        if "size" in self.sensors:
            out[13] = 10 * (np.max(e.throw["sides"]) + rng.normal(0, 0.005))
        if "encoder" in self.sensors:
            out[14] = 4 * e.p.close_speed_mps * (1 + rng.normal(0, 0.03))
        if "ft" in self.sensors and stage == 1 and e.t_contact is not None:
            k0 = int(round(e.t_contact * e.p.control_hz)) - 1
            f = np.mean(e.f_hist[k0:k0 + max(1, int(round(self.ft_window * e.p.control_hz)))])
            out[15] = f / 100 * (1 + rng.normal(0, 0.05))
            out[16] = self.v_contact
        out[17] = stage
        out[18] = float(e.t_contact is not None)
        out[19] = 0.0 if e.t_contact is None else e.t_contact
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        e = self.env
        s = int(self.np_random.integers(2 ** 31)) if seed is None else seed
        self.rng_s = np.random.default_rng(s + 99)
        self.obs, _ = e.reset(seed=s)
        self.u = np.zeros(5)
        self.done, self.info, self.stage, self.v_contact = False, {}, 0, 0.0
        self._run_until(lambda: e.t >= self.t_decide - 1e-9)
        return self._obs(0), {}

    def step(self, action):
        e = self.env
        a = np.clip(np.asarray(action, float), -1, 1)
        if self.stage == 0:
            self.u = a.copy()
            if self.two_step:
                self._run_until(lambda: e.t_contact is not None)
                est = e.kf.estimate(e.t)
                self.v_contact = 0.0 if est is None else float(np.linalg.norm(est[1]))
                tc = e.t_contact
                self._run_until(lambda: tc is not None and e.t >= tc + self.ft_window - 1e-9)
                if not self.done:
                    self.stage = 1
                    return self._obs(1), 0.0, False, False, {}
        else:
            self.u[list(theta.POST)] = a[list(theta.POST)]
        self._run_until(lambda: False)
        info = dict(self.info, u=self.u.tolist(), **theta.to_kwargs(self.u))
        self.last_info = info
        return self._obs(self.stage), float(theta.reward(self.info, self.weights)), True, False, info
