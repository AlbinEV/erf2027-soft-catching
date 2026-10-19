"""Gymnasium environment: a UR5e with the four-arm hand catches a parcel thrown by an operator.

Episode: the parcel leaves the operator's hand at t = 0 aimed at a point in front of the robot (with the operator's
aiming error and spin); the robot starts in the ready pose. The episode ends when the parcel is held (success),
falls, passes the robot, or the time runs out.
Action (7, in [-1, 1]): TCP linear velocity (x tcp_v_max) and angular velocity (x tcp_w_max), world frame, and the
hand motor command. A damped least-squares controller turns the twist into joint position targets for the UR5e
position servos (with velocity lead kv/kp), within the joint speed limit.
Observation (student, sensors only, 62 values): the last 3 camera measurements of the parcel centre relative to the
palm (with age and validity), the ballistic Kalman-filter estimate from the same measurements (position relative to
the palm, velocity, standard deviations, validity), UR joint angles and speeds, palm position, orientation and
velocity, motor state, last action, time since release. The teacher reads the true state from the simulator, or the
filter estimate (sensor-based classical pipeline).
Info reports the peak contact force (1 ms), the peak parcel acceleration without gravity and the peak palm
acceleration, both averaged over the 10 ms control step.
Held: parcel centre inside the closed hand (hand frame |x|, |y| < 6 cm, 0 < z < 14 cm), corner joints above 80 % of
their stops, relative speed below 0.15 m/s, for 0.3 s.
"""
import gymnasium as gym
import mujoco
import numpy as np

from . import kin
from .config import Params
from .model import CatchModel
from .perception import BallisticKF

# full-force closing time to 95 % of the corner stops versus motor stroke speed (runs/closing_times.json, 20 N)
_CT_V = np.array([0.01, 0.03, 0.0625, 0.125, 0.25])
_CT_T = np.array([2.189, 0.78, 0.409, 0.265, 0.192])


def closing_time(v):
    return float(np.exp(np.interp(np.log(v), np.log(_CT_V), np.log(_CT_T))))

Q_READY = np.array([0.398, -1.628, -2.487, -1.342, -1.293, 1.281])  # palm at (0.45, 0, base + 0.30), axis (1, 0, 1)
G = np.array([0.0, 0.0, -9.81])
R = dict(success=10.0, drop=2.0, approach=1.0, contact_ur=0.5, force=0.002, action=1e-3)


def random_quat(rng):
    u1, u2, u3 = rng.random(3)
    return np.array([np.sqrt(u1) * np.cos(2 * np.pi * u3), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                     np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u3)])


class CatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params: Params = None, overrides: dict = None, randomize: dict = None):
        """overrides: fixed parameter values. randomize: {name: [lo, hi]} sampled uniformly at every reset (scalar
        parameters, e.g. camera noise and latency, operator aiming error, spin)."""
        p = params or Params()
        for k, v in (overrides or {}).items():
            setattr(p, k, v)
        self.randomize = randomize or {}
        self.cm = CatchModel(p)
        self.p, self.m, self.d = self.cm.p, self.cm.model, self.cm.data
        self.n_sub = int(round(1.0 / (p.control_hz * p.dt_s)))
        self.dt_c = 1.0 / p.control_hz
        self.lead = 400.0 / 2000.0                     # kv / kp of the Menagerie UR5e servos
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (62,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    # --- state helpers -------------------------------------------------------------------------------------------------
    def parcel_state(self):
        d, cm = self.d, self.cm
        return (d.qpos[cm.parcel_q:cm.parcel_q + 3].copy(), d.qvel[cm.parcel_d:cm.parcel_d + 3].copy(),
                d.xmat[cm.parcel_body].reshape(3, 3).copy())

    def palm_velocity(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.cm.palm_site, v, 0)
        return v[3:], v[:3]                              # linear, angular (world frame)

    def hand_frame_parcel(self):
        P, Rh = kin.palm_pose(self.cm)
        pc, vc, _ = self.parcel_state()
        vp, _ = self.palm_velocity()
        return Rh.T @ (pc - P), Rh.T @ (vc - vp)

    def corners(self):
        q = self.d.qpos[self.cm.arm_q]
        return q[:, 5:8].mean(axis=1) / self.p.stop_rad[5:8].mean()

    def held_now(self):
        rel, relv = self.hand_frame_parcel()
        inside = abs(rel[0]) < 0.06 and abs(rel[1]) < 0.06 and -self.p.pad_stroke_m < rel[2] < 0.14
        return bool(inside and np.all(self.corners() > 0.8) and np.linalg.norm(relv) < 0.15), bool(inside)

    def contacts(self):
        """Max normal force on the parcel from the hand; parcel touching the UR or the floor."""
        m, d, cm = self.m, self.d, self.cm
        f6, fmax, ur, floor = np.zeros(6), 0.0, False, False
        for i in range(d.ncon):
            c = d.contact[i]
            if cm.parcel_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == cm.parcel_geom else c.geom1
            if other in cm.hand_geoms:
                mujoco.mj_contactForce(m, d, i, f6)
                fmax = max(fmax, abs(f6[0]))
            elif other in cm.ur_geoms:
                ur = True
            elif other in cm.floor_geoms:
                floor = True
        return fmax, ur, floor

    # --- camera ----------------------------------------------------------------------------------------------------------
    def _camera_update(self):
        p, t = self.p, self.t
        if self.next_frame <= t + 1e-9:                 # frame captured now (60 Hz camera sampled at the control rate)
            pos = self.d.qpos[self.cm.parcel_q:self.cm.parcel_q + 3]
            ok = self.rng.random() > p.cam_dropout
            meas = pos + self.rng.normal(0, p.cam_noise_m, 3) if ok else None
            self.pending.append((t, t + p.cam_latency_s, meas))
            self.next_frame += 1.0 / p.cam_rate_hz
        while self.pending and self.pending[0][1] <= t + 1e-9:
            tc, _, meas = self.pending.pop(0)
            self.frames = [(tc, meas)] + self.frames[:2]
            if meas is not None:
                self.kf.update(tc, meas)

    def observe(self):
        cm, d, p = self.cm, self.d, self.p
        P, Rh = kin.palm_pose(cm)
        vp, wp = self.palm_velocity()
        cam = []
        for k in range(3):
            if k < len(self.frames) and self.frames[k][1] is not None:
                tc, meas = self.frames[k]
                cam += list(meas - P) + [self.t - tc, 1.0]
            else:
                cam += [0.0, 0.0, 0.0, 0.0, 0.0]
        est = self.kf.estimate(self.t)
        kf = [0.0] * 9 if est is None else list(est[0] - P) + list(0.3 * est[1]) + [100 * est[2], est[3], 1.0]
        base = np.array([0.0, 0.0, p.base_height_m])
        motor = [d.qpos[cm.motor_q] / p.stroke_max_m, d.qvel[cm.motor_d] / p.close_speed_mps, self.last_action[6]]
        obs = np.concatenate((cam, kf, d.qpos[cm.ur_q], 0.3 * d.qvel[cm.ur_d], P - base, Rh[:, 0], Rh[:, 2], vp, 0.3 * wp,
                              motor, self.last_action, [self.t]))
        return obs.astype(np.float32)

    # --- gym API ---------------------------------------------------------------------------------------------------------
    def sample_throw(self, rng):
        p = self.p
        sides = rng.uniform(*p.box_side_m, 3)
        if len(p.box_dims_m):                   # same random draws, so every shape sees the same throws
            sides = np.array(p.box_dims_m, dtype=float)
        mass = rng.uniform(*p.box_mass_kg)
        p0 = np.array([rng.uniform(*p.release_x), rng.uniform(*p.release_y), rng.uniform(*p.release_z)])
        aim = np.array([0.45, 0.0, p.base_height_m + p.aim_dz]) + rng.normal(0, p.aim_noise_m, 3)
        T = rng.uniform(*p.flight_s)
        v0 = (aim - p0 - 0.5 * G * T ** 2) / T
        axis = rng.normal(size=3)
        w0 = rng.uniform(0, p.spin_max) * axis / np.linalg.norm(axis)
        return dict(sides=sides, mass=mass, p0=p0, v0=v0, w0=w0, q0=random_quat(rng), aim=aim, T=T)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        m, d, cm, p = self.m, self.d, self.cm, self.p
        for k, (lo, hi) in self.randomize.items():
            setattr(p, k, float(self.rng.uniform(lo, hi)))
        if "close_speed_mps" in self.randomize or "t_max_n" in self.randomize:
            self.cm.set_hand(p.t_max_n, p.close_speed_mps)
        self.t_close_true = closing_time(p.close_speed_mps)
        th = (options or {}).get("throw") or self.sample_throw(self.rng)
        self.throw = th
        mujoco.mj_resetData(m, d)
        cm.set_parcel(th["sides"], th["mass"])
        q0 = Q_READY + self.rng.normal(0, 0.02, 6)
        d.qpos[cm.ur_q] = q0
        d.ctrl[cm.ur_act] = q0
        self.q_ref = q0.copy()
        d.qpos[cm.motor_q] = -p.stroke_min_m
        d.ctrl[cm.motor_act] = -0.1
        d.qpos[cm.parcel_q:cm.parcel_q + 3] = th["p0"]
        d.qpos[cm.parcel_q + 3:cm.parcel_q + 7] = th["q0"]
        d.qvel[cm.parcel_d:cm.parcel_d + 3] = th["v0"]
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, th["q0"])
        d.qvel[cm.parcel_d + 3:cm.parcel_d + 6] = Rq.reshape(3, 3).T @ th["w0"]    # free joint: body-frame spin
        mujoco.mj_forward(m, d)
        self.t, self.steps = 0.0, 0
        self.next_frame, self.pending, self.frames = 0.0, [], []
        self.kf = BallisticKF(meas_std=p.cam_noise_m)
        self._camera_update()
        self.last_action = np.zeros(7)
        self.held_s, self.max_force, self.touched_ur, self.was_inside = 0.0, 0.0, False, False
        self.max_acc_g, self.max_tcp_acc = 0.0, 0.0
        self.t_contact, self.f_hist, self.v_rel_contact = None, [], None
        self.damp_cmd = [p.pad_c, p.wrist_c]           # semi-active: default to the passive values
        self.substep_hook = None                        # optional fast loop (1 kHz), called before every mj_step
        self.prev_dist = np.linalg.norm(self.hand_frame_parcel()[0])
        self.t_max = th["T"] + 1.2
        return self.observe(), {}

    def apply_action(self, a):
        p, cm, d = self.p, self.cm, self.d
        twist = np.concatenate((a[:3] * p.tcp_v_max, a[3:6] * p.tcp_w_max))
        dq = kin.twist_to_dq(cm, twist, p.joint_v_max)
        q = d.qpos[cm.ur_q]
        self.q_ref = np.clip(self.q_ref + dq * self.dt_c, q - 0.15, q + 0.15)
        d.ctrl[cm.ur_act] = self.q_ref + self.lead * dq
        d.ctrl[cm.motor_act] = a[6]
        for i, c in zip(cm.damper_act, self.damp_cmd):  # semi-active dampers (N s/m), set by the caller
            d.ctrl[i] = c

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        self.apply_action(a)
        fmax, ur, floor = 0.0, False, False
        vc0 = self.d.qvel[self.cm.parcel_d:self.cm.parcel_d + 3].copy()
        vp0, _ = self.palm_velocity()
        for _ in range(self.n_sub):
            if self.substep_hook is not None:
                self.substep_hook(self)
            mujoco.mj_step(self.m, self.d)
            f, u, fl = self.contacts()
            fmax, ur, floor = max(fmax, f), ur or u, floor or fl
        self.t += self.dt_c
        self.steps += 1
        self._camera_update()
        self.last_action = a
        self.max_force = max(self.max_force, fmax)
        self.f_hist.append(fmax)
        if self.t_contact is None and fmax > 1.0:
            self.t_contact = self.t
            self.v_rel_contact = float(np.linalg.norm(vc0 - vp0))   # parcel speed relative to the palm at contact
        # accelerations averaged over the control step (10 ms): parcel without gravity (in g), palm (m/s^2)
        acc = (self.d.qvel[self.cm.parcel_d:self.cm.parcel_d + 3] - vc0) / self.dt_c - G
        self.max_acc_g = max(self.max_acc_g, float(np.linalg.norm(acc)) / 9.81)
        vp1, _ = self.palm_velocity()
        self.max_tcp_acc = max(self.max_tcp_acc, float(np.linalg.norm(vp1 - vp0)) / self.dt_c)
        self.touched_ur = self.touched_ur or ur
        held, inside = self.held_now()
        self.was_inside = self.was_inside or inside
        self.held_s = self.held_s + self.dt_c if held else 0.0
        pc, vc, _ = self.parcel_state()
        rel, _ = self.hand_frame_parcel()
        dist = np.linalg.norm(rel)
        success = self.held_s >= 0.3
        dropped = floor or pc[2] < 0.3 or pc[0] < -0.4
        timeout = self.t >= self.t_max
        r = R["approach"] * (self.prev_dist - dist) - R["contact_ur"] * float(ur) - R["force"] * max(0.0, fmax - 50.0)
        r -= R["action"] * float(np.sum(a ** 2))
        r += R["success"] if success else 0.0
        r -= R["drop"] if dropped and not success else 0.0
        self.prev_dist = dist
        info = dict(success=success, dropped=dropped, held=held, inside=inside, max_force=self.max_force,
                    max_acc_g=self.max_acc_g, max_tcp_acc=self.max_tcp_acc, v_rel_contact=self.v_rel_contact,
                    touched_ur=self.touched_ur, t=self.t,
                    dist=dist)
        return self.observe(), float(r), bool(success or dropped), bool(timeout and not (success or dropped)), info
