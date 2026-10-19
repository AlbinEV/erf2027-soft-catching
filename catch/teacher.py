"""Teacher with the true state (privileged): intercept, align, close with a timing rule, absorb.

1. Predict the ballistic path of the parcel and pick the intercept time: among the reachable points of the path
   that the palm can reach in time (TCP speed budget), the one closest to the palm. The intercept time is then kept
   (the ballistic prediction is exact with the true state) and re-planned only once it is well past.
2. Place the palm so that at the intercept time the parcel centre is `depth` in front of it, with the hand axis
   opposite to the parcel velocity there.
3. Close (full motor force) when the gap between the palm and the closest parcel corner reaches
   ratio * v_rel * t_c and the parcel is centred on the hand. ratio = 1/sqrt(2) is a timing rule meant for a
   slow approach; for fast parcels a smaller ratio is needed (swept in the study).
4. From the trigger, move the palm along the parcel velocity (fraction `absorb`, for `absorb_time`) to reduce the
   impact, then brake: linearly to zero in `brake_time` (0 = zero twist at once, the joint servos brake the arm).
   After the closing time the motor command drops to `motor_hold` (less squeeze on the parcel).
   With absorb_lead > 0 the palm starts moving with the parcel already when the predicted time to contact falls below
   absorb_lead (lateral alignment still corrected), which lowers the impact speed; absorb_ramp limits the palm
   acceleration when this motion starts.
   ratio, absorb, absorb_lead, absorb_ramp, absorb_time, brake_time (and motor_hold) are the catch parameters chosen
   per throw by the RL policy. With turn_up=True the hand is then turned palm up, so that the
   parcel weight rests on the palm and the arms only cage it (single-tendon arms cannot carry heavy parcels sideways).
source="true" reads the true parcel state (privileged teacher); source="kf" reads the Kalman-filter estimate from the
camera and treats the parcel as a sphere of its largest half side (sensor-based classical pipeline).
"""
import numpy as np

from . import kin

G = np.array([0.0, 0.0, -9.81])
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


class Teacher:
    def __init__(self, env, t_close=0.19, ratio=0.45, depth=0.02, absorb=0.4, kp=12.0, kr=10.0,
                 v_budget=0.5, lateral_max=0.05, source="true", turn_up=False, kup=4.0, absorb_time=0.15,
                 brake_time=0.0, motor_hold=1.0, absorb_lead=0.0, absorb_ramp=0.0, close_cmd=1.0, turn_gain=0.0):
        self.env, self.t_close, self.ratio, self.depth, self.absorb = env, t_close, ratio, depth, absorb
        self.lateral_max, self.source, self.turn_up, self.kup = lateral_max, source, turn_up, kup
        self.absorb_time, self.brake_time, self.motor_hold = absorb_time, brake_time, motor_hold
        self.v_brake0 = None
        self.absorb_lead, self.absorb_ramp, self.t_absorb = absorb_lead, absorb_ramp, None
        self.close_cmd = close_cmd                      # motor command while closing (slower, softer closing)
        self.turn_gain = turn_gain                      # palm-up turn after the catch chosen per throw (0 = off)
        self.min_lateral = np.inf
        self.kp, self.kr, self.v_budget = kp, kr, v_budget
        self.triggered, self.t_trig, self.goal, self.t_abs = False, None, None, None
        hb = env.p.base_height_m
        self.shoulder = np.array([0.0, 0.0, hb + 0.163])

    def absorb_factor(self, t):
        if self.t_absorb is None:
            return 0.0
        return 1.0 if self.absorb_ramp <= 0 else min(1.0, (t - self.t_absorb) / self.absorb_ramp)

    def reachable(self, x):
        hb = self.env.p.base_height_m
        r = np.linalg.norm(x - self.shoulder)
        return (0.25 < x[0] < 0.85) and abs(x[1]) < 0.5 and (hb - 0.05 < x[2] < hb + 0.8) and 0.3 < r < 0.85

    def pose_at(self, pc, vc, t):
        x = pc + vc * t + 0.5 * G * t * t
        v = vc + G * t
        z = -v / np.linalg.norm(v)
        return x, x - z * (self.depth + 0.5 * np.max(self.env.throw["sides"])), z

    def plan(self, pc, vc, P):
        p = self.env.p
        best = None
        for t in np.arange(0.03, 1.2, 0.01):
            x, palm, z = self.pose_at(pc, vc, t)
            if not self.reachable(x):
                continue
            travel = np.linalg.norm(palm - P) / (self.v_budget * p.tcp_v_max)
            slack = t - travel
            key = (0 if slack > 0.1 else 1, np.linalg.norm(palm - P) if slack > 0.1 else -slack)
            if best is None or key < best[0]:
                best = (key, t)
        return None if best is None else best[1]

    def __call__(self, obs=None):
        env, p = self.env, self.env.p
        P, Rh = kin.palm_pose(env.cm)
        vp, wp = env.palm_velocity()
        a = np.zeros(7)
        a[6] = -0.1
        if self.source == "true":
            pc, vc, Rp = env.parcel_state()
        else:
            est = env.kf.estimate(env.t)
            if est is None:
                return a
            pc, vc, Rp = est[0], est[1], None
        if not self.triggered:
            t_rel = None if self.t_abs is None else self.t_abs - env.t
            if t_rel is None or t_rel <= -0.2:                  # no plan, or the planned time is well past
                t_new = self.plan(pc, vc, P)
                if t_new is not None:
                    self.t_abs, t_rel = env.t + t_new, t_new
            if t_rel is not None:
                x, palm_d, z_d = self.pose_at(pc, vc, max(t_rel, 0.0))
                self.goal = (palm_d, z_d, t_rel)
            z = Rh[:, 2]
            half = 0.5 * env.throw["sides"]
            if Rp is not None:
                gap = float(np.min((pc + (CORNERS * half) @ Rp.T - P) @ z))
            else:
                gap = float((pc - P) @ z - np.max(half))
            rel = pc - P
            lateral = np.linalg.norm(rel - np.dot(rel, z) * z)
            v_rel = -float(np.dot(vc - vp, z))
            ttc = gap / v_rel if v_rel > 0.1 else np.inf
            if self.absorb_lead > 0 and self.t_absorb is None and ttc <= self.absorb_lead:
                self.t_absorb = env.t                   # start moving with the parcel before contact
            if self.goal is not None:
                palm_d, z_d, _ = self.goal
                R_d = kin.frame_from_axis(z_d)
                e = palm_d - P
                if self.t_absorb is not None:           # keep the lateral alignment, give way along the axis
                    v_cmd = self.kp * (e - np.dot(e, z_d) * z_d) + self.absorb_factor(env.t) * self.absorb * vc
                else:
                    v_cmd = self.kp * e
                w_cmd = self.kr * kin.rot_error(Rh, R_d)
                a[:3] = v_cmd / p.tcp_v_max
                a[3:6] = w_cmd / p.tcp_w_max
            # closing trigger
            if gap < 0.25:
                self.min_lateral = min(self.min_lateral, lateral)
            if v_rel > 0 and lateral < self.lateral_max and gap <= self.ratio * v_rel * self.t_close:
                self.triggered, self.t_trig = True, env.t
        if self.triggered:
            since = env.t - self.t_trig
            a[6] = self.close_cmd if since < self.t_close else self.motor_hold
            if self.t_absorb is None:
                self.t_absorb = self.t_trig
            if since < self.absorb_time:
                a[:3] = self.absorb_factor(env.t) * self.absorb * vc / p.tcp_v_max
            else:
                if self.v_brake0 is None:
                    self.v_brake0 = vp.copy()
                s_ = (since - self.absorb_time) / self.brake_time if self.brake_time > 0 else 1.0
                a[:3] = max(0.0, 1.0 - s_) * self.v_brake0 / p.tcp_v_max
                a[3:6] = 0.0                            # zero twist at the end: the servos hold the arm
                gain = self.kup if self.turn_up else self.turn_gain
                if gain > 0:
                    w = gain * np.cross(Rh[:, 2], [0.0, 0.0, 1.0])
                    a[3:6] = w / p.tcp_w_max
        return np.clip(a, -1, 1)
