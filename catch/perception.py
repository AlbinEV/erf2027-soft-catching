"""Ballistic Kalman filter on the camera measurements of the parcel centre (sensor-only estimate).

State [p, v] in the world frame, known gravity, white-acceleration process noise. Frames arrive late (latency):
each one is processed at its capture time; the estimate is then predicted to the current time.
"""
import numpy as np

G = np.array([0.0, 0.0, -9.81])


class BallisticKF:
    def __init__(self, meas_std=0.005, acc_std=0.5, v0_std=5.0):
        self.r, self.q, self.v0_std = meas_std ** 2, acc_std ** 2, v0_std
        self.x, self.P, self.t, self.n = None, None, None, 0

    @staticmethod
    def _F(dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        return F

    def _Q(self, dt):
        Q = np.zeros((6, 6))
        Q[:3, :3] = dt ** 4 / 4 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = dt ** 3 / 2 * np.eye(3)
        Q[3:, 3:] = dt ** 2 * np.eye(3)
        return self.q * Q

    def _predict(self, x, P, dt):
        x = self._F(dt) @ x + np.concatenate((0.5 * G * dt * dt, G * dt))
        P = self._F(dt) @ P @ self._F(dt).T + self._Q(dt)
        return x, P

    def update(self, t, z):
        if self.x is None:
            self.x = np.concatenate((z, np.zeros(3)))
            self.P = np.diag([self.r] * 3 + [self.v0_std ** 2] * 3)
            self.t, self.n = t, 1
            return
        self.x, self.P = self._predict(self.x, self.P, t - self.t)
        self.t = t
        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        S = H @ self.P @ H.T + self.r * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P
        self.n += 1

    def estimate(self, t):
        """Position, velocity and position/velocity standard deviations predicted to time t (None before data)."""
        if self.x is None:
            return None
        x, P = self._predict(self.x, self.P, t - self.t)
        return x[:3], x[3:], float(np.sqrt(np.trace(P[:3, :3]) / 3)), float(np.sqrt(np.trace(P[3:, 3:]) / 3))
