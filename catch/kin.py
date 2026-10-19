"""Kinematics of the UR5e + hand: palm pose, Jacobian, damped least-squares IK and TCP-twist control.

TCP = site `palm_c` (centre of the palm surface); its frame is the hand frame (+z = hand axis).
"""
import mujoco
import numpy as np


def palm_pose(cm):
    d = cm.data
    return d.site_xpos[cm.palm_site].copy(), d.site_xmat[cm.palm_site].reshape(3, 3).copy()


def jacobian(cm):
    m, d = cm.model, cm.data
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, cm.palm_site)
    return np.vstack((jp[:, cm.ur_d], jr[:, cm.ur_d]))


def rot_error(R, R_des):
    """Rotation vector (world frame) that takes R to R_des."""
    E = R_des @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, E.ravel())
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def frame_from_axis(z, up=(0.0, 0.0, 1.0)):
    """Rotation matrix with hand axis z and the hand x axis as close as possible to `up`."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(up, float) - np.dot(up, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def ik(cm, pos, R_des, q0, iters=200, tol=1e-4):
    """Damped least-squares IK on the UR joints (kinematics only). Returns joint angles and final error."""
    m, d = cm.model, cm.data
    q = np.array(q0, float)
    for _ in range(iters):
        d.qpos[cm.ur_q] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R = palm_pose(cm)
        e = np.concatenate((pos - p, rot_error(R, R_des)))
        if np.linalg.norm(e) < tol:
            break
        J = jacobian(cm)
        q += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = (q + np.pi) % (2 * np.pi) - np.pi                 # stay within the elbow range and avoid wraps
    return q, float(np.linalg.norm(e))


def ik_restarts(cm, pos, R_des, q_pref, n=60, seed=0):
    """IK from several starts; among converged solutions, the one closest to q_pref."""
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n):
        q0 = np.asarray(q_pref, float) + (0 if k == 0 else 1) * rng.uniform(-1.5, 1.5, 6)
        q, err = ik(cm, pos, R_des, q0)
        if err < 1e-3:
            cost = np.linalg.norm(q - q_pref)
            if best is None or cost < best[1]:
                best = (q, cost)
    return None if best is None else best[0]


def twist_to_dq(cm, twist, joint_v_max, damping=1e-3):
    """Joint velocities for a world-frame TCP twist [v, w], scaled to respect the joint speed limit."""
    J = jacobian(cm)
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)
    s = np.max(np.abs(dq)) / joint_v_max
    return dq / s if s > 1 else dq
