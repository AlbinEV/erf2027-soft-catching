"""Catch parameters chosen per throw by the high-level RL policy (normalised u in [-1, 1]^3).

ratio: closing instant (closing ratio); absorb: fraction of the parcel velocity the palm follows from the trigger;
brake_time: time to bring the palm to rest after giving way. Ranges from the teacher sweeps of 18 Oct 2026
(ratio 0.45 -> 99 % success, 0.55 -> 98 % with lower force, 0.60 -> 77 % with the lowest force).
"""
import numpy as np

RANGES = dict(ratio=(0.40, 0.70), absorb=(0.0, 0.6), brake_time=(0.0, 0.6))
NAMES = tuple(RANGES)
# reward of the high-level policy: success minus penalties on parcel acceleration, contact force and palm acceleration
W = dict(acc_g=0.25 / 25.0, force=0.25 / 100.0, tcp=0.1 / 100.0, drop=0.5)


def to_kwargs(u):
    u = np.clip(np.asarray(u, float), -1, 1)
    return {k: float(lo + (ui + 1) / 2 * (hi - lo)) for (k, (lo, hi)), ui in zip(RANGES.items(), u)}


def from_kwargs(**kw):
    return np.array([2 * (kw[k] - lo) / (hi - lo) - 1 for k, (lo, hi) in RANGES.items()])


def reward(info):
    if info["success"]:
        return 1.0 - W["acc_g"] * info["max_acc_g"] - W["force"] * info["max_force"] - W["tcp"] * info["max_tcp_acc"]
    return -W["drop"]
