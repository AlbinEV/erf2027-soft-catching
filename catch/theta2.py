"""Catch parameters v2 (18 Oct 2026): five parameters chosen by the high-level RL policy, normalised u in [-1, 1]^5.

Before the catch (decided shortly after release): ratio (closing instant, with the nominal 0.19 s closing time in the
rule, so a slower motor needs a larger ratio), absorb (fraction of the parcel velocity the palm follows).
After the first contact: brake_time (palm to rest), turn_gain (palm-up turn, 0 = off), motor_hold (motor command
once closed). The low-level student is conditioned on u; the high-level policy may change the post-contact part.
"""
import os

import numpy as np

# CATCH_RATIO_MAX widens the closing-ratio range for the degraded-actuation study (19 Oct); default 0.95
RANGES = dict(ratio=(0.35, float(os.environ.get("CATCH_RATIO_MAX", "0.95"))), absorb=(0.0, 0.6), brake_time=(0.0, 0.6), turn_gain=(0.0, 8.0),
              motor_hold=(0.5, 1.0))
NAMES = tuple(RANGES)
PRE, POST = (0, 1), (2, 3, 4)
W = dict(acc_g=0.25 / 25.0, force=0.25 / 100.0, tcp=0.1 / 100.0, drop=0.5)


def to_kwargs(u):
    u = np.clip(np.asarray(u, float), -1, 1)
    return {k: float(lo + (ui + 1) / 2 * (hi - lo)) for (k, (lo, hi)), ui in zip(RANGES.items(), u)}


def from_kwargs(**kw):
    return np.array([2 * (kw[k] - lo) / (hi - lo) - 1 for k, (lo, hi) in RANGES.items()])


def reward(info, w=None):
    w = W if w is None else dict(W, **w)
    if info["success"]:
        return 1.0 - w["acc_g"] * info["max_acc_g"] - w["force"] * info["max_force"] - w["tcp"] * info["max_tcp_acc"]
    return -w["drop"]
