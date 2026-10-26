"""PPO for the high-level catch-parameter policy v2 (catch/high_env2.py) and evaluation helpers.

Training conditions: --overrides / --randomize (JSON). Evaluation every --eval-every decisions on 300 selection throws
(seeds 100000+), best policy kept by mean reward. Baselines evaluated with the same executor: fixed parameters and
the adaptive rule (closing ratio scaled with the encoder-measured motor speed).
Usage: nice -n 19 .venv/bin/python scripts/train_high2.py --name hi2_cam --sensors "" --envs 7 --steps 100000
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEL = list(range(100000, 100300))


def make(seed, kw):
    def _f():
        from catch.high_env2 import CatchParamEnv2
        env = CatchParamEnv2(**kw)
        env.reset(seed=seed)
        return env
    return _f


def episode(env, policy, seed):
    """policy(obs, env) -> action; returns the final info and reward."""
    obs, _ = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        if term or trunc:
            return info, r


def summarise(rows):
    ok = [x for x in rows if x["success"]]
    return dict(n=len(rows), reward=float(np.mean([x["r"] for x in rows])), success=len(ok) / len(rows),
                acc_median=float(np.median([x["acc"] for x in ok])) if ok else None,
                acc_p90=float(np.percentile([x["acc"] for x in ok], 90)) if ok else None,
                force_median=float(np.median([x["F"] for x in ok])) if ok else None,
                tcp_median=float(np.median([x["tcp"] for x in ok])) if ok else None)


def run_eval(policy, kw, seeds):
    from catch.high_env2 import CatchParamEnv2
    env = CatchParamEnv2(**kw)
    rows = []
    for s in seeds:
        info, r = episode(env, policy, s)
        rows.append(dict(seed=s, r=r, success=bool(info["success"]), acc=info["max_acc_g"], F=info["max_force"],
                         tcp=info["max_tcp_acc"], u=info["u"], mass=float(env.env.throw["mass"]),
                         speed_motor=float(env.env.p.close_speed_mps)))
    return summarise(rows), rows


def fixed_policy(**kw):
    from catch import theta2
    u = theta2.from_kwargs(**kw)
    return lambda obs, env: u


def adaptive_rule(**kw):
    """Closing ratio scaled with the motor speed measured by the encoder (classical adaptation)."""
    from catch import theta2
    from catch.env import closing_time

    def pol(obs, env):
        v = env.env.p.close_speed_mps * env.env.p.motor_force_scale * (1 + env.rng_s.normal(0, 0.03))
        k = dict(kw, ratio=min(theta2.RANGES["ratio"][1], kw["ratio"] * closing_time(v) / closing_time(0.25)))
        return theta2.from_kwargs(**k)
    return pol


NOMINAL = dict(ratio=0.45, absorb=0.4, brake_time=0.0, turn_gain=0.0, motor_hold=1.0)


class Cb(BaseCallback):
    def __init__(self, every, path, kw):
        super().__init__()
        self.every, self.next, self.path, self.kw, self.best = every, every, path, kw, -np.inf

    def record(self):
        vn = self.training_env
        pol = lambda o, env: self.model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]  # noqa: E731
        summ, _ = run_eval(pol, self.kw, SEL)
        rec = dict(steps=self.num_timesteps, time=time.strftime("%H:%M:%S"), **summ)
        self.model.save(self.path / "model")
        vn.save(str(self.path / "vecnormalize.pkl"))
        if rec["reward"] >= self.best:
            self.best = rec["reward"]
            shutil.copy(self.path / "model.zip", self.path / "model_best.zip")
            shutil.copy(self.path / "vecnormalize.pkl", self.path / "vecnormalize_best.pkl")
            rec["best"] = True
        with open(self.path / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.record()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sensors", default="", help="comma list among size,encoder,ft")
    ap.add_argument("--two-step", action="store_true")
    ap.add_argument("--envs", type=int, default=7)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--student", default="dagger_theta2")
    ap.add_argument("--overrides", default='{"box_mass_kg": [0.1, 0.8], "flight_s": [0.45, 0.8]}')
    ap.add_argument("--randomize", default='{"close_speed_mps": [0.12, 0.25], "aim_noise_m": [0.06, 0.2]}')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="{}", help="JSON reward weight overrides (catch/theta2.W)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    path = ROOT / "runs" / a.name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, path / "train_high2.py")
    from catch import theta2
    sensors = tuple(x for x in a.sensors.split(",") if x)
    kw = dict(student_run=a.student, sensors=sensors, two_step=a.two_step, overrides=json.loads(a.overrides),
              randomize=json.loads(a.randomize), weights=json.loads(a.weights) or None)
    (path / "config.json").write_text(json.dumps(dict(vars(a), ranges=theta2.RANGES, weights=theta2.W), indent=1))
    venv = SubprocVecEnv([make(400000 + 10000 * a.seed + 1000 * i, kw) for i in range(a.envs)])
    venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=a.envs * 16, n_epochs=10, gamma=1.0, gae_lambda=1.0,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.7), tensorboard_log=str(path / "tb"),
                verbose=0, device="cpu", seed=a.seed)
    cb = Cb(a.eval_every, path, kw)
    model.learn(total_timesteps=a.steps, callback=cb)
    cb.record()


if __name__ == "__main__":
    main()
