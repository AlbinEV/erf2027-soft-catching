"""PPO for the high-level catch-parameter policy (one decision per throw, catch/high_env.py).

Evaluation every --eval-every throws on 300 selection throws (seeds 100000+, deterministic actions), compared with
fixed parameter settings; the best policy by mean reward is kept (model_best.zip + vecnormalize_best.pkl).
Usage: nice -n 19 .venv/bin/python scripts/train_high.py --name high_v1 --envs 13 --steps 60000
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
FIXED = {"nominal (0.45, 0.4, 0)": dict(ratio=0.45, absorb=0.4, brake_time=0.0),
         "best fixed (0.55, 0.2, 0.4)": dict(ratio=0.55, absorb=0.2, brake_time=0.4)}


def make(seed, kw):
    def _f():
        from catch.high_env import CatchParamEnv
        env = CatchParamEnv(**kw)
        env.reset(seed=seed)
        return env
    return _f


def run_eval(policy, kw, seeds):
    from catch.high_env import CatchParamEnv
    env = CatchParamEnv(**kw)
    out = []
    for s in seeds:
        ctx, _ = env.reset(seed=s)
        u = policy(ctx)
        _, r, _, _, info = env.step(u)
        out.append(dict(r=r, success=info["success"], acc=info["max_acc_g"], F=info["max_force"],
                        tcp=info["max_tcp_acc"], u=list(map(float, u))))
    ok = [o for o in out if o["success"]]
    return dict(reward=float(np.mean([o["r"] for o in out])), success=len(ok) / len(out),
                acc_median=float(np.median([o["acc"] for o in ok])) if ok else None,
                force_median=float(np.median([o["F"] for o in ok])) if ok else None,
                tcp_median=float(np.median([o["tcp"] for o in ok])) if ok else None,
                u_mean=np.mean([o["u"] for o in out], axis=0).round(3).tolist())


class Cb(BaseCallback):
    def __init__(self, every, path, kw):
        super().__init__()
        self.every, self.next, self.path, self.kw, self.best = every, every, path, kw, -np.inf

    def record(self):
        vn = self.training_env
        pol = lambda c: self.model.predict(vn.normalize_obs(c[None]), deterministic=True)[0][0]  # noqa: E731
        rec = dict(steps=self.num_timesteps, time=time.strftime("%H:%M:%S"), **run_eval(pol, self.kw, SEL))
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
    ap.add_argument("--name", default="high_v1")
    ap.add_argument("--envs", type=int, default=13)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--student", default="dagger_theta")
    ap.add_argument("--executor", default="student")
    a = ap.parse_args()
    torch.set_num_threads(1)
    path = ROOT / "runs" / a.name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, path / "train_high.py")
    from catch import theta
    kw = dict(executor=a.executor, student_run=a.student)
    (path / "config.json").write_text(json.dumps(dict(vars(a), ranges=theta.RANGES, weights=theta.W), indent=1))
    with open(path / "fixed_baselines.jsonl", "w") as f:
        for name, k in FIXED.items():
            u = theta.from_kwargs(**k)
            rec = dict(name=name, **run_eval(lambda c, u=u: u, kw, SEL))
            f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    venv = SubprocVecEnv([make(400000 + 1000 * i, kw) for i in range(a.envs)])
    venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=a.envs * 16, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5),
                tensorboard_log=str(path / "tb"), verbose=0, device="cpu", seed=0)
    cb = Cb(a.eval_every, path, kw)
    model.learn(total_timesteps=a.steps, callback=cb)
    cb.record()


if __name__ == "__main__":
    main()
