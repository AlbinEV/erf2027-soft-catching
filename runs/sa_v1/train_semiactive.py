"""PPO for the semi-active damping policy (catch/semiactive_env.py).

Evaluation every --eval-every steps on 200 selection throws (seeds 100000+), deterministic actions: success, median
and 90th-percentile parcel acceleration, mean return; best policy by mean return kept (model_best.zip).
Usage: nice -n 19 .venv/bin/python scripts/train_semiactive.py --name sa_v1 --envs 12 --steps 3000000
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
SEL = list(range(100000, 100200))


def make(seed, kw):
    def _f():
        from catch.semiactive_env import SemiActiveEnv
        env = SemiActiveEnv(**kw)
        env.reset(seed=seed)
        return env
    return _f


def evaluate(policy, kw, seeds):
    from catch.semiactive_env import SemiActiveEnv
    env = SemiActiveEnv(**kw)
    rows = []
    for s in seeds:
        o, _ = env.reset(seed=s)
        R, bottom = 0.0, 0
        while True:
            o, r, te, tr, info = env.step(policy(o))
            R += r
            bottom += env.bottoming() > 0
            if te or tr:
                break
        rows.append(dict(seed=s, success=bool(info["success"]), acc=float(info["max_acc_g"]), F=float(info["max_force"]),
                         ret=R, bottom_steps=int(bottom)))
    ok = [x for x in rows if x["success"]]
    return dict(success=len(ok) / len(rows), ret=float(np.mean([x["ret"] for x in rows])),
                acc_median=float(np.median([x["acc"] for x in ok])) if ok else None,
                acc_p90=float(np.percentile([x["acc"] for x in ok], 90)) if ok else None,
                force_median=float(np.median([x["F"] for x in ok])) if ok else None,
                bottom_mean=float(np.mean([x["bottom_steps"] for x in rows]))), rows


class Cb(BaseCallback):
    def __init__(self, every, path, kw):
        super().__init__()
        self.every, self.next, self.path, self.kw, self.best = every, every, path, kw, -np.inf

    def record(self):
        vn = self.training_env
        pol = lambda o: self.model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]  # noqa: E731
        summ, _ = evaluate(pol, self.kw, SEL)
        rec = dict(steps=self.num_timesteps, time=time.strftime("%H:%M:%S"), **summ)
        self.model.save(self.path / "model")
        vn.save(str(self.path / "vecnormalize.pkl"))
        if rec["ret"] >= self.best:
            self.best = rec["ret"]
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
    ap.add_argument("--name", default="sa_v1")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--eval-every", type=int, default=200_000)
    ap.add_argument("--kw", default="{}", help="JSON options of SemiActiveEnv (overrides, randomize, weights)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    path = ROOT / "runs" / a.name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, path / "train_semiactive.py")
    kw = json.loads(a.kw)
    (path / "config.json").write_text(json.dumps(vars(a), indent=1))
    venv = SubprocVecEnv([make(450000 + 10000 * a.seed + 1000 * i, kw) for i in range(a.envs)])
    venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=a.envs * 128, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-0.5), tensorboard_log=str(path / "tb"),
                verbose=0, device="cpu", seed=a.seed)
    cb = Cb(a.eval_every, path, kw)
    model.learn(total_timesteps=a.steps, callback=cb)
    cb.record()


if __name__ == "__main__":
    main()
