"""PPO fine-tuning of a saved policy (e.g. the DAgger student) on the catching task.

The starting policy is evaluated first; then --warmup steps update only the value function (actor frozen), then the
whole policy with a small learning rate and target KL 0.01. --reward overrides the reward weights inside each worker
(project defaults untouched); --overrides and --randomize set the training and selection conditions. Evaluation on
the selection throws (seeds 100000+) every --eval-every steps; best checkpoint kept as model_best.zip +
vecnormalize_best.pkl.
Usage: nice -n 19 .venv/bin/python scripts/train_ft.py --init dagger_v1 --name ft_v1 --envs 13 --deadline 08:00
"""
import argparse
import datetime
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
def make(seed, reward, ov, rnd):
    def _f():
        import catch.env as E
        E.R.update(reward)
        env = E.CatchEnv(overrides=ov, randomize=rnd)
        env.reset(seed=seed)
        return env
    return _f


def evaluate(model, vecnorm, seeds, ov=None, rnd=None):
    from catch.env import CatchEnv
    out = []
    env = CatchEnv(overrides=ov, randomize=rnd)
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            a, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
            obs, r, term, trunc, info = env.step(a[0])
            if term or trunc:
                break
        out.append((bool(info["success"]), float(info["max_force"])))
    return out


class Cb(BaseCallback):
    def __init__(self, every, n_eval, path, deadline, ov=None, rnd=None):
        super().__init__()
        self.ov, self.rnd = ov, rnd
        self.every, self.next, self.path, self.deadline = every, every, path, deadline
        self.seeds, self.best = list(range(100000, 100000 + n_eval)), -1.0

    def record(self, final=False):
        res = evaluate(self.model, self.training_env, self.seeds, self.ov, self.rnd)
        succ = float(np.mean([r[0] for r in res]))
        forces = [r[1] for r in res if r[0]]
        rec = dict(steps=self.num_timesteps, time=time.strftime("%H:%M:%S"), final=final, success=succ,
                   median_force_ok=float(np.median(forces)) if forces else None,
                   p90_force_ok=float(np.percentile(forces, 90)) if forces else None)
        self.model.save(self.path / "model")
        self.training_env.save(str(self.path / "vecnormalize.pkl"))
        score = succ - 0.001 * (rec["median_force_ok"] or 0.0)      # success first, then lower force
        if score >= self.best:
            self.best = score
            shutil.copy(self.path / "model.zip", self.path / "model_best.zip")
            shutil.copy(self.path / "vecnormalize.pkl", self.path / "vecnormalize_best.pkl")
            rec["best"] = True
        with open(self.path / "evaluations.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def _on_step(self):
        if self.deadline is not None and time.time() >= self.deadline:
            return False
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.record()
        return True


def deadline_ts(hhmm):
    if not hhmm:
        return None
    now = datetime.datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += datetime.timedelta(days=1)
    return t.timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", default="ft_v1")
    ap.add_argument("--envs", type=int, default=13)
    ap.add_argument("--steps", type=int, default=50_000_000)
    ap.add_argument("--warmup", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward", default='{"force": 0.01}', help="JSON overrides of the reward weights")
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--randomize", default="{}")
    a = ap.parse_args()
    REWARD, ov, rnd = json.loads(a.reward), json.loads(a.overrides), json.loads(a.randomize)
    torch.set_num_threads(1)
    path = ROOT / "runs" / a.name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, path / "train_ft.py")
    (path / "config.json").write_text(json.dumps(dict(vars(a), target_kl=0.01), indent=1))
    venv = SubprocVecEnv([make(300000 + a.seed * 1000 + i, REWARD, ov, rnd) for i in range(a.envs)])
    src = ROOT / "runs" / a.init
    venv = VecNormalize.load(str(src / "vecnormalize.pkl"), VecMonitor(venv))
    # frozen observation statistics: the actor was trained on these; updating them shifts its inputs (ft_v2 collapse)
    venv.training, venv.norm_reward = False, False
    lr = a.lr
    model = PPO.load(src / "model", env=venv, device="cpu", tensorboard_log=str(path / "tb"),
                     custom_objects=dict(learning_rate=lr, lr_schedule=lambda _: lr, target_kl=0.01,
                                         n_steps=256, batch_size=a.envs * 128))
    cb = Cb(a.eval_every, a.eval_episodes, path, deadline_ts(a.deadline), ov, rnd)
    cb.init_callback(model)
    cb.record()
    actor = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    actor.append(model.policy.log_std)
    for q in actor:
        q.requires_grad_(False)
    model.learn(total_timesteps=a.warmup, callback=cb)
    for q in actor:
        q.requires_grad_(True)
    if cb.deadline is None or time.time() < cb.deadline:
        model.learn(total_timesteps=a.steps, callback=cb, reset_num_timesteps=False)
    cb.record(final=True)


if __name__ == "__main__":
    main()
