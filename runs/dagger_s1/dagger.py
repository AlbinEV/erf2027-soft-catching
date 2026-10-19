"""DAgger for the catching task: a sensor-only student imitates the privileged teacher.

The student sees the 62-value observation (camera frames, Kalman-filter estimate, robot and hand state); the teacher
(true state) labels every visited state. Iteration 0 rolls out the teacher; later iterations roll out the student
(mixed with the teacher with probability beta) and aggregate the labels. The student has the PPO actor architecture
(62 -> 256 -> 256 -> 7, tanh) and is exported as an SB3 PPO model + VecNormalize statistics (model.zip,
vecnormalize.pkl, the best iteration on the selection throws), ready for evaluation, rendering and PPO fine-tuning.
Seeds: training 200000+, selection 100000-100099; the final test uses other seeds (scripts/eval_policies.py).
Usage: nice -n 19 .venv/bin/python scripts/dagger.py --name dagger_v1 --procs 13
"""
import argparse
import json
import math
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEL_SEEDS = list(range(100000, 100100))
OBS = 62


def student(W, obs):
    x = np.clip((obs - W["mean"]) / W["std"], -10, 10)
    h = np.tanh(W["w0"] @ x + W["b0"])
    h = np.tanh(W["w1"] @ h + W["b1"])
    return np.clip(W["w2"] @ h + W["b2"], -1, 1)


def episode(job):
    from catch.env import CatchEnv
    from catch.teacher import Teacher
    seed, beta, W, rnd, ov = job
    env = CatchEnv(overrides=ov, randomize=rnd)
    obs, _ = env.reset(seed=seed)
    te = Teacher(env)
    use_teacher = W is None or np.random.default_rng(seed + 12345).random() < beta
    O, A = [], []
    while True:
        at = te()
        O.append(obs)
        A.append(at)
        obs, r, term, trunc, info = env.step(at if use_teacher else student(W, obs))
        if term or trunc:
            break
    meta = dict(seed=seed, teacher=bool(use_teacher), success=bool(info["success"]), dropped=bool(info["dropped"]),
                max_force=float(info["max_force"]))
    return np.asarray(O, np.float32), np.asarray(A, np.float32), meta


def train(net, X, Y, mean, std, epochs, torch):
    xs = torch.as_tensor(np.clip((X - mean) / std, -10, 10), dtype=torch.float32)
    ys = torch.as_tensor(Y, dtype=torch.float32)
    wdim = torch.tensor([1, 1, 1, 1, 1, 1, 3.0])          # motor command: it sets the closing instant
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(xs)
    tot = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = (((net(xs[idx]) - ys[idx]) ** 2) * wdim).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
    return tot / n


def weights(net, mean, std):
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    return dict(mean=mean, std=std, w0=sd["0.weight"], b0=sd["0.bias"], w1=sd["2.weight"], b1=sd["2.bias"],
                w2=sd["4.weight"], b2=sd["4.bias"])


def export(net, mean, var, n, path, torch):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from catch.env import CatchEnv
    venv = VecNormalize(DummyVecEnv([lambda: CatchEnv()]), norm_obs=True, norm_reward=True, clip_obs=10.0,
                        gamma=0.99, epsilon=1e-8)
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = mean.astype(np.float64), var.astype(np.float64), float(n)
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=1664, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-5, target_kl=0.01, policy_kwargs=dict(net_arch=[256, 256], log_std_init=math.log(0.1)),
                verbose=0, device="cpu")
    pn, an = model.policy.mlp_extractor.policy_net, model.policy.action_net
    with torch.no_grad():
        pn[0].weight.copy_(net[0].weight); pn[0].bias.copy_(net[0].bias)
        pn[2].weight.copy_(net[2].weight); pn[2].bias.copy_(net[2].bias)
        an.weight.copy_(net[4].weight); an.bias.copy_(net[4].bias)
    model.save(path / "model")
    venv.save(str(path / "vecnormalize.pkl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dagger_v1")
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--teacher-episodes", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0, help="network initialisation and training throws")
    ap.add_argument("--randomize", default="{}", help='JSON {param: [lo, hi]} sampled per training throw')
    ap.add_argument("--overrides", default="{}", help='JSON fixed parameter values (e.g. wider ranges)')
    a = ap.parse_args()
    rnd, ov = json.loads(a.randomize), json.loads(a.overrides)
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    path = ROOT / "runs" / a.name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, path / "dagger.py")
    (path / "config.json").write_text(json.dumps(vars(a), indent=1))
    log = open(path / "iterations.jsonl", "a")
    net = torch.nn.Sequential(torch.nn.Linear(OBS, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, 7))
    betas = [1.0, 0.5, 0.25] + [0.0] * 100
    X = Y = W = mean = std = var = None
    seed, best = 200000 + 100000 * a.seed, -1.0
    with Pool(a.procs) as pool:
        for it in range(a.iters + 1):
            t0 = time.time()
            n_ep = a.teacher_episodes if it == 0 else a.episodes
            res = pool.map(episode, [(seed + k, betas[it], W, rnd, ov) for k in range(n_ep)], chunksize=4)
            seed += n_ep
            Xi = np.concatenate([r[0] for r in res]); Yi = np.concatenate([r[1] for r in res])
            metas = [r[2] for r in res]
            X = Xi if X is None else np.concatenate([X, Xi]); Y = Yi if Y is None else np.concatenate([Y, Yi])
            if it == 0:
                mean = X.mean(axis=0).astype(np.float64)
                var = np.maximum(X.var(axis=0), 1e-4).astype(np.float64)
                std = np.sqrt(var + 1e-8)
            loss = train(net, X, Y, mean, std, a.epochs, torch)
            W = weights(net, mean, std)
            sel = [r[2] for r in pool.map(episode, [(s, 0.0, W, rnd, ov) for s in SEL_SEEDS], chunksize=4)]
            score = float(np.mean([m["success"] for m in sel]))
            stu = [m for m in metas if not m["teacher"]]
            rec = dict(iter=it, time=time.strftime("%H:%M:%S"), samples=int(len(X)), loss=round(loss, 5),
                       rollout_success=float(np.mean([m["success"] for m in stu])) if stu else None,
                       teacher_success=float(np.mean([m["success"] for m in metas if m["teacher"]] or [np.nan])),
                       selection_success=score,
                       selection_median_force=float(np.median([m["max_force"] for m in sel if m["success"]] or [np.nan])),
                       wall_s=round(time.time() - t0, 1))
            if score >= best:
                best = score
                torch.save(net.state_dict(), path / "student_best.pt")
                export(net, mean, var, len(X), path, torch)
                rec["best"] = True
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    np.savez_compressed(path / "dataset.npz", X=X, Y=Y, mean=mean, var=var)


if __name__ == "__main__":
    main()
