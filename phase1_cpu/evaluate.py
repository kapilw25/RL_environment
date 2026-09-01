"""Greedy evaluation + random baseline.

run_episodes runs N episodes of a policy (or a random policy when model is None)
and returns mean +/- std return. Reused by benchmark.py for the results table.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

from envs import make_env


def run_episodes(model, env_id, n_episodes=30, seed=123, deterministic=True):
    env = make_env(env_id, seed)()
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_final(model_path, env_id, n_episodes=30, seed=123):
    model = PPO.load(model_path, device="cpu")
    ours = run_episodes(model, env_id, n_episodes, seed)
    rand = run_episodes(None, env_id, n_episodes, seed)
    return ours, rand


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--env-id", required=True)
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    (om, osd), (rm, rsd) = evaluate_final(args.model, args.env_id, args.episodes)
    print(f"{args.env_id}: ours {om:.2f} +/- {osd:.2f} | random {rm:.2f} +/- {rsd:.2f}")
