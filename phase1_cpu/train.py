"""The ONE PPO recipe, parameterized by --env-id.

Identical hyperparameters (from config.yaml) train the custom GridWorld and every
MinAtar benchmark; only the feature extractor differs (tiny CNN for image obs).
Logs per-step metrics (csv/json/tensorboard), evaluation series, and checkpoints.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from envs import make_env, load_config, policy_kwargs_for, slug


def build_vec(env_id, seed, n_envs):
    # DummyVecEnv (not Subproc) avoids pickling the custom registration/wrappers.
    return VecMonitor(DummyVecEnv([make_env(env_id, seed, i) for i in range(n_envs)]))


def train(env_id, timesteps, seed, outdir, cfg=None):
    cfg = cfg or load_config()
    r, ev = cfg["recipe"], cfg["eval"]
    os.makedirs(outdir, exist_ok=True)
    n_envs = r["n_envs"]

    venv = build_vec(env_id, seed, n_envs)
    eval_venv = build_vec(env_id, seed + 1000, 1)

    model = PPO(
        r["policy"], venv, seed=seed, device="cpu", verbose=0,
        learning_rate=r["learning_rate"], n_steps=r["n_steps"], batch_size=r["batch_size"],
        n_epochs=r["n_epochs"], gamma=r["gamma"], gae_lambda=r["gae_lambda"],
        ent_coef=r["ent_coef"], vf_coef=r["vf_coef"], max_grad_norm=r["max_grad_norm"],
        policy_kwargs=policy_kwargs_for(env_id, cfg),
    )
    model.set_logger(configure(outdir, ["csv", "json", "tensorboard"]))

    callbacks = [
        EvalCallback(
            eval_venv, log_path=outdir, best_model_save_path=outdir,
            eval_freq=max(ev["eval_freq"] // n_envs, 1),
            n_eval_episodes=ev["n_eval_episodes"], deterministic=True, verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(ev["save_freq"] // n_envs, 1),
            save_path=os.path.join(outdir, "ckpts"), name_prefix="ppo",
        ),
    ]
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(outdir, "final_model"))
    venv.close()
    eval_venv.close()
    return os.path.join(outdir, "final_model.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    cfg = load_config()
    ts = args.timesteps or cfg["run"]["timesteps_fast"]
    outdir = args.outdir or os.path.join("results", slug(args.env_id), f"seed{args.seed}")
    print(f"[train] {args.env_id} seed={args.seed} timesteps={ts} -> {outdir}")
    path = train(args.env_id, ts, args.seed, outdir, cfg)
    print(f"[train] saved {path}")
