"""Shared env helpers for Phase 1.

Registers the custom GridWorld-v0 on import, lazily registers the MinAtar suite,
and exposes make_env / policy_kwargs_for so train.py, evaluate.py, benchmark.py all
build environments the same way (the "one recipe" contract).
"""
import os
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from . import gridworld_env  # noqa: F401  (registers GridWorld-v0)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_MINATAR_REGISTERED = False
# MinAtar's gym env has no step limit; a deterministic policy can produce a
# non-terminating episode that hangs evaluation. Cap episodes to guarantee termination.
MINATAR_MAX_STEPS = 1000


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def is_minatar(env_id):
    return env_id.startswith("MinAtar/")


def _ensure_minatar():
    """minatar defines register_envs() but does not call it on import (gymnasium 1.x
    dropped auto-registration), so we call it once ourselves."""
    global _MINATAR_REGISTERED
    if _MINATAR_REGISTERED:
        return
    from minatar.gym import register_envs
    try:
        register_envs()
    except gym.error.Error:
        pass
    _MINATAR_REGISTERED = True


class MinatarChannelFirst(gym.ObservationWrapper):
    """(H, W, C) bool -> (C, H, W) float32 in [0,1] so the small CNN can consume it."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32).transpose(2, 0, 1)


def minatar_available():
    try:
        _ensure_minatar()
        e = gym.make("MinAtar/Breakout-v1")
        e.close()
        return True
    except Exception:
        return False


def make_env(env_id, seed=0, idx=0):
    """Return a thunk that builds a single (wrapped) environment."""
    def thunk():
        if is_minatar(env_id):
            _ensure_minatar()
            env = gym.make(env_id, max_episode_steps=MINATAR_MAX_STEPS)
            env = MinatarChannelFirst(env)
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return thunk


def policy_kwargs_for(env_id, cfg):
    """MinAtar (image obs) swaps in the tiny CNN; everything else uses the default MLP."""
    if is_minatar(env_id):
        from .minatar_cnn import MinatarCNN
        return dict(
            features_extractor_class=MinatarCNN,
            features_extractor_kwargs=dict(features_dim=cfg["recipe"]["features_dim"]),
            normalize_images=False,
        )
    return None


def slug(env_id):
    return env_id.replace("/", "_")
