"""Custom Gymnasium environment: a tiny 5x5 GridWorld.

This is the "build an RL environment" artifact. The agent starts in a corner and
must reach the opposite-corner goal. It is deliberately simple and fast so the
same PPO recipe used on the MinAtar benchmarks can be shown to learn here first.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, size=5, max_steps=100, render_mode=None):
        super().__init__()
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        # obs = [agent_row, agent_col, goal_row, goal_col], each normalized to [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        # actions: 0 up, 1 down, 2 left, 3 right
        self.action_space = spaces.Discrete(4)
        self._agent = None
        self._goal = None
        self._steps = 0

    def _obs(self):
        s = max(self.size - 1, 1)
        return np.array(
            [self._agent[0] / s, self._agent[1] / s, self._goal[0] / s, self._goal[1] / s],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent = np.array([0, 0], dtype=np.int64)
        self._goal = np.array([self.size - 1, self.size - 1], dtype=np.int64)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[int(action)]
        self._agent = np.clip(self._agent + np.array([dr, dc]), 0, self.size - 1)
        self._steps += 1
        reached = bool(np.array_equal(self._agent, self._goal))
        reward = 1.0 if reached else -0.01
        terminated = reached
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        grid[self._goal[0]][self._goal[1]] = "G"
        grid[self._agent[0]][self._agent[1]] = "A"
        return "\n".join(" ".join(row) for row in grid)


# Register on import (guarded so re-import does not raise).
try:
    register(id="GridWorld-v0", entry_point="envs.gridworld_env:GridWorldEnv", max_episode_steps=100)
except gym.error.Error:
    pass
