# stage2h.py
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation import MAX_STEERING
from stage2c import observation_features
from street import (Control, MAX_ACCEL, MAX_BRAKE, StreetEpisode,
                    initial_layouts)

ARRIVAL_BONUS = 10.0
COLLISION_PENALTY = 4.0
PROGRESS_SCALE = 2.0
STEP_COST = 0.001


def action_to_control(action) -> Control:
    steer, drive = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    acceleration = drive * (MAX_ACCEL if drive >= 0.0 else MAX_BRAKE)
    return Control(float(steer * MAX_STEERING), float(acceleration))


def episode_reward(previous_distance, new_distance, initial_distance, outcome):
    reward = PROGRESS_SCALE * (previous_distance - new_distance) / initial_distance - STEP_COST
    if outcome == "arrival":
        reward += ARRIVAL_BONUS
    elif outcome == "collision":
        reward -= COLLISION_PENALTY
    return float(reward)


class StreetNavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenarios, layouts=None, seed=0):
        super().__init__()
        if not scenarios:
            raise ValueError("StreetNavigationEnv requires at least one scenario")
        self.scenarios = list(scenarios)
        self.layouts = layouts or initial_layouts()
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(
            np.array([-1.0] * 4 + [0.0] * 6, dtype=np.float32),
            np.ones(10, dtype=np.float32))
        self.episode = None
        self.scenario_index = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.scenario_index = int(options.get(
            "scenario_index", self.rng.integers(len(self.scenarios))))
        scenario = self.scenarios[self.scenario_index]
        self.episode = StreetEpisode(scenario, self.layouts[scenario.layout])
        self.initial_distance = math.hypot(
            scenario.target_x - scenario.start.x,
            scenario.target_y - scenario.start.y)
        return self._observation(), {
            "scenario_index": self.scenario_index, "layout": scenario.layout}

    def _observation(self):
        return observation_features(self.episode.observe()).astype(np.float32)

    def step(self, action):
        scenario = self.episode.scenario
        previous = math.hypot(
            scenario.target_x - self.episode.state.x,
            scenario.target_y - self.episode.state.y)
        result = self.episode.step(action_to_control(action))
        current = math.hypot(
            scenario.target_x - self.episode.state.x,
            scenario.target_y - self.episode.state.y)
        outcome = result.outcome if result is not None else None
        terminated = outcome in ("arrival", "collision")
        truncated = outcome == "timeout"
        observation = (np.zeros(10, dtype=np.float32) if result is not None
                       else self._observation())
        info = {"outcome": outcome, "layout": scenario.layout}
        return observation, episode_reward(
            previous, current, self.initial_distance, outcome), terminated, truncated, info
