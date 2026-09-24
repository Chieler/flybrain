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


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from evaluate_stage2 import GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE

PPO_SEEDS = (0, 1, 2)
TOTAL_TIMESTEPS = 1_000_000
PPO_KWARGS = {
    "n_steps": 256, "batch_size": 256, "n_epochs": 5,
    "learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95,
    "ent_coef": 0.01, "verbose": 0, "device": "cpu",
}


def make_vec_env(scenarios, seed, n_envs=8):
    return sb3_make_vec_env(
        lambda: StreetNavigationEnv(scenarios), n_envs=n_envs, seed=seed,
        vec_env_cls=DummyVecEnv)


def make_model(recurrent, scenarios, seed, n_envs=8, overrides=None):
    kwargs = dict(PPO_KWARGS)
    kwargs.update(overrides or {})
    env = make_vec_env(scenarios, seed, n_envs)
    cls, policy = ((RecurrentPPO, "MlpLstmPolicy") if recurrent
                   else (PPO, "MlpPolicy"))
    return cls(policy, env, seed=seed, **kwargs)


def train_model(recurrent, scenarios, seed, path):
    model = make_model(recurrent, scenarios, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(path)
    return model


def score_model(model, scenarios, layouts=None):
    layouts = layouts or initial_layouts()
    outcomes = []
    by_layout = {name: [] for name in layouts}
    for scenario in scenarios:
        env = StreetNavigationEnv([scenario], layouts)
        obs, _ = env.reset(options={"scenario_index": 0})
        state, episode_start = None, np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_start[:] = done
        outcomes.append(info["outcome"])
        by_layout[scenario.layout].append(info["outcome"])
    arrivals = outcomes.count("arrival")
    rates = {name: values.count("arrival") / len(values)
             for name, values in by_layout.items() if values}
    overall = arrivals / len(outcomes)
    return {
        "trials": len(outcomes), "arrivals": arrivals,
        "collisions": outcomes.count("collision"),
        "timeouts": outcomes.count("timeout"),
        "overall_arrival_rate": overall,
        "by_layout_arrival_rate": rates,
        "by_layout": {name: {"trials": len(values),
                              "arrivals": values.count("arrival")}
                      for name, values in by_layout.items() if values},
        "passes_gate": overall >= GATE_OVERALL_RATE and all(
            rate >= GATE_PER_LAYOUT_RATE for rate in rates.values()),
    }


def gate_score(score):
    return min(
        score["overall_arrival_rate"] / GATE_OVERALL_RATE,
        *(rate / GATE_PER_LAYOUT_RATE
          for rate in score["by_layout_arrival_rate"].values()))
