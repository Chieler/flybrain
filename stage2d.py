"""Stage 2d: a reward-trained same-observation baseline (route-memory gate, take 2).

Purpose (see docs/superpowers/specs/2026-09-16-stage2d-reward-baseline-design.md).
Stage 2c ruled out one setup -- behavior-cloning a fixed-reservoir ESN against the
privileged waypoint teacher reached 57%, no meaningful recurrence benefit. That
does NOT falsify route memory; it says imitation of a teacher carrying hidden
state is the wrong instrument.

Stage 2d changes exactly ONE variable versus 2c: the training signal. Instead of
cloning teacher actions, it optimizes the policy against an EPISODE REWARD for
reaching the goal -- a memory-requiring signal. Everything else is held fixed:
the observation interface, the seeded fixed recurrent reservoir (frozen to the 2c
winner: n=64, spectral_radius=0.8, leak=0.5, input_scale=1.0), and the
recurrent-vs-memoryless-ablation comparison. Only the linear readout `W_out`
(150 parameters) is trained, by the Cross-Entropy Method (pure NumPy, no BPTT).

Reward is EVALUATOR-ONLY privileged geometry (bounded net progress from start to
final position, an arrival bonus that dominates the maximum shaping, a collision
penalty, and a small time cost). It enters the reward, NEVER the controller's
inputs -- the legitimate dopamine-style teaching role. Only a PASS on the fresh,
disjoint one-shot gate is conclusive; a failure stays confounded between
optimization budget, capacity, and observation limits.

Not neural in the connectome sense; does not touch the connectome or the frozen
Stage 2 / 2b / 2c artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from street import MAX_SIM_TIME, run_street_episode
from stage2c import N_FEATURES, EchoStateNetwork, RecurrentController

# Frozen reservoir config -- the Stage 2c winner (runs/stage2c/gate_results.json).
RESERVOIR_CONFIG = {
    "n_reservoir": 64, "spectral_radius": 0.8, "leak": 0.5, "input_scale": 1.0,
}
ESN_SEED = 0

# Frozen reward coefficients (design choices; not tuned on the gate). The arrival
# bonus (2.0) dominates the maximum possible shaping contribution (progress <= 1).
W_ARRIVE = 2.0
W_PROGRESS = 1.0
W_COLLIDE = 1.0
W_TIME = 0.2


def _final_xy(scenario, result) -> tuple[float, float]:
    """Final (x, y) of an episode -- last recorded trajectory point (run with
    record=True); falls back to the start if no trajectory was recorded."""
    if result.trajectory:
        return result.trajectory[-1][0], result.trajectory[-1][1]
    return scenario.start.x, scenario.start.y


def episode_reward(scenario, result) -> float:
    """Bounded-net-progress reward for one episode.

    Net progress uses the FINAL position (not accumulated positive progress and
    not closest approach), so darting near the goal and leaving cannot be farmed.
    Privileged distances are used here only -- never as controller inputs.
    """
    d0 = math.hypot(scenario.target_x - scenario.start.x,
                    scenario.target_y - scenario.start.y)
    fx, fy = _final_xy(scenario, result)
    d_final = math.hypot(scenario.target_x - fx, scenario.target_y - fy)
    progress = (d0 - d_final) / d0 if d0 > 0.0 else 0.0
    progress = max(-1.0, min(1.0, progress))
    reward = W_PROGRESS * progress
    reward += W_ARRIVE if result.outcome == "arrival" else 0.0
    reward -= W_COLLIDE if result.outcome == "collision" else 0.0
    reward -= W_TIME * (result.elapsed_time / MAX_SIM_TIME)
    return reward


@dataclass
class CEMConfig:
    """Cross-Entropy Method budget. The Stage 2d run freezes population=64,
    n_iter=25, elite_frac=0.20 (see the spec); the rest have safe defaults."""
    population: int = 64
    n_iter: int = 25
    elite_frac: float = 0.20
    init_std: float = 0.5
    std_floor: float = 1e-3
    seed: int = 0


def cem_maximize(fitness_fn, dim: int, cfg: CEMConfig):
    """Maximize `fitness_fn` over R^dim by CEM with diagonal covariance.

    Deterministic given `cfg.seed`. Returns `(mu, info)` where `mu` is the final
    elite mean and `info["history"]` records per-iteration best/mean fitness.
    """
    rng = np.random.default_rng(cfg.seed)
    mu = np.zeros(dim)
    sigma = np.full(dim, cfg.init_std)
    n_elite = max(1, int(round(cfg.population * cfg.elite_frac)))
    history = []
    for it in range(cfg.n_iter):
        samples = rng.normal(mu, sigma, size=(cfg.population, dim))
        scores = np.array([fitness_fn(s) for s in samples])
        elite_idx = np.argsort(scores)[-n_elite:]
        elites = samples[elite_idx]
        mu = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0), cfg.std_floor)
        history.append({"iter": it, "best": float(scores.max()),
                        "mean": float(scores.mean())})
    return mu, {"history": history}


def make_reservoir(recurrent: bool) -> EchoStateNetwork:
    """Fixed Stage 2c-winner reservoir (recurrent) or its matched ablation."""
    return EchoStateNetwork(
        n_reservoir=RESERVOIR_CONFIG["n_reservoir"],
        spectral_radius=RESERVOIR_CONFIG["spectral_radius"],
        leak=RESERVOIR_CONFIG["leak"], input_scale=RESERVOIR_CONFIG["input_scale"],
        seed=ESN_SEED, recurrent=recurrent)


def train_readout_by_reward(esn: EchoStateNetwork, fitness_scenarios, layouts,
                            cfg: CEMConfig):
    """Optimize only `esn.W_out` against mean episode reward, by CEM.

    Fitness is the mean reward over `fitness_scenarios` (deterministic: the
    simulator and controller are deterministic given the readout). Sets the fitted
    readout on `esn` and returns `(esn, info)`.
    """
    d = 1 + N_FEATURES + esn.n
    dim = 2 * d
    controller = RecurrentController(esn)

    def fitness(theta: np.ndarray) -> float:
        esn.W_out = theta.reshape(2, d)
        total = 0.0
        for scenario in fitness_scenarios:
            controller.reset()
            result = run_street_episode(scenario, layouts[scenario.layout],
                                        controller, record=True)
            total += episode_reward(scenario, result)
        return total / len(fitness_scenarios)

    theta, info = cem_maximize(fitness, dim, cfg)
    esn.W_out = theta.reshape(2, d)
    info["final_mu_fitness"] = float(fitness(theta))
    return esn, info
