"""Stage 2e: warm-start reward fine-tuning (route-memory gate, take 3).

Purpose (see docs/superpowers/specs/2026-09-17-stage2e-warmstart-reward-design.md).
Stage 2d showed CEM-from-zero cannot bootstrap goal-reaching behavior that is
demonstrably reachable by this exact 150-parameter policy class: the deterministic
Stage 2c readouts score ~1.1 reward / 22-of-40 arrivals on the fitness scenarios,
while CEM-from-mu=0 plateaued at the no-op floor (-0.18). Stage 2e removes that
one failure mode: it INITIALIZES the CEM search at the Stage 2c readout and
fine-tunes small perturbations around it, so the sparse arrival term has gradient
signal from the first iteration.

Only the linear readout `W_out` (150 params) is trained; the reservoir is frozen
to the 2c winner (n=64, sr=0.8, leak=0.5, in=1.0, seed=0). Reward is the SAME
evaluator-only bounded-net-progress signal as Stage 2d (never a controller input).
The warm start trades away 2d's "clean reward-only" purity -- a deliberate,
documented choice -- to test the question 2d could not reach: given a half-working
policy, can reward improve it to the 90%/80% gate on a fresh, disjoint split?

Only a PASS is conclusive (evidence reward exploited the available policy state);
a failure stays bounded between optimization budget/search, capacity, and
observation limits. Does not touch the connectome or the frozen Stage 2/2b/2c/2d
artifacts; the spent Stage 2d gate is never rerun or replaced.
"""

from __future__ import annotations

import json

import numpy as np

from street import initial_layouts, run_street_episode
from evaluate_stage2 import load_scenarios
from stage2c import RecurrentController
import evaluate_stage2c as s2c
from stage2d import CEMConfig, cem_maximize, episode_reward, make_reservoir

# Deterministic sources for the warm start (the Stage 2c winner + its train demos).
STAGE2C_GATE_RESULTS = "runs/stage2c/gate_results.json"
STAGE2C_TRAIN_SPLIT = "runs/stage2b/sm_train_split.json"


def stage2c_winner_hp() -> dict:
    """The frozen Stage 2c winner hyperparameters (n=64, sr=0.8, leak=0.5,
    in=1.0, ridge=1e-4), read from the persisted gate artifact."""
    return json.load(open(STAGE2C_GATE_RESULTS))["hyperparameters"]


def warm_start_readout(recurrent: bool, layouts=None):
    """Reconstruct the deterministic Stage 2c readout for this model.

    Returns `(esn, theta0)` where `esn` is the frozen 2c-winner reservoir carrying
    the ridge-fit 2c readout and `theta0 = esn.W_out.flatten()` is the 150-vector
    warm start. Fully reproducible: the waypoint teacher and ridge fit are
    deterministic (ESN_SEED=0), so this matches the Stage 2c run bit-for-bit.
    """
    layouts = layouts or initial_layouts()
    hp = stage2c_winner_hp()
    feats, targs = s2c.collect_demos(load_scenarios(STAGE2C_TRAIN_SPLIT), layouts)
    esn = s2c._fit(recurrent, hp, feats, targs)   # deterministic ridge fit
    return esn, esn.W_out.flatten().copy()


def _mean_episode_reward(esn, scenarios, layouts) -> float:
    """The Stage 2e fitness: mean bounded-net-progress episode reward. Default so
    2e semantics are unchanged when no `fitness_fn` is supplied."""
    controller = RecurrentController(esn)
    total = 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        total += episode_reward(scenario, result)
    return total / len(scenarios)


def evaluate_policy(esn, scenarios, layouts, fitness_fn=None) -> dict:
    """Mean reward and arrival/collision/timeout counts for the policy currently
    on `esn`. When `fitness_fn` is given, also report `"fitness"` (the training
    objective's value); otherwise the 2e-shaped dict is returned unchanged."""
    controller = RecurrentController(esn)
    total = 0.0
    counts = {"arrival": 0, "collision": 0, "timeout": 0}
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        total += episode_reward(scenario, result)
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    out = {"mean_reward": total / len(scenarios), "n": len(scenarios),
           "arrivals": counts["arrival"], "collisions": counts["collision"],
           "timeouts": counts["timeout"]}
    if fitness_fn is not None:
        out["fitness"] = float(fitness_fn(esn, scenarios, layouts))
    return out


def train_readout_by_reward_warmstart(esn, theta0, fitness_scenarios, layouts,
                                      cfg: CEMConfig, fitness_fn=None):
    """Fine-tune `esn.W_out` by CEM from `theta0` against `fitness_fn` (default the
    Stage 2e mean episode reward). Uses the BEST-EVER candidate with a warm-start
    guard (never ships worse than theta0). Records the warm-start outcomes on the
    fitness set and the trained policy's outcomes. Returns `(esn, info)`."""
    fitness_fn = fitness_fn or _mean_episode_reward
    theta0 = np.asarray(theta0, dtype=float)
    d = theta0.size // 2

    def fitness(theta: np.ndarray) -> float:
        esn.W_out = theta.reshape(2, d)
        return fitness_fn(esn, fitness_scenarios, layouts)

    f0 = fitness(theta0)   # warm-start fitness -- never ship worse than this
    warmstart_outcomes = evaluate_policy(esn, fitness_scenarios, layouts, fitness_fn)
    _, info = cem_maximize(fitness, theta0.size, cfg, init_mu=theta0)
    if info["best"]["fitness"] >= f0:
        best_theta = np.asarray(info["best"]["theta"], dtype=float)
        info["improved_over_warmstart"] = True
    else:
        best_theta = theta0   # no perturbation beat the warm start; keep it
        info["improved_over_warmstart"] = False
    info["warmstart_fitness"] = float(f0)
    info["warmstart_outcomes"] = warmstart_outcomes
    info["best_fitness"] = float(max(info["best"]["fitness"], f0))
    esn.W_out = best_theta.reshape(2, d)
    info["best_theta"] = best_theta.tolist()
    info["best_outcomes"] = evaluate_policy(esn, fitness_scenarios, layouts, fitness_fn)
    return esn, info
