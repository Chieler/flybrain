"""Stage 2g: a fully-trainable recurrent baseline (route-memory gate, take 5).

Purpose (see docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md).
Stages 2d-2f froze the reservoir (W_in, W) to the 2c winner and trained only the
150-parameter readout; 2f cleared the optimization floor but not the 0.90/0.80 gate.
Stage 2g makes the recurrent weights themselves TRAINABLE: an n=8 echo-state net
whose W_in (8x11) + W (8x8) + W_out (2x19) = 190 params are optimized end-to-end by
block-scaled CEM against the frozen 2f aligned reward (no BPTT -- the stack has no
autograd and the reward is non-differentiable through the simulator). The matched
recurrence-off ablation freezes W=0 with unit leak and trains 126 params.

Honest attribution: n=64 -> n=8 is itself an architectural change, so this is not
literally one variable versus 2f; exact attribution would need a matched n=8
readout-only control (not run). A pass is meaningful under the pass-only asymmetry.
Does not touch the connectome or the frozen Stage 2/2b/2c/2d/2e/2f artifacts.
"""

from __future__ import annotations

import numpy as np

from street import Control, MAX_ACCEL, MAX_BRAKE, run_street_episode
from simulation import MAX_STEERING
from stage2c import EchoStateNetwork, N_FEATURES, RecurrentController, observation_features
from street import initial_layouts
from evaluate_stage2 import load_scenarios
from stage2f import episode_secondary
import evaluate_stage2c as s2c

N8_CONFIG = {"n_reservoir": 8, "spectral_radius": 0.8, "leak": 0.5, "input_scale": 1.0}
ESN_SEED = 0
RIDGE = 1e-3   # BC init ridge (initialization only; frozen, not tuned)
BC_DEMO_SPLIT = "runs/stage2b/sm_train_split.json"


def make_net(recurrent: bool) -> EchoStateNetwork:
    """The n=8 net (recurrent) or its recurrence-off ablation (W=0, leak=1)."""
    return EchoStateNetwork(
        n_reservoir=N8_CONFIG["n_reservoir"],
        spectral_radius=N8_CONFIG["spectral_radius"], leak=N8_CONFIG["leak"],
        input_scale=N8_CONFIG["input_scale"], seed=ESN_SEED, recurrent=recurrent)


def _dout(esn: EchoStateNetwork) -> int:
    return 1 + N_FEATURES + esn.n            # readout input width (19 for n=8)


def flatten_theta(esn: EchoStateNetwork, recurrent: bool) -> np.ndarray:
    """Flatten the trainable blocks in order [W_in, (W if recurrent), W_out]."""
    parts = [esn.W_in.flatten()]
    if recurrent:
        parts.append(esn.W.flatten())
    parts.append(esn.W_out.flatten())
    return np.concatenate(parts)


def set_theta(esn: EchoStateNetwork, theta: np.ndarray, recurrent: bool) -> None:
    """Inverse of `flatten_theta`. Leaves esn.W at its frozen zeros for the ablation."""
    theta = np.asarray(theta, dtype=float)
    i = 0
    n_in = esn.W_in.size
    esn.W_in = theta[i:i + n_in].reshape(esn.W_in.shape); i += n_in
    if recurrent:
        n_w = esn.n * esn.n
        esn.W = theta[i:i + n_w].reshape(esn.n, esn.n); i += n_w
    d = _dout(esn)
    esn.W_out = theta[i:i + 2 * d].reshape(2, d)


def trainable_slices(esn: EchoStateNetwork, recurrent: bool) -> list[tuple[int, int]]:
    """(start, stop) index ranges of each trainable block in flatten order."""
    slices, i = [], 0
    n_in = esn.W_in.size
    slices.append((i, i + n_in)); i += n_in
    if recurrent:
        n_w = esn.n * esn.n
        slices.append((i, i + n_w)); i += n_w
    d = _dout(esn)
    slices.append((i, i + 2 * d)); i += 2 * d
    return slices


def block_scales(theta: np.ndarray, slices) -> np.ndarray:
    """Per-parameter perturbation scale: 0.1 * max(RMS(block), 1e-3) per block."""
    theta = np.asarray(theta, dtype=float)
    scale = np.empty_like(theta)
    for start, stop in slices:
        rms = float(np.sqrt(np.mean(theta[start:stop] ** 2)))
        scale[start:stop] = 0.1 * max(rms, 1e-3)
    return scale


def warm_start_theta(recurrent: bool, layouts=None):
    """BC-initialize the net: ridge-fit W_out on the spent teacher-demo split
    (W_in/W at their seeded init), return (esn, theta0). Deterministic: the
    waypoint teacher and ridge fit are deterministic given ESN_SEED. BC is
    initialization only -- the clone is never scored."""
    layouts = layouts or initial_layouts()
    feats, targs = s2c.collect_demos(load_scenarios(BC_DEMO_SPLIT), layouts)
    esn = make_net(recurrent)
    esn.fit(feats, targs, ridge=RIDGE)
    return esn, flatten_theta(esn, recurrent)


class NoOpController:
    """Zero-control baseline: never steers or accelerates."""

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> Control:
        return Control(0.0, 0.0)


def policy_fitness_arrivals(controller, scenarios, layouts) -> tuple[int, float]:
    """Arrival-count-primary aligned fitness AND the arrival count for any
    controller. Mirrors stage2f.aligned_fitness (arrivals + mean(secondary)/4)
    but also returns arrivals, and accepts a bare controller (e.g. no-op)."""
    arrivals, secondary_total = 0, 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        if result.outcome == "arrival":
            arrivals += 1
        secondary_total += episode_secondary(scenario, result)
    fitness = arrivals + (secondary_total / len(scenarios)) / 4.0
    return arrivals, fitness


def go_no_go(esn, theta0, scales, scenarios, layouts, cfg, recurrent) -> dict:
    """One-iteration CEM probe: sample the iteration-0 population exactly as
    cem_maximize does (same seed/std), evaluate theta0 and every candidate, and
    require >=1 arrival AND a fitness above the no-op baseline. The full run at
    the same seed reproduces this population as its iteration 0."""
    theta0 = np.asarray(theta0, dtype=float)
    dim = theta0.size
    rng = np.random.default_rng(cfg.seed)                       # matches cem line
    sigma = np.full(dim, cfg.init_std)
    samples = rng.normal(np.zeros(dim), sigma, size=(cfg.population, dim))

    best_arr, best_fit, pop_best_fit = 0, -np.inf, -np.inf
    controller = RecurrentController(esn)
    for k, z in enumerate(np.vstack([np.zeros(dim), samples])):
        set_theta(esn, theta0 + scales * z, recurrent)
        arr, fit = policy_fitness_arrivals(controller, scenarios, layouts)
        best_arr = max(best_arr, arr)
        best_fit = max(best_fit, fit)
        if k > 0:                                               # population only
            pop_best_fit = max(pop_best_fit, fit)
    noop_arr, noop_fit = policy_fitness_arrivals(NoOpController(), scenarios, layouts)
    return {
        "probe_best_arrivals": int(best_arr),
        "population_best_fitness": float(pop_best_fit),
        "noop_fitness": float(noop_fit),
        "beats_noop": bool(best_fit > noop_fit),
        "passed": bool(best_arr >= 1 and best_fit > noop_fit),
    }
