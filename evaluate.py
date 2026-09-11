"""Scenario generation, the conventional baseline, calibration, controls, metrics.

The evaluator owns target geometry and rewards. The conventional compass
controller below is a SEPARATELY LABELED baseline: it reads angular error
directly and is not the neural controller.

Adapter calibration (declared grid), the neural controls/interventions
(zero/random steering, cue-withheld, pathway-silenced), and the
shuffled-connectivity control all live here. Running them against the real
graph needs a prepared connectome under `data/` (see SETUP.md); the logic is
unit-tested on synthetic graphs so it is verifiable without that download.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from brain import Adapter, Brain, NeuralController, RateParams
from simulation import (
    ARENA_BOUND, MAX_STEERING, TARGET_RADIUS,
    CarState, Scenario, EpisodeResult, run_episode, wrap_angle,
)

ARRIVAL_REWARD = 1.0
BOUNDARY_REWARD = -1.0
TIME_COST = 0.01  # per simulated second


def episode_return(result: EpisodeResult) -> float:
    base = {"arrival": ARRIVAL_REWARD, "boundary": BOUNDARY_REWARD}.get(result.outcome, 0.0)
    return base - TIME_COST * result.elapsed_time


def conventional_baseline(k: float = 4.0):
    """Proportional compass controller. LABELED: reads angular error directly."""
    def controller(obs):
        err = wrap_angle(obs.goal_bearing - obs.heading)
        return max(-MAX_STEERING, min(MAX_STEERING, k * err))
    return controller


def generate_scenarios(n: int, seed: int) -> list[Scenario]:
    """Held-out scenarios: start in [-8,8]^2, target offset 6-15 at uniform
    bearing, uniform initial heading. Reject any target region not fully
    inside the arena; only accepted scenarios are returned."""
    rng = random.Random(seed)
    out: list[Scenario] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError(f"could not sample {n} valid scenarios")
        sx = rng.uniform(-8.0, 8.0)
        sy = rng.uniform(-8.0, 8.0)
        heading = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(6.0, 15.0)
        bearing = rng.uniform(-math.pi, math.pi)
        tx = sx + dist * math.cos(bearing)
        ty = sy + dist * math.sin(bearing)
        # Whole target region must lie inside the arena.
        if abs(tx) + TARGET_RADIUS > ARENA_BOUND or abs(ty) + TARGET_RADIUS > ARENA_BOUND:
            continue
        out.append(Scenario(CarState(sx, sy, heading), tx, ty,
                            label=f"heldout-{len(out)}"))
    return out


def scenario_to_dict(s: Scenario) -> dict:
    return {
        "start": {"x": s.start.x, "y": s.start.y, "heading": s.start.heading},
        "target_x": s.target_x, "target_y": s.target_y,
        "target_radius": s.target_radius, "timeout": s.timeout, "label": s.label,
    }


def scenario_from_dict(d: dict) -> Scenario:
    st = d["start"]
    return Scenario(CarState(st["x"], st["y"], st["heading"]),
                    d["target_x"], d["target_y"],
                    d.get("target_radius", TARGET_RADIUS),
                    d.get("timeout", 30.0), d.get("label", ""))


def save_scenarios(path: str, scenarios: list[Scenario]) -> None:
    with open(path, "w") as f:
        json.dump([scenario_to_dict(s) for s in scenarios], f, indent=2)


def load_scenarios(path: str) -> list[Scenario]:
    with open(path) as f:
        return [scenario_from_dict(d) for d in json.load(f)]


@dataclass
class Summary:
    trials: int
    arrivals: int
    boundaries: int
    timeouts: int
    mean_arrival_time: float | None    # among successes
    mean_route_length: float | None    # among successes
    mean_return: float

    def as_dict(self) -> dict:
        return self.__dict__


def evaluate_controller(scenarios: list[Scenario], controller,
                        reset=None) -> Summary:
    """Run every scenario; report numerators and denominators, incl. failures."""
    arrivals = boundaries = timeouts = 0
    arrival_times: list[float] = []
    route_lengths: list[float] = []
    returns: list[float] = []
    for s in scenarios:
        if reset is not None:
            reset()
        r = run_episode(s, controller)
        r.return_ = episode_return(r)
        returns.append(r.return_)
        if r.outcome == "arrival":
            arrivals += 1
            arrival_times.append(r.elapsed_time)
            route_lengths.append(r.path_length)
        elif r.outcome == "boundary":
            boundaries += 1
        else:
            timeouts += 1
    n = len(scenarios)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return Summary(n, arrivals, boundaries, timeouts,
                   mean(arrival_times), mean(route_lengths),
                   sum(returns) / n if n else 0.0)


# --- Declared adapter calibration grid (plan §Calibration and evaluation) ---
CALIB_GAINS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # rad per unit activity
CALIB_BIASES = (-0.05, 0.0, 0.05)                           # rad


def calibrate_adapter(scenarios, make_controller, gains=CALIB_GAINS,
                      biases=CALIB_BIASES):
    """Exhaust the declared gain×bias grid; return (gain, bias, grid).

    `make_controller(gain, bias) -> (controller, reset)`. Ties broken by lower
    absolute bias, then smaller gain (plan). This is adapter calibration, not a
    claim of learning inside the fly brain; the budget is fixed and reported,
    never silently expanded.
    """
    grid = []
    best = None  # (key, gain, bias)
    for g in gains:
        for b in biases:
            controller, reset = make_controller(g, b)
            summ = evaluate_controller(scenarios, controller, reset)
            grid.append({"gain": g, "bias": b, "mean_return": summ.mean_return,
                         "arrivals": summ.arrivals, "trials": summ.trials})
            key = (summ.mean_return, -abs(b), -g)  # max return, then |bias|, then gain
            if best is None or key > best[0]:
                best = (key, g, b)
    return best[1], best[2], grid


def neural_factory(brain: Brain, neural_updates: int = 2):
    """make_controller for calibrate_adapter, backed by a real/synthetic Brain."""
    def make(gain, bias):
        ctrl = NeuralController(brain, Adapter(gain=gain, bias=bias), neural_updates)
        return ctrl, ctrl.reset
    return make


# --- Controls (no neural graph) ---

def zero_steering():
    return (lambda obs: 0.0), None


def random_steering(seed: int):
    """Seeded uniform steering. reset() re-seeds so replay is deterministic."""
    state = {"rng": random.Random(seed)}

    def controller(obs):
        return state["rng"].uniform(-MAX_STEERING, MAX_STEERING)

    def reset():
        state["rng"] = random.Random(seed)
    return controller, reset


# --- Neural interventions (share Brain wiring, lesion one path) ---

def cue_withheld_controller(brain: Brain, adapter: Adapter,
                            neural_updates: int = 2, role: str = "goal"):
    """Neural controller with one input population's cue zeroed (no retuning)."""
    entry = brain.input_map.get(role)
    idx = entry[0] if entry is not None else np.array([], dtype=int)

    def controller(obs):
        stim = brain.encode(obs.heading, obs.goal_bearing)
        stim[idx] = 0.0
        for _ in range(neural_updates):
            brain.step(stim)
        left, right = brain.outputs()
        return adapter(left, right)
    return controller, brain.reset


def pathway_silenced_controller(brain: Brain, adapter: Adapter,
                                neural_updates: int = 2):
    """Neural controller with both mapped output populations clamped to 0.

    Lesions the PFL3 steering readout (also removes its recurrent contribution),
    so the adapter sees (0, 0) and steering collapses to its bias. No retuning.
    """
    silenced = np.concatenate([brain.output_map["left"], brain.output_map["right"]])

    def controller(obs):
        stim = brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(neural_updates):
            brain.step(stim)
            brain.activity[silenced] = 0.0
        left, right = brain.outputs()
        return adapter(left, right)
    return controller, brain.reset


def shuffle_connectivity(W, seed: int):
    """Shuffle anatomical destinations, preserving per-source identity, sign, and
    outgoing-weight multiset (plan's shuffled-connectivity control).

    Preserves: per-source out-degree and its multiset of signed weights.
    Does NOT preserve: destination identity, per-destination in-degree, or any
    dest-side structure. Operate on the pre-normalization signed weights; Brain
    row-normalizes on construction.
    """
    C = sp.csc_array(W)  # columns = presynaptic sources
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    new_rows = np.empty_like(C.indices)
    for c in range(C.shape[1]):
        s, e = C.indptr[c], C.indptr[c + 1]
        k = e - s
        if k:
            # replace=False keeps within-column dest distinct (no CSR summing).
            # ponytail: O(sources) Python loop, run once offline; vectorize if it bites.
            new_rows[s:e] = rng.choice(n, size=k, replace=False)
    shuffled = sp.csc_array((C.data.copy(), new_rows, C.indptr.copy()), shape=C.shape)
    return sp.csr_array(shuffled)


# --- Checkpoint I/O ---

def save_checkpoint(path: str, gain: float, bias: float,
                    params: RateParams, data_dir: str) -> None:
    with open(path, "w") as f:
        json.dump({"gain": gain, "bias": bias, "data_dir": data_dir,
                   "params": params.__dict__}, f, indent=2)


def load_checkpoint(path: str) -> dict:
    with open(path) as f:
        cp = json.load(f)
    cp["params"] = RateParams(**cp.get("params", {}))
    return cp


def run_controls(data_dir: str, gain: float, bias: float, params: RateParams,
                 heldout: list[Scenario], calib: list[Scenario],
                 shuffle_seeds=(1, 2, 3)) -> dict:
    """Held-out neural controller plus every declared control/intervention.

    Shuffled connectivity is RE-calibrated with exactly the same grid and
    interfaces (three fixed seeds); all other controls use the frozen adapter.
    Returns {name: Summary.as_dict()}.
    """
    W = sp.load_npz(f"{data_dir}/weights_signed.npz")
    base = Brain.load(data_dir, params)
    adapter = Adapter(gain=gain, bias=bias)
    results: dict = {}

    def record(name, controller, reset):
        results[name] = evaluate_controller(heldout, controller, reset).as_dict()

    ctrl = NeuralController(base, adapter)
    record("neural", ctrl, ctrl.reset)
    record("conventional", conventional_baseline(), None)
    record("zero", *zero_steering())
    record("random", *random_steering(seed=0))
    record("cue_withheld", *cue_withheld_controller(base, adapter))
    record("pathway_silenced", *pathway_silenced_controller(base, adapter))

    for seed in shuffle_seeds:
        sh = Brain(shuffle_connectivity(W, seed), base.input_map,
                   base.output_map, params)
        g, b, _ = calibrate_adapter(calib, neural_factory(sh))
        c = NeuralController(sh, Adapter(gain=g, bias=b))
        results[f"shuffled_seed{seed}"] = {
            "calibrated_gain": g, "calibrated_bias": b,
            **evaluate_controller(heldout, c, c.reset).as_dict()}
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 evaluation (baseline slice).")
    p.add_argument("--gen-scenarios", type=int, metavar="N",
                   help="Generate N held-out scenarios and save them.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="scenarios.json")
    p.add_argument("--baseline", action="store_true",
                   help="Run the conventional compass baseline over scenarios.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the declared adapter grid on a real graph; write a checkpoint.")
    p.add_argument("--controls", action="store_true",
                   help="Run held-out neural controller + all controls from a checkpoint.")
    p.add_argument("--scenarios", default="scenarios.json",
                   help="Held-out scenarios (also the calibration set unless --calib-scenarios).")
    p.add_argument("--calib-scenarios", help="Calibration scenarios (defaults to --scenarios).")
    p.add_argument("--data", help="Prepared connectome dir (needed for --calibrate/--controls).")
    p.add_argument("--checkpoint", default="checkpoint.json")
    args = p.parse_args()

    if args.gen_scenarios:
        sc = generate_scenarios(args.gen_scenarios, args.seed)
        save_scenarios(args.out, sc)
        print(f"wrote {len(sc)} scenarios to {args.out}")
    elif args.baseline:
        sc = load_scenarios(args.scenarios)
        summary = evaluate_controller(sc, conventional_baseline())
        print(json.dumps(summary.as_dict(), indent=2))
    elif args.calibrate:
        if not args.data:
            p.error("--calibrate needs --data (prepared connectome dir)")
        calib = load_scenarios(args.calib_scenarios or args.scenarios)
        brain = Brain.load(args.data)
        gain, bias, grid = calibrate_adapter(calib, neural_factory(brain))
        save_checkpoint(args.checkpoint, gain, bias, brain.params, args.data)
        print(json.dumps({"gain": gain, "bias": bias, "budget": len(grid),
                          "checkpoint": args.checkpoint}, indent=2))
    elif args.controls:
        if not args.data:
            p.error("--controls needs --data (prepared connectome dir)")
        cp = load_checkpoint(args.checkpoint)
        heldout = load_scenarios(args.scenarios)
        calib = load_scenarios(args.calib_scenarios or args.scenarios)
        res = run_controls(args.data, cp["gain"], cp["bias"], cp["params"],
                           heldout, calib)
        with open(args.out if args.out != "scenarios.json" else "results.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
    else:
        p.print_help()
