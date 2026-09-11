"""Scenario generation, the conventional baseline, and metrics.

The evaluator owns target geometry and rewards. The conventional compass
controller below is a SEPARATELY LABELED baseline: it reads angular error
directly and is not the neural controller.

BLOCKED in this slice (needs the prepared connectome + biological interface
mapping): adapter calibration grid, neural controls/interventions, and the
shuffled-connectivity control. See README.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 evaluation (baseline slice).")
    p.add_argument("--gen-scenarios", type=int, metavar="N",
                   help="Generate N held-out scenarios and save them.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="scenarios.json")
    p.add_argument("--baseline", action="store_true",
                   help="Run the conventional compass baseline over scenarios.")
    p.add_argument("--scenarios", default="scenarios.json")
    args = p.parse_args()

    if args.gen_scenarios:
        sc = generate_scenarios(args.gen_scenarios, args.seed)
        save_scenarios(args.out, sc)
        print(f"wrote {len(sc)} scenarios to {args.out}")
    elif args.baseline:
        sc = load_scenarios(args.scenarios)
        summary = evaluate_controller(sc, conventional_baseline())
        print(json.dumps(summary.as_dict(), indent=2))
    else:
        p.print_help()
