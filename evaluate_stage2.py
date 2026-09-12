# evaluate_stage2.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from street import (
    ARENA_BOUND, MAX_SIM_TIME, SENSOR_RANGE, Control, StreetCarState, StreetScenario,
    StreetEpisodeResult, StreetLayout, initial_layouts, run_street_episode,
)

AVOIDANCE_GAINS = (0.0, 0.25, 0.5, 1.0)
BRAKE_DISTANCES = (2.0, 4.0, 6.0)


def _waypoints(layout: StreetLayout) -> list[tuple[float, float]]:
    edge = ARENA_BOUND - 4.0
    points = [(x, y) for x in layout.x_roads for y in layout.y_roads]
    points += [(x, -edge) for x in layout.x_roads] + [(x, edge) for x in layout.x_roads]
    points += [(-edge, y) for y in layout.y_roads] + [(edge, y) for y in layout.y_roads]
    return sorted(set(points))


def generate_scenario_splits(seed: int = 2):
    rng = random.Random(seed)
    layouts = initial_layouts()
    headings = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    heldout_counts = {"cross": 12, "regular": 44, "asymmetric": 44}
    training, heldout = [], []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        pairs = [(start, target) for start in points for target in points
                 if start != target and math.dist(start, target) >= 12.0]
        rng.shuffle(pairs)
        train_pairs = pairs[:8]
        test_pairs = pairs[8:8 + heldout_counts[name]]
        training += [StreetScenario(
            name, StreetCarState(*start, rng.choice(headings), 0.0), *target,
            timeout=MAX_SIM_TIME, label=f"train-{name}-{i:03d}",
        ) for i, (start, target) in enumerate(train_pairs)]
        heldout += [StreetScenario(
            name, StreetCarState(*start, rng.choice(headings), 0.0), *target,
            timeout=MAX_SIM_TIME, label=f"heldout-{name}-{i:03d}",
        ) for i, (start, target) in enumerate(test_pairs)]
    return training, heldout


def scenario_to_dict(s: StreetScenario) -> dict:
    return {
        "layout": s.layout,
        "start": {"x": s.start.x, "y": s.start.y,
                  "heading": s.start.heading, "speed": s.start.speed},
        "target_x": s.target_x, "target_y": s.target_y,
        "target_radius": s.target_radius, "timeout": s.timeout, "label": s.label,
    }


def scenario_from_dict(data: dict) -> StreetScenario:
    start = data["start"]
    return StreetScenario(data["layout"], StreetCarState(**start),
                          data["target_x"], data["target_y"],
                          data["target_radius"], data["timeout"], data["label"])


def save_scenarios(path: str, scenarios: list[StreetScenario]) -> None:
    Path(path).write_text(json.dumps([scenario_to_dict(s) for s in scenarios], indent=2) + "\n")


def load_scenarios(path: str) -> list[StreetScenario]:
    return [scenario_from_dict(data) for data in json.loads(Path(path).read_text())]


@dataclass(frozen=True)
class StreetSummary:
    trials: int
    arrivals: int
    collisions: int
    timeouts: int
    arrival_rate: float
    mean_arrival_time: float | None
    mean_route_length: float | None

    def as_dict(self) -> dict:
        return self.__dict__


def _run_controller(scenarios: list[StreetScenario],
                    layouts: dict[str, StreetLayout], controller, reset=None):
    results = []
    for scenario in scenarios:
        if reset is not None:
            reset()
        result = run_street_episode(scenario, layouts[scenario.layout], controller)
        results.append((scenario.layout, result))
    return results


def _summarize(results: list[StreetEpisodeResult]) -> StreetSummary:
    arrivals = [r for r in results if r.outcome == "arrival"]
    n = len(results)
    mean = lambda values: sum(values) / len(values) if values else None
    return StreetSummary(
        n, len(arrivals), sum(r.outcome == "collision" for r in results),
        sum(r.outcome == "timeout" for r in results), len(arrivals) / n if n else 0.0,
        mean([r.elapsed_time for r in arrivals]), mean([r.path_length for r in arrivals]),
    )


def evaluate_controller(scenarios: list[StreetScenario],
                        layouts: dict[str, StreetLayout], controller,
                        reset=None) -> StreetSummary:
    return _summarize([result for _, result in
                       _run_controller(scenarios, layouts, controller, reset)])


def evaluate_breakdown(scenarios: list[StreetScenario],
                       layouts: dict[str, StreetLayout], controller,
                       reset=None) -> dict:
    rows = _run_controller(scenarios, layouts, controller, reset)
    return {
        "overall": _summarize([result for _, result in rows]).as_dict(),
        "by_layout": {
            name: _summarize([result for layout, result in rows if layout == name]).as_dict()
            for name in layouts
        },
    }


def calibrate_stage2_adapter(scenarios, make_controller,
                             avoidance_gains=AVOIDANCE_GAINS,
                             brake_distances=BRAKE_DISTANCES):
    layouts, grid, best = initial_layouts(), [], None
    for gain in avoidance_gains:
        for distance in brake_distances:
            controller, reset = make_controller(gain, distance)
            summary = evaluate_controller(scenarios, layouts, controller, reset)
            row = {"avoidance_gain": gain, "brake_distance": distance,
                   **summary.as_dict()}
            grid.append(row)
            route = summary.mean_route_length if summary.mean_route_length is not None else math.inf
            key = (summary.arrivals, -summary.collisions, -summary.timeouts,
                   -route, -gain, -distance)
            if best is None or key > best[0]:
                best = (key, gain, distance)
    return best[1], best[2], grid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 street evaluation.")
    parser.add_argument("--freeze-scenarios", action="store_true",
                        help="Write the frozen training/held-out scenario splits.")
    parser.add_argument("--seed", type=int, default=2)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.freeze_scenarios:
        train, heldout = generate_scenario_splits(seed=args.seed)
        Path("runs/stage2").mkdir(parents=True, exist_ok=True)
        save_scenarios("runs/stage2/training.json", train)
        save_scenarios("runs/stage2/heldout.json", heldout)
        print(f"Froze {len(train)} training and {len(heldout)} held-out scenarios.")


if __name__ == "__main__":
    main()
