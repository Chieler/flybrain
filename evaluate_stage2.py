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
from brain import Brain, Stage2Adapter, Stage2NeuralController
from evaluate import load_checkpoint
from simulation import MAX_STEERING, wrap_angle

AVOIDANCE_GAINS = (0.0, 0.25, 0.5, 1.0)
BRAKE_DISTANCES = (2.0, 4.0, 6.0)

CONTROL_NAMES = (
    "neural", "direct_compass", "sensor_only", "neural_no_sensors",
    "goal_cue_withheld", "pfl3_silenced",
)


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


def stage2_factory(brain: Brain, stage1_checkpoint: dict):
    def make(avoidance_gain: float, brake_distance: float):
        adapter = Stage2Adapter(stage1_checkpoint["gain"], stage1_checkpoint["bias"],
                                avoidance_gain, brake_distance)
        controller = Stage2NeuralController(brain, adapter)
        return controller, controller.reset
    return make


def sensor_only_controller(adapter: Stage2Adapter):
    local = Stage2Adapter(0.0, 0.0, adapter.avoidance_gain,
                          adapter.brake_distance, adapter.cruise_speed)
    return lambda obs: local(0.0, 0.0, obs.ranges, obs.speed)


def direct_compass_controller(adapter: Stage2Adapter):
    local = Stage2Adapter(1.0, 0.0, adapter.avoidance_gain,
                          adapter.brake_distance, adapter.cruise_speed)
    def controller(obs):
        error = wrap_angle(obs.goal_bearing - obs.heading)
        signal = max(-MAX_STEERING, min(MAX_STEERING, 4.0 * error))
        return local(signal, 0.0, obs.ranges, obs.speed)
    return controller


def _controlled_neural(brain: Brain, adapter: Stage2Adapter,
                       withhold_goal: bool = False, silence_pfl3: bool = False,
                       ranges_override: tuple[float, ...] | None = None):
    goal = brain.input_map.get("goal")
    goal_indices = goal[0] if goal is not None else None
    outputs = list(brain.output_map["left"]) + list(brain.output_map["right"])
    def controller(obs):
        stimulus = brain.encode(obs.heading, obs.goal_bearing)
        if withhold_goal and goal_indices is not None:
            stimulus[goal_indices] = 0.0
        for _ in range(2):
            brain.step(stimulus)
            if silence_pfl3:
                brain.activity[outputs] = 0.0
        left, right = brain.outputs()
        ranges = obs.ranges if ranges_override is None else ranges_override
        return adapter(left, right, ranges, obs.speed)
    return controller, brain.reset


def run_stage2_controls(data_dir: str, stage1_checkpoint: dict,
                        stage2_checkpoint: dict,
                        heldout: list[StreetScenario]) -> dict:
    layouts = initial_layouts()
    brain = Brain.load(data_dir, stage1_checkpoint["params"])
    adapter = Stage2Adapter(stage1_checkpoint["gain"], stage1_checkpoint["bias"],
                            stage2_checkpoint["avoidance_gain"],
                            stage2_checkpoint["brake_distance"])
    full = Stage2NeuralController(brain, adapter)
    controllers = {
        "neural": (full, full.reset),
        "direct_compass": (direct_compass_controller(adapter), None),
        "sensor_only": (sensor_only_controller(adapter), None),
        "neural_no_sensors": _controlled_neural(
            brain, adapter, ranges_override=(SENSOR_RANGE,) * 5),
        "goal_cue_withheld": _controlled_neural(brain, adapter, withhold_goal=True),
        "pfl3_silenced": _controlled_neural(brain, adapter, silence_pfl3=True),
    }
    results = {}
    for name, (controller, reset) in controllers.items():
        results[name] = evaluate_breakdown(heldout, layouts, controller, reset)
        print(f"  control {name}: {results[name]['overall']}")
    return results


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_stage2_checkpoint(path: str, stage1_path: str, training_path: str,
                           heldout_path: str, gain: float, distance: float,
                           grid: list[dict]) -> None:
    payload = {
        "stage1_checkpoint": stage1_path,
        "stage1_checkpoint_sha256": sha256(stage1_path),
        "training_scenarios": training_path,
        "training_scenarios_sha256": sha256(training_path),
        "heldout_scenarios": heldout_path,
        "heldout_scenarios_sha256": sha256(heldout_path),
        "learned_parameters": ["avoidance_gain", "brake_distance"],
        "avoidance_gain": gain,
        "brake_distance": distance,
        "selection": "max arrivals, min collisions, min timeouts, min successful route length, min gain, min distance",
        "candidates": grid,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 street evaluation.")
    parser.add_argument("--freeze-scenarios", action="store_true",
                        help="Write the frozen training/held-out scenario splits.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate the Stage 2 adapter on the real graph.")
    parser.add_argument("--controls", action="store_true",
                        help="Run the six-way held-out control evaluation.")
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--stage1-checkpoint", default="runs/stage1/checkpoint.json")
    parser.add_argument("--training", default="runs/stage2/training.json")
    parser.add_argument("--heldout", default="runs/stage2/heldout.json")
    parser.add_argument("--checkpoint", default="runs/stage2/checkpoint.json")
    parser.add_argument("--out", default="runs/stage2/results.json")
    return parser


def _do_calibrate(args) -> None:
    stage1 = load_checkpoint(args.stage1_checkpoint)
    training = load_scenarios(args.training)
    brain = Brain.load(args.data, stage1["params"])
    factory = stage2_factory(brain, stage1)
    counter = {"n": 0}
    def make(gain, distance):
        counter["n"] += 1
        print(f"  candidate {counter['n']}/12: avoidance_gain={gain} brake_distance={distance}")
        return factory(gain, distance)
    gain, distance, grid = calibrate_stage2_adapter(training, make)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    save_stage2_checkpoint(args.checkpoint, args.stage1_checkpoint, args.training,
                           args.heldout, gain, distance, grid)
    print(f"Selected avoidance_gain={gain} brake_distance={distance}; wrote {args.checkpoint}")


def _do_controls(args) -> None:
    checkpoint = json.loads(Path(args.checkpoint).read_text())
    expected = {
        args.stage1_checkpoint: checkpoint["stage1_checkpoint_sha256"],
        args.training: checkpoint["training_scenarios_sha256"],
        args.heldout: checkpoint["heldout_scenarios_sha256"],
    }
    for path, recorded in expected.items():
        actual = sha256(path)
        if actual != recorded:
            raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {recorded}")
    stage1 = load_checkpoint(args.stage1_checkpoint)
    heldout = load_scenarios(args.heldout)
    results = run_stage2_controls(args.data, stage1, checkpoint, heldout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.freeze_scenarios:
        train, heldout = generate_scenario_splits(seed=args.seed)
        Path("runs/stage2").mkdir(parents=True, exist_ok=True)
        save_scenarios("runs/stage2/training.json", train)
        save_scenarios("runs/stage2/heldout.json", heldout)
        print(f"Froze {len(train)} training and {len(heldout)} held-out scenarios.")
    if args.calibrate:
        _do_calibrate(args)
    if args.controls:
        _do_controls(args)


if __name__ == "__main__":
    main()
