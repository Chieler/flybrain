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

# --- step #3: predeclared direct-compass bracket grid + reliability gate ---
DEV_AVOIDANCE_GAINS = (1.0, 1.5, 2.0, 3.0)
DEV_COMMIT_DISTANCES = (1.5, 2.5)
DEV_RELEASE_MARGINS = (1.5, 3.0)   # release_distance = commit_distance + margin
DEV_RECOVER_SPEEDS = (1.0,)
# step #4: goal-aware commit clearance (normalized 0..1). 0.0 reproduces the
# pure-openness commit (step #2/#3 baseline) inside the same grid.
DEV_GOAL_CLEARANCES = (0.0, 0.3, 0.5, 0.7)
DEV_BRAKE_DISTANCE = 6.0           # inert lever; held fixed (see HANDOFF diagnosis)
GATE_OVERALL_RATE = 0.90           # >= 90/100 overall
GATE_PER_LAYOUT_RATE = 0.80        # >= 80% in every layout

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


def _route_key(s: StreetScenario) -> tuple:
    return (s.layout, s.start.x, s.start.y, s.target_x, s.target_y)


def _scenario_key(s: StreetScenario) -> tuple:
    return (s.layout, s.start.x, s.start.y, s.start.heading, s.target_x, s.target_y)


def generate_dev_split(seed: int = 7,
                       exclude: list[StreetScenario] | None = None) -> list[StreetScenario]:
    """A fresh 100-scenario development split (12/44/44) for the step-#3 gate.

    Every dev scenario differs from every scenario in `exclude` (the seed-2
    training + held-out sets) by at least its start heading, and fully-fresh
    routes are preferred first — a route already used by a prior set is only
    reused (with a new heading) where the layout's route pool is exhausted, which
    happens for the single-intersection `cross` layout. Never reuse the held-out
    set for tuning.
    """
    rng = random.Random(seed)
    layouts = initial_layouts()
    headings = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    counts = {"cross": 12, "regular": 44, "asymmetric": 44}
    excluded_scenarios = {_scenario_key(s) for s in (exclude or [])}
    excluded_routes = {_route_key(s) for s in (exclude or [])}
    dev = []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        pairs = [(start, target) for start in points for target in points
                 if start != target and math.dist(start, target) >= 12.0]
        candidates = [(start, target, h) for (start, target) in pairs for h in headings
                      if (name, *start, h, *target) not in excluded_scenarios]
        rng.shuffle(candidates)
        # Stable sort: route-fresh candidates (key False) keep their shuffled
        # order and come before route-reused ones.
        candidates.sort(key=lambda c: (name, *c[0], *c[1]) in excluded_routes)
        if len(candidates) < counts[name]:
            raise SystemExit(f"layout {name}: only {len(candidates)} fresh scenarios "
                             f"available, need {counts[name]}")
        dev += [StreetScenario(
            name, StreetCarState(*start, h, 0.0), *target,
            timeout=MAX_SIM_TIME, label=f"dev-{name}-{i:03d}",
        ) for i, (start, target, h) in enumerate(candidates[:counts[name]])]
    return dev


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


def _avoidance_clone(adapter: Stage2Adapter, brain_gain: float,
                     bias: float) -> Stage2Adapter:
    """Copy a source adapter's avoidance + recovery config with a new steer term."""
    return Stage2Adapter(brain_gain, bias, adapter.avoidance_gain,
                         adapter.brake_distance, adapter.cruise_speed,
                         adapter.max_steering, adapter.commit_distance,
                         adapter.release_distance, adapter.recover_speed,
                         adapter.goal_clearance)


def sensor_only_controller(adapter: Stage2Adapter):
    local = _avoidance_clone(adapter, 0.0, 0.0)
    controller = lambda obs: local(0.0, 0.0, obs.ranges, obs.speed)
    controller.reset, controller.adapter = local.reset, local
    return controller


def direct_compass_controller(adapter: Stage2Adapter):
    local = _avoidance_clone(adapter, 1.0, 0.0)
    def controller(obs):
        error = wrap_angle(obs.goal_bearing - obs.heading)
        signal = max(-MAX_STEERING, min(MAX_STEERING, 4.0 * error))
        return local(signal, 0.0, obs.ranges, obs.speed)
    controller.reset, controller.adapter = local.reset, local
    return controller


def bracket_direct_compass(dev_scenarios: list[StreetScenario],
                           avoidance_gains=DEV_AVOIDANCE_GAINS,
                           commit_distances=DEV_COMMIT_DISTANCES,
                           release_margins=DEV_RELEASE_MARGINS,
                           recover_speeds=DEV_RECOVER_SPEEDS,
                           goal_clearances=DEV_GOAL_CLEARANCES,
                           brake_distance=DEV_BRAKE_DISTANCE) -> tuple[dict, list[dict]]:
    """Bracket avoidance + commit-and-hold recovery on the cheap direct-compass
    controller (no connectome). Scores each predeclared config against the gate.
    """
    layouts = initial_layouts()
    grid, best = [], None
    for gain in avoidance_gains:
        for commit in commit_distances:
            for margin in release_margins:
                for recover in recover_speeds:
                    for clearance in goal_clearances:
                        source = Stage2Adapter(0.0, 0.0, gain, brake_distance,
                                               commit_distance=commit,
                                               release_distance=commit + margin,
                                               recover_speed=recover,
                                               goal_clearance=clearance)
                        controller = direct_compass_controller(source)
                        breakdown = evaluate_breakdown(dev_scenarios, layouts,
                                                       controller, controller.reset)
                        overall = breakdown["overall"]["arrival_rate"]
                        per_layout = {name: breakdown["by_layout"][name]["arrival_rate"]
                                      for name in layouts}
                        passes = (overall >= GATE_OVERALL_RATE and
                                  all(r >= GATE_PER_LAYOUT_RATE for r in per_layout.values()))
                        row = {"avoidance_gain": gain, "commit_distance": commit,
                               "release_distance": commit + margin,
                               "recover_speed": recover, "goal_clearance": clearance,
                               "overall_arrival_rate": overall,
                               "by_layout_arrival_rate": per_layout, "passes_gate": passes,
                               "overall": breakdown["overall"], "by_layout": breakdown["by_layout"]}
                        grid.append(row)
                        print(f"  gain={gain} commit={commit} release={commit + margin} "
                              f"recover={recover} clearance={clearance}: overall={overall:.2f} "
                              f"per={ {k: round(v, 2) for k, v in per_layout.items()} } "
                              f"{'PASS' if passes else 'fail'}", flush=True)
                        key = (passes, overall, min(per_layout.values()),
                               -gain, -commit, -clearance)
                        if best is None or key > best[0]:
                            best = (key, row)
    return best[1], grid


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
    parser.add_argument("--freeze-dev-split", action="store_true",
                        help="Write the fresh 100-scenario development split (step #3).")
    parser.add_argument("--dev-seed", type=int, default=7)
    parser.add_argument("--bracket", action="store_true",
                        help="Bracket avoidance+recovery on the cheap direct-compass "
                             "controller against the reliability gate (step #3).")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate the Stage 2 adapter on the real graph.")
    parser.add_argument("--controls", action="store_true",
                        help="Run the six-way held-out control evaluation.")
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--stage1-checkpoint", default="runs/stage1/checkpoint.json")
    parser.add_argument("--training", default="runs/stage2/training.json")
    parser.add_argument("--heldout", default="runs/stage2/heldout.json")
    parser.add_argument("--dev-split", default="runs/stage2/dev_split.json")
    parser.add_argument("--checkpoint", default="runs/stage2/checkpoint.json")
    parser.add_argument("--out", default="runs/stage2/results.json")
    parser.add_argument("--bracket-out", default="runs/stage2/dev_bracket.json")
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


def _do_bracket(args) -> None:
    dev = load_scenarios(args.dev_split)
    print(f"Bracketing direct-compass over {len(dev)} dev scenarios "
          f"(gate: overall>={GATE_OVERALL_RATE:.0%}, each layout>={GATE_PER_LAYOUT_RATE:.0%})")
    best, grid = bracket_direct_compass(dev)
    payload = {
        "dev_split": args.dev_split,
        "dev_split_sha256": sha256(args.dev_split),
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "grid_declared": {
            "avoidance_gains": list(DEV_AVOIDANCE_GAINS),
            "commit_distances": list(DEV_COMMIT_DISTANCES),
            "release_margins": list(DEV_RELEASE_MARGINS),
            "recover_speeds": list(DEV_RECOVER_SPEEDS),
            "goal_clearances": list(DEV_GOAL_CLEARANCES),
            "brake_distance": DEV_BRAKE_DISTANCE,
        },
        "any_passes_gate": any(row["passes_gate"] for row in grid),
        "best": best,
        "candidates": grid,
    }
    Path(args.bracket_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.bracket_out).write_text(json.dumps(payload, indent=2) + "\n")
    gated = "PASSES" if payload["any_passes_gate"] else "does NOT pass"
    print(f"Wrote {args.bracket_out}. Best config {gated} the gate: {best['passes_gate']} "
          f"(overall={best['overall_arrival_rate']:.2f}, "
          f"per_layout={ {k: round(v, 2) for k, v in best['by_layout_arrival_rate'].items()} })")


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.freeze_scenarios:
        train, heldout = generate_scenario_splits(seed=args.seed)
        Path("runs/stage2").mkdir(parents=True, exist_ok=True)
        save_scenarios("runs/stage2/training.json", train)
        save_scenarios("runs/stage2/heldout.json", heldout)
        print(f"Froze {len(train)} training and {len(heldout)} held-out scenarios.")
    if args.freeze_dev_split:
        exclude = []
        for path in (args.training, args.heldout):
            if Path(path).exists():
                exclude += load_scenarios(path)
        dev = generate_dev_split(seed=args.dev_seed, exclude=exclude)
        Path("runs/stage2").mkdir(parents=True, exist_ok=True)
        save_scenarios(args.dev_split, dev)
        print(f"Froze {len(dev)} development scenarios at {args.dev_split} "
              f"(disjoint from {len(exclude)} prior routes).")
    if args.bracket:
        _do_bracket(args)
    if args.calibrate:
        _do_calibrate(args)
    if args.controls:
        _do_controls(args)


if __name__ == "__main__":
    main()
