# diagnose_stage2.py
"""Per-episode diagnostic for the Stage 2 street controller.

Runs the *real, frozen* neural Stage 2 controller over the 24 frozen training
scenarios and records, at every physics step, how the steering command is
assembled:

  * neural / compass contribution  = brain_gain * (left - right) + bias
  * avoidance contribution         = avoidance_gain * (left_open - right_open)
  * the combined clipped steering, the five raw ranges, and speed

Per episode it also records outcome, elapsed time, path length, and collision
(terminal) speed, plus two evaluator-only geometry classifications:

  * line_of_sight_blocked : does the straight start->target segment cross a
    building rectangle?
  * turn_required         : LOS blocked, or large initial bearing error?

INTEGRITY: the LOS/turn classification uses scenario geometry but is computed
here in the evaluator only. The controller still receives nothing but a
`StreetObservation` (heading, goal bearing, speed, ranges). This module is
read-only w.r.t. every run artifact and the frozen checkpoints/scenarios; it
writes exactly one JSON diagnostic file.

The instrumented controller reproduces the real neural control path exactly
(`brain.encode` -> `neural_updates` steps -> `brain.outputs()` -> the same
`Stage2Adapter` call), so the recorded dynamics match the evaluated controller.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from street import (
    SENSOR_RANGE, StreetScenario, StreetLayout, initial_layouts,
    run_street_episode, _segment_rect_entry,
)
from brain import Brain, Stage2Adapter, Stage2NeuralController
from evaluate import load_checkpoint
from evaluate_stage2 import load_scenarios, sha256
from simulation import wrap_angle

# A start heading more than this far off the straight target bearing means the
# car cannot simply drive forward; combined with LOS this flags a real turn.
TURN_ERROR_THRESHOLD = math.radians(45.0)


def line_of_sight_blocked(ax: float, ay: float, bx: float, by: float,
                          layout: StreetLayout) -> bool:
    """True if the straight segment a->b enters any (un-expanded) building."""
    return any(_segment_rect_entry(ax, ay, bx, by, rect) is not None
               for rect in layout.buildings)


def classify_scenario(scenario: StreetScenario, layout: StreetLayout) -> dict:
    """Evaluator-only geometry classification. Never shown to the controller."""
    sx, sy = scenario.start.x, scenario.start.y
    tx, ty = scenario.target_x, scenario.target_y
    bearing = math.atan2(ty - sy, tx - sx)
    bearing_error = wrap_angle(bearing - scenario.start.heading)
    los_blocked = line_of_sight_blocked(sx, sy, tx, ty, layout)
    turn_required = los_blocked or abs(bearing_error) > TURN_ERROR_THRESHOLD
    return {
        "straight_distance": math.hypot(tx - sx, ty - sy),
        "initial_bearing_error": bearing_error,
        "line_of_sight_blocked": los_blocked,
        "turn_required": turn_required,
    }


def _logging_controller(brain: Brain, adapter: Stage2Adapter, log: list,
                        neural_updates: int = 2):
    """Reproduces Stage2NeuralController exactly, logging the steering split.

    The two contributions mirror `Stage2Adapter.__call__`: the neural/compass
    term and the sensor avoidance term, both before the shared clip.
    """
    def controller(obs):
        stimulus = brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(neural_updates):
            brain.step(stimulus)
        left, right = brain.outputs()
        normalized = tuple(r / SENSOR_RANGE for r in obs.ranges)
        left_open = sum(normalized[:2]) / 2
        right_open = sum(normalized[-2:]) / 2
        neural_term = adapter.brain_gain * (left - right) + adapter.bias
        avoid_term = adapter.avoidance_gain * (left_open - right_open)
        control = adapter(left, right, obs.ranges, obs.speed)
        log.append({
            "left": left, "right": right,
            "neural_term": neural_term, "avoid_term": avoid_term,
            "steering": control.steering, "acceleration": control.acceleration,
            "ranges": list(obs.ranges), "speed": obs.speed,
            "goal_bearing": obs.goal_bearing, "heading": obs.heading,
        })
        return control
    return controller


def _sign(x: float, eps: float = 1e-9) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def _aggregate(steps: list) -> dict:
    """Episode-level summary of the per-step steering split."""
    n = len(steps)
    if n == 0:
        return {"n_steps": 0}
    mean = lambda vals: sum(vals) / n
    forward = [s["ranges"][2] for s in steps]
    opposed = sum(_sign(s["neural_term"]) * _sign(s["avoid_term"]) < 0
                  for s in steps)
    avoid_dominant = sum(abs(s["avoid_term"]) > abs(s["neural_term"])
                         for s in steps)
    return {
        "n_steps": n,
        "mean_neural_term": mean([s["neural_term"] for s in steps]),
        "mean_abs_neural_term": mean([abs(s["neural_term"]) for s in steps]),
        "mean_avoid_term": mean([s["avoid_term"] for s in steps]),
        "mean_abs_avoid_term": mean([abs(s["avoid_term"]) for s in steps]),
        "final_neural_term": steps[-1]["neural_term"],
        "final_avoid_term": steps[-1]["avoid_term"],
        "fraction_terms_opposed": opposed / n,
        "fraction_avoid_dominant": avoid_dominant / n,
        "min_forward_range": min(forward),
        "final_forward_range": forward[-1],
        "mean_speed": mean([s["speed"] for s in steps]),
        "final_speed": steps[-1]["speed"],
    }


def run_episode_diagnostic(scenario: StreetScenario, layout: StreetLayout,
                           brain: Brain, adapter: Stage2Adapter,
                           downsample: int = 5) -> dict:
    brain.reset()
    steps: list = []
    controller = _logging_controller(brain, adapter, steps)
    result = run_street_episode(scenario, layout, controller, record=True)
    # Terminal speed at the collision/arrival point comes from the last
    # recorded trajectory tuple (x, y, heading, speed).
    terminal_speed = result.trajectory[-1][3] if result.trajectory else None
    return {
        "label": scenario.label,
        "layout": scenario.layout,
        "outcome": result.outcome,
        "elapsed_time": result.elapsed_time,
        "path_length": result.path_length,
        "collision_speed": terminal_speed if result.outcome == "collision" else None,
        "terminal_speed": terminal_speed,
        "classification": classify_scenario(scenario, layout),
        "aggregate": _aggregate(steps),
        "steps": steps[::downsample],
    }


def run_diagnostic(data_dir: str, stage1_checkpoint: dict,
                   stage2_checkpoint: dict,
                   scenarios: list[StreetScenario],
                   downsample: int = 5) -> dict:
    layouts = initial_layouts()
    brain = Brain.load(data_dir, stage1_checkpoint["params"])
    adapter = Stage2Adapter(stage1_checkpoint["gain"], stage1_checkpoint["bias"],
                            stage2_checkpoint["avoidance_gain"],
                            stage2_checkpoint["brake_distance"])
    episodes = []
    for i, scenario in enumerate(scenarios):
        episode = run_episode_diagnostic(scenario, layouts[scenario.layout],
                                         brain, adapter, downsample)
        episodes.append(episode)
        cls, agg = episode["classification"], episode["aggregate"]
        print(f"  [{i + 1:2d}/{len(scenarios)}] {scenario.label:24s} "
              f"{episode['outcome']:9s} los={int(cls['line_of_sight_blocked'])} "
              f"turn={int(cls['turn_required'])} "
              f"opp={agg.get('fraction_terms_opposed', 0):.2f} "
              f"min_fwd={agg.get('min_forward_range', 0):.2f}", flush=True)
    return {
        "downsample": downsample,
        "avoidance_gain": stage2_checkpoint["avoidance_gain"],
        "brake_distance": stage2_checkpoint["brake_distance"],
        "brain_gain": stage1_checkpoint["gain"],
        "bias": stage1_checkpoint["bias"],
        "summary": _summarize_episodes(episodes),
        "episodes": episodes,
    }


def _summarize_episodes(episodes: list) -> dict:
    def rate(subset):
        arr = sum(e["outcome"] == "arrival" for e in subset)
        return {"n": len(subset), "arrivals": arr,
                "collisions": sum(e["outcome"] == "collision" for e in subset),
                "timeouts": sum(e["outcome"] == "timeout" for e in subset)}
    turn = [e for e in episodes if e["classification"]["turn_required"]]
    los = [e for e in episodes if e["classification"]["line_of_sight_blocked"]]
    straight = [e for e in episodes if not e["classification"]["turn_required"]]
    coll_speeds = [e["collision_speed"] for e in episodes
                   if e["collision_speed"] is not None]
    return {
        "overall": rate(episodes),
        "turn_required": rate(turn),
        "line_of_sight_blocked": rate(los),
        "straight_shot": rate(straight),
        "mean_collision_speed": (sum(coll_speeds) / len(coll_speeds)
                                 if coll_speeds else None),
        "n_collisions_below_speed_0.25": sum(s < 0.25 for s in coll_speeds),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 per-episode steering diagnostic (training set).")
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--checkpoint", default="runs/stage2/checkpoint.json")
    parser.add_argument("--training", default="runs/stage2/training.json")
    parser.add_argument("--downsample", type=int, default=5,
                        help="Keep every Nth per-step record in the output.")
    parser.add_argument("--out", default="runs/stage2/training_diagnostic.json")
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    checkpoint = json.loads(Path(args.checkpoint).read_text())
    stage1_path = checkpoint["stage1_checkpoint"]
    # Diagnose exactly the frozen artifacts the evaluation used.
    expected = {
        stage1_path: checkpoint["stage1_checkpoint_sha256"],
        args.training: checkpoint["training_scenarios_sha256"],
    }
    for path, recorded in expected.items():
        actual = sha256(path)
        if actual != recorded:
            raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {recorded}")
    stage1 = load_checkpoint(stage1_path)
    scenarios = load_scenarios(args.training)
    print(f"Diagnosing {len(scenarios)} training scenarios "
          f"(avoidance_gain={checkpoint['avoidance_gain']}, "
          f"brake_distance={checkpoint['brake_distance']})", flush=True)
    diagnostic = run_diagnostic(args.data, stage1, checkpoint, scenarios,
                                args.downsample)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(f"Summary: {json.dumps(diagnostic['summary'], indent=2)}")


if __name__ == "__main__":
    main()
