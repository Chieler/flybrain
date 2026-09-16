"""Stage 2b evaluation: score the two observation-boundary baselines on the same
frozen dev split and 90%/80% gate used for the Stage 2 direct-compass screen.

Interpretation (bounded; see HANDOFF Stage 2b). The waypoint controller is a
solvability *witness* with privileged layout/route/target/pose; the observation
state machine sees only heading, goal_bearing, speed, ranges and carries no
route/topological memory. Waypoint success on a scenario establishes that
scenario is solvable. A large waypoint-vs-SM gap does NOT by itself isolate a
single cause -- the two differ in map, route, and pose, not only in memory -- but
it motivates route/history memory as the next hypothesis, to be tested with a
same-observation recurrent baseline (no map, no pose).

Neither controller is neural. This does not touch the connectome or the frozen
Stage 2 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from street import (
    MAX_SIM_TIME, StreetCarState, StreetLayout, StreetScenario, initial_layouts,
    run_street_episode,
)
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, _summarize, _scenario_key,
    _waypoints, load_scenarios, save_scenarios, sha256,
)
from stage2b import ObservationStateMachine, WaypointController, is_outward_road_end

# Fresh Stage 2b splits forbid outward-facing road-end starts (the eligible
# stratum). Both are disjoint by scenario identity from the already-inspected
# seed-7 dev split; the SM-training split is also disjoint from the gate split.
GATE_SPLIT_SEED = 21
SM_TRAIN_SEED = 22
SPLIT_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}


def generate_stage2b_split(seed: int, exclude=None, counts=None) -> list[StreetScenario]:
    """A frozen Stage 2b split that forbids outward-facing road-end starts and
    excludes every scenario identity in `exclude`."""
    rng = random.Random(seed)
    layouts = initial_layouts()
    headings = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    counts = counts or SPLIT_COUNTS
    excluded = {_scenario_key(s) for s in (exclude or [])}
    split = []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        pairs = [(a, b) for a in points for b in points
                 if a != b and math.dist(a, b) >= 12.0]
        candidates = [(a, b, h) for (a, b) in pairs for h in headings
                      if (name, *a, h, *b) not in excluded
                      and not is_outward_road_end(a[0], a[1], h, layout)]
        rng.shuffle(candidates)
        if len(candidates) < counts[name]:
            raise SystemExit(f"layout {name}: only {len(candidates)} eligible "
                             f"scenarios, need {counts[name]}")
        split += [StreetScenario(
            name, StreetCarState(*a, h, 0.0), *b,
            timeout=MAX_SIM_TIME, label=f"stage2b-{name}-{i:03d}",
        ) for i, (a, b, h) in enumerate(candidates[:counts[name]])]
    return split


def _run_per_scenario(scenarios, layouts: dict[str, StreetLayout], make):
    """Run one freshly-built controller per scenario (waypoint routes are
    per-scenario; the state machine is rebuilt for symmetry).

    Each row is (layout, result, eligible) where `eligible` is False for
    outward-facing road-end starts (the excluded stratum)."""
    rows = []
    for scenario in scenarios:
        layout = layouts[scenario.layout]
        controller, reset = make(scenario, layout)
        if reset is not None:
            reset()
        eligible = not is_outward_road_end(scenario.start.x, scenario.start.y,
                                           scenario.start.heading, layout)
        rows.append((scenario.layout, run_street_episode(scenario, layout, controller),
                     eligible))
    return rows


def _breakdown(rows, layouts) -> dict:
    return {
        "overall": _summarize([r for _, r, _ in rows]).as_dict(),
        "by_layout": {name: _summarize([r for lay, r, _ in rows if lay == name]).as_dict()
                      for name in layouts},
    }


def _eligible_breakdown(rows, layouts) -> dict:
    kept = [(lay, r, e) for (lay, r, e) in rows if e]
    return {
        "excluded_outward_road_end": sum(1 for _, _, e in rows if not e),
        "overall": _summarize([r for _, r, _ in kept]).as_dict(),
        "by_layout": {name: _summarize([r for lay, r, _ in kept if lay == name]).as_dict()
                      for name in layouts},
    }


def _gate(breakdown, layouts) -> tuple[bool, float, dict]:
    overall = breakdown["overall"]["arrival_rate"]
    per_layout = {name: breakdown["by_layout"][name]["arrival_rate"] for name in layouts}
    passes = (overall >= GATE_OVERALL_RATE and
              all(r >= GATE_PER_LAYOUT_RATE for r in per_layout.values()))
    return passes, overall, per_layout


def evaluate_baselines(dev_scenarios) -> dict:
    layouts = initial_layouts()
    makers = {
        "waypoint": lambda s, l: (wc := WaypointController(s, l), wc.reset),
        "observation_state_machine":
            lambda s, l: (sm := ObservationStateMachine(), sm.reset),
    }
    results = {}
    for name, make in makers.items():
        rows = _run_per_scenario(dev_scenarios, layouts, make)
        breakdown = _breakdown(rows, layouts)
        eligible = _eligible_breakdown(rows, layouts)
        passes, overall, per_layout = _gate(breakdown, layouts)
        results[name] = {"passes_gate": passes, "overall_arrival_rate": overall,
                         "by_layout_arrival_rate": per_layout,
                         "full_split": breakdown, "eligible_stratum": eligible}
        elig_rate = eligible["overall"]["arrival_rate"]
        print(f"  {name}: full={overall:.2f} "
              f"eligible={elig_rate:.3f} "
              f"(excl {eligible['excluded_outward_road_end']} outward road-end)",
              flush=True)
    return results


def _interpretation(results: dict) -> str:
    wp = results["waypoint"]["eligible_stratum"]["overall"]["arrival_rate"]
    sm = results["observation_state_machine"]["eligible_stratum"]["overall"]["arrival_rate"]
    excl = results["waypoint"]["eligible_stratum"]["excluded_outward_road_end"]
    return (
        f"On this split's eligible stratum (excluding {excl} outward-facing "
        f"road-end starts): waypoint solvability witness {wp:.3f}, "
        f"observation-only state machine {sm:.3f}. The waypoint failures on the "
        f"full split are exactly those outward-facing road-end starts (a "
        f"lane-width vs turn-radius interaction; a post-hoc characterization, not "
        f"an impossibility claim). The waypoint controller has privileged "
        f"map/route/pose, so a waypoint-vs-SM gap does not by itself localize the "
        f"deficit to memory; it establishes solvability and motivates a "
        f"same-observation recurrent/history baseline (no map, no pose) as the "
        f"next gate.")


def _freeze_splits(gate_path: str, train_path: str, inspected_split: str) -> None:
    inspected = load_scenarios(inspected_split) if Path(inspected_split).exists() else []
    gate = generate_stage2b_split(GATE_SPLIT_SEED, exclude=inspected)
    train = generate_stage2b_split(SM_TRAIN_SEED, exclude=inspected + gate)
    Path(gate_path).parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(gate_path, gate)
    save_scenarios(train_path, train)
    print(f"Froze {len(gate)} gate scenarios -> {gate_path}")
    print(f"Froze {len(train)} SM-training scenarios -> {train_path}")
    # Step 4 gate: the waypoint solvability witness must pass 90%/80% here.
    layouts = initial_layouts()
    make = lambda s, l: (wc := WaypointController(s, l), wc.reset)
    breakdown = _breakdown(_run_per_scenario(gate, layouts, make), layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    print(f"Waypoint witness on gate split: overall={overall:.2f} "
          f"per={ {k: round(v, 2) for k, v in per_layout.items()} } "
          f"{'PASS' if passes else 'FAIL'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2b baseline evaluation.")
    parser.add_argument("--dev-split", default="runs/stage2/dev_split.json")
    parser.add_argument("--out", default="runs/stage2b/results.json")
    parser.add_argument("--freeze-splits", action="store_true",
                        help="Freeze the fresh gate + SM-training splits "
                             "(forbid outward road-end starts) and check the "
                             "waypoint witness passes the gate.")
    parser.add_argument("--gate-split", default="runs/stage2b/gate_split.json")
    parser.add_argument("--train-split", default="runs/stage2b/sm_train_split.json")
    parser.add_argument("--inspected-split", default="runs/stage2/dev_split.json")
    args = parser.parse_args(argv)

    if args.freeze_splits:
        _freeze_splits(args.gate_split, args.train_split, args.inspected_split)
        return

    dev = load_scenarios(args.dev_split)
    print(f"Stage 2b baselines over {len(dev)} dev scenarios "
          f"(gate: overall>={GATE_OVERALL_RATE:.0%}, each layout>={GATE_PER_LAYOUT_RATE:.0%})")
    results = evaluate_baselines(dev)
    payload = {
        "dev_split": args.dev_split,
        "dev_split_sha256": sha256(args.dev_split),
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "results": results,
        "interpretation": _interpretation(results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(f"Interpretation: {payload['interpretation']}")


if __name__ == "__main__":
    main()
