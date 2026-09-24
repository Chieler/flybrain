# evaluate_stage2h.py
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from evaluate_stage2 import (_scenario_key, _waypoints, load_scenarios,
                             sha256, verify_or_save_scenarios)
from evaluate_stage2g import ALL_PRIOR_SPLITS
from stage2b import WaypointController
from street import (MAX_SIM_TIME, StreetCarState, StreetScenario,
                    initial_layouts, run_street_episode)

HEADINGS = tuple(i * math.pi / 16 for i in range(-15, 17))
PRIOR_SPLITS = list(dict.fromkeys(ALL_PRIOR_SPLITS + [
    "runs/stage2g/train_split.json",
    "runs/stage2g/dev_split.json",
    "runs/stage2g/gate_split.json",
]))
TRAIN_COUNTS = {name: 150 for name in ("cross", "regular", "asymmetric")}
READINESS_COUNTS = {name: 100 for name in TRAIN_COUNTS}
GATE_COUNTS = dict(READINESS_COUNTS)


def _load(paths):
    return [scenario for path in paths for scenario in load_scenarios(path)]


def build_split(seed, counts, extra_exclude):
    rng = random.Random(seed)
    layouts = initial_layouts()
    excluded = {_scenario_key(s) for s in _load(PRIOR_SPLITS) + list(extra_exclude)}
    split = []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        candidates = [(start, target, heading)
                      for start in points for target in points
                      if start != target and math.dist(start, target) >= 12.0
                      for heading in HEADINGS
                      if (name, *start, heading, *target) not in excluded]
        rng.shuffle(candidates)
        chosen = []
        for start, target, heading in candidates:
            scenario = StreetScenario(
                name, StreetCarState(*start, heading, 0.0), *target,
                timeout=MAX_SIM_TIME,
                label=f"stage2h-{seed}-{name}-{len(chosen):03d}")
            controller = WaypointController(scenario, layout)
            if run_street_episode(scenario, layout, controller).outcome == "arrival":
                chosen.append(scenario)
                if len(chosen) == counts[name]:
                    break
        if len(chosen) != counts[name]:
            raise SystemExit(
                f"{name}: only {len(chosen)} fresh witnessed scenarios; need {counts[name]}")
        split.extend(chosen)
    return split


def build_all_splits():
    train = build_split(80, TRAIN_COUNTS, [])
    readiness = build_split(81, READINESS_COUNTS, train)
    gate = build_split(82, GATE_COUNTS, train + readiness)
    return train, readiness, gate


def freeze_splits():
    train, readiness, gate = build_all_splits()
    paths = {
        "runs/stage2h/train_split.json": train,
        "runs/stage2h/readiness_split.json": readiness,
        "runs/stage2h/gate_split.json": gate,
    }
    for path, split in paths.items():
        verify_or_save_scenarios(path, split)
    provenance = {"splits": {path: sha256(path) for path in paths},
                  "excluded": {path: sha256(path) for path in PRIOR_SPLITS}}
    Path("runs/stage2h/split_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2h PPO experiment")
    parser.add_argument("--freeze-splits", action="store_true")
    args = parser.parse_args(argv)
    if not args.freeze_splits:
        parser.error("Task 4 supports only --freeze-splits")
    freeze_splits()


if __name__ == "__main__":
    main()
