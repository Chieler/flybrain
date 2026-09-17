"""Stage 2f evaluation: aligned-reward warm-start fine-tuning, scored once on a
fresh disjoint gate with a diagonal, waypoint-witnessed cross stratum.

Provenance discipline (see the spec). The cross layout's cardinal-heading pool is
exhausted (all 64 eligible pairs consumed by Stages 2/2b/2d/2e). The ONLY source of
fresh cross is non-cardinal start headings, so this stage adds the four diagonal
headings and filters the result to WAYPOINT-WITNESSED scenarios (the privileged
witness arrives) -- a positive filter, never a solvability claim. Regular/asym are
generated unchanged with `generate_stage2b_split` (cardinal). The fitness split is
frozen FIRST (seed 60); the gate is frozen SECOND (seed 61) excluding it, so
gate INTERSECT fitness = 0 by construction and by assertion.

Interpretation (bounded, asymmetric). Only a PASS is conclusive, scoped to this
diagonal witnessed cross stratum plus the cardinal regular/asym strata. Both
failing stays confounded between reward alignment and scenario coverage; it does
not implicate the interface. No spent gate is rerun or replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from street import (
    MAX_SIM_TIME, StreetCarState, StreetScenario, initial_layouts,
    run_street_episode,
)
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, _scenario_key, _waypoints,
    load_scenarios, save_scenarios, sha256,
)
from evaluate_stage2b import (
    _breakdown, _gate, _run_per_scenario, generate_stage2b_split,
)
from evaluate_stage2d import PRIOR_SPLIT_PATHS, select_fitness_scenarios
from stage2b import WaypointController, is_outward_road_end
from stage2c import RecurrentController
from stage2d import CEMConfig, ESN_SEED, RESERVOIR_CONFIG
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
)
from stage2f import aligned_fitness

DIAGONAL_HEADINGS = (math.pi / 4, 3 * math.pi / 4, -math.pi / 4, -3 * math.pi / 4)

# Complete exclusion: the corrected prior set PLUS every spent one-shot split.
SPENT_SPLITS = [
    "runs/stage2d/gate_split.json",
    "runs/stage2d/train_split.json",
    "runs/stage2e/gate_split.json",
]
EXCLUDE_PATHS = list(PRIOR_SPLIT_PATHS) + SPENT_SPLITS

# Preregistered seeds (distinct from 2b 21/22, 2d 41/42, 2e 51). Fitness FIRST.
STAGE2F_FITNESS_SEED = 60
STAGE2F_GATE_SEED = 61
FITNESS_COUNTS = {"cross": 6, "regular": 40, "asymmetric": 40}    # 86
GATE_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}      # 100

# The 40 Stage 2d fitness scenarios -- used ONLY as the reproduction fingerprint
# (arrivals == 22), never optimized on. Excluded from both fresh splits via
# runs/stage2d/train_split.json in EXCLUDE_PATHS.
REPRO_REFERENCE_SPLIT = "runs/stage2d/train_split.json"

# Frozen warm-start CEM budget (identical for recurrent and ablation).
CEM_BUDGET = CEMConfig(population=64, n_iter=25, elite_frac=0.20,
                       init_std=0.1, seed=0)

REPRO_ARRIVALS = 22
REPRO_REWARD = {True: 1.106, False: 1.086}
REPRO_REWARD_TOL = 0.02

REF_WAYPOINT_GATE = 1.00
REF_STATE_MACHINE_GATE = 0.62
REF_STAGE2C_RECURRENT = 0.57
REF_STAGE2D_RECURRENT = 0.00
REF_STAGE2E_RECURRENT = 0.516


def _load(paths):
    out = []
    for p in paths:
        out += load_scenarios(p)
    return out


def _cardinal_strata(seed: int, exclude, counts: dict) -> list[StreetScenario]:
    """Regular + asymmetric from the unchanged cardinal generator (cross count 0)."""
    strata = dict(counts)
    strata["cross"] = 0
    return [s for s in generate_stage2b_split(seed, exclude=exclude, counts=strata)
            if s.layout != "cross"]


def build_fitness_split() -> list[StreetScenario]:
    """Fresh 86-scenario fitness split (6 diagonal cross / 40 regular / 40 asym),
    disjoint from the complete exclusion. Generated FIRST (seed 60)."""
    exclude = _load(EXCLUDE_PATHS)
    cross = generate_expanded_cross(STAGE2F_FITNESS_SEED, exclude, FITNESS_COUNTS["cross"])
    rest = _cardinal_strata(STAGE2F_FITNESS_SEED, exclude, FITNESS_COUNTS)
    split = cross + rest
    _assert_disjoint(split, exclude, "fitness")
    return split


def build_gate_split(fitness) -> list[StreetScenario]:
    """Fresh 100-scenario gate (12 diagonal cross / 44 regular / 44 asym), disjoint
    from the complete exclusion AND the frozen fitness split. Generated SECOND
    (seed 61) excluding `fitness`, so gate INTERSECT fitness = 0."""
    exclude = _load(EXCLUDE_PATHS) + list(fitness)
    cross = generate_expanded_cross(STAGE2F_GATE_SEED, exclude, GATE_COUNTS["cross"])
    rest = _cardinal_strata(STAGE2F_GATE_SEED, exclude, GATE_COUNTS)
    split = cross + rest
    _assert_disjoint(split, exclude, "gate")
    return split


def _assert_disjoint(split, exclude, label: str) -> None:
    keys = {_scenario_key(s) for s in split}
    assert len(keys) == len(split), f"duplicate identity within {label}"
    assert keys.isdisjoint({_scenario_key(s) for s in exclude}), \
        f"{label} overlaps an excluded/fitness split"


def _freeze(path: str, builder) -> list[StreetScenario]:
    if Path(path).exists():
        return load_scenarios(path)
    split = builder()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(path, split)
    return split


def generate_expanded_cross(seed: int, exclude, count: int) -> list[StreetScenario]:
    """Fresh diagonal-heading cross scenarios, waypoint-witnessed and identity-
    disjoint from `exclude`. Shuffles candidates deterministically by `seed`, then
    takes the first `count` whose waypoint witness arrives."""
    rng = random.Random(seed)
    layout = initial_layouts()["cross"]
    points = _waypoints(layout)
    pairs = [(a, b) for a in points for b in points
             if a != b and math.dist(a, b) >= 12.0]
    excluded = {_scenario_key(s) for s in (exclude or [])}
    candidates = [(a, b, h) for (a, b) in pairs for h in DIAGONAL_HEADINGS
                  if ("cross", a[0], a[1], h, b[0], b[1]) not in excluded
                  and not is_outward_road_end(a[0], a[1], h, layout)]
    rng.shuffle(candidates)
    chosen: list[StreetScenario] = []
    for (a, b, h) in candidates:
        scenario = StreetScenario(
            "cross", StreetCarState(a[0], a[1], h, 0.0), b[0], b[1],
            timeout=MAX_SIM_TIME, label=f"stage2f-cross-{len(chosen):03d}")
        wc = WaypointController(scenario, layout)
        wc.reset()
        if run_street_episode(scenario, layout, wc).outcome == "arrival":
            chosen.append(scenario)
            if len(chosen) == count:
                return chosen
    raise SystemExit(f"expanded cross: only {len(chosen)} witnessed scenarios "
                     f"(need {count}) under the given exclusion")
