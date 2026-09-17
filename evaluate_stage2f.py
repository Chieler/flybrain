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
