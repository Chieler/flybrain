# Stage 2f — aligned-reward warm-start fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the frozen-reservoir readout by warm-start CEM against an *arrival-count-primary* fitness, and score it once on a fresh disjoint gate whose cross stratum is expanded with diagonal, waypoint-witnessed scenarios.

**Architecture:** Adds `stage2f.py` (arrival-primary fitness primitives) and `evaluate_stage2f.py` (expanded-cross generator, fresh split builders, one-shot run), extending the existing `stage2e` trainer/evaluator with a pluggable `fitness_fn`. The reservoir, warm start, and CEM machinery are reused unchanged from Stage 2d/2e; only the fitness signal and the cross stratum change.

**Tech Stack:** Python 3, NumPy, pure-NumPy CEM (`stage2d.cem_maximize`), the `street` simulator, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-17-stage2f-aligned-reward-design.md` — read it alongside this plan; the plan argues the spec.

## Global Constraints

- **Fitness (frozen, arrival-primary):** `fitness(θ) = arrivals(θ) + mean_over_episodes(secondary) / 4.0`, where per episode `secondary = 0.5·progress − 2.0·[collision] − 0.1·(elapsed_time/MAX_SIM_TIME)` and `progress = clip((d0 − d_final)/d0, −1, 1)` using the FINAL position. Shaping coefficients `W_PROGRESS=0.5, W_COLLIDE=2.0, W_TIME=0.1`, `SECONDARY_DIVISOR=4.0` — frozen, never tuned on the 2f gate or fitness set.
- **Reservoir (frozen 2c winner):** n=64, spectral_radius=0.8, leak=0.5, input_scale=1.0, seed=0; 150 readout params. Matched memoryless ablation. Reuse `stage2d.RESERVOIR_CONFIG` / `ESN_SEED`.
- **CEM budget (both models, identical, unconditional):** `CEMConfig(population=64, n_iter=25, elite_frac=0.20, init_std=0.1, std_floor=0.001, seed=0)`. `n_iter=25` is fixed — no timing-driven fallback; the timing smoke reports only.
- **Warm start:** deterministic Stage 2c readout via `stage2e.warm_start_readout`; reproduction gate `arrivals == 22` on the fixed 40 Stage 2d fitness scenarios asserted before optimizing (reward value ~1.106 recurrent / ~1.086 ablation, `delta=0.02`).
- **Preregistered seeds / order:** `STAGE2F_FITNESS_SEED = 60`, `STAGE2F_GATE_SEED = 61`. Fitness split generated FIRST; gate generated SECOND excluding the frozen fitness split.
- **Fresh counts:** fitness split `6 cross / 40 regular / 40 asymmetric = 86`; gate split `12 cross / 44 regular / 44 asymmetric = 100`. Cross gate (0.80) needs 10/12.
- **Cross expansion:** diagonal headings `±π/4, ±3π/4`, `dist ≥ 12`, outward-road-end filtered, **waypoint-witnessed** (positive filter, not a solvability claim). Regular/asym unchanged (`generate_stage2b_split`, cardinal).
- **Exclusion (exact scenario identity):** the corrected `PRIOR_SPLIT_PATHS` plus spent `runs/stage2d/{gate_split,train_split}.json` and `runs/stage2e/gate_split.json`. Gate ∩ fitness = ∅ (asserted). Route reuse allowed; only identities must differ.
- **Verdict asymmetry:** only a PASS is conclusive, scoped to the diagonal waypoint-witnessed cross stratum plus the cardinal regular/asym strata. A failure stays bounded/confounded and never implicates the observation interface.
- **Do not** touch the connectome or any frozen Stage 2/2b/2c/2d/2e artifact; never rerun or replace a spent gate.

---

### Task 1: `stage2f.py` — arrival-primary fitness primitives

**Files:**
- Create: `stage2f.py`
- Test: `test_stage2f.py`

**Interfaces:**
- Consumes: `stage2d._final_xy`, `stage2d.RESERVOIR_CONFIG`; `street.MAX_SIM_TIME`, `street.run_street_episode`; `stage2c.RecurrentController`; `street.StreetEpisodeResult`.
- Produces:
  - `stage2f.W_PROGRESS = 0.5`, `stage2f.W_COLLIDE = 2.0`, `stage2f.W_TIME = 0.1`, `stage2f.SECONDARY_DIVISOR = 4.0`
  - `stage2f.episode_secondary(scenario, result) -> float`
  - `stage2f.aligned_fitness(esn, scenarios, layouts) -> float`

- [ ] **Step 1: Write failing tests for the aligned reward ordering and bounds**

Add to a new `test_stage2f.py`:

```python
"""Stage 2f tests: arrival-primary fitness + diagonal witnessed cross + fresh gate."""
from __future__ import annotations

import math
import unittest

import numpy as np

from street import (
    MAX_SIM_TIME, StreetCarState, StreetEpisodeResult, StreetScenario,
    initial_layouts,
)
import stage2f


def _scenario(start_xy, target_xy, heading=0.0, layout="cross"):
    return StreetScenario(
        layout, StreetCarState(start_xy[0], start_xy[1], heading, 0.0),
        target_xy[0], target_xy[1], timeout=MAX_SIM_TIME, label="t")


def _result(outcome, final_xy, elapsed=MAX_SIM_TIME):
    # A single-point trajectory whose last point is `final_xy` (x, y, heading, speed).
    traj = [(final_xy[0], final_xy[1], 0.0, 0.0)]
    return StreetEpisodeResult(outcome, elapsed, 0.0, traj)


class TestEpisodeSecondary(unittest.TestCase):
    def test_strict_outcome_ordering_at_equal_arrival_count(self):
        # start at origin, target 10 units east; d0 = 10.
        s = _scenario((0.0, 0.0), (10.0, 0.0))
        collide = stage2f.episode_secondary(s, _result("collision", (0.0, 0.0)))
        standing = stage2f.episode_secondary(s, _result("timeout", (0.0, 0.0)))
        progress = stage2f.episode_secondary(s, _result("timeout", (5.0, 0.0)))
        self.assertLess(collide, standing)      # collision < standing still
        self.assertLess(standing, progress)     # standing still < forward progress
        self.assertAlmostEqual(collide, 0.5 * 0.0 - 2.0 - 0.1)   # = -2.1
        self.assertAlmostEqual(standing, -0.1)                    # progress 0, time 1
        self.assertAlmostEqual(progress, 0.5 * 0.5 - 0.1)         # = 0.15

    def test_secondary_is_bounded(self):
        s = _scenario((0.0, 0.0), (10.0, 0.0))
        # Worst case: fully receded + collision + full time.
        worst = stage2f.episode_secondary(s, _result("collision", (-20.0, 0.0)))
        # Best case: full progress, no collision, no time.
        best = stage2f.episode_secondary(s, _result("arrival", (10.0, 0.0), elapsed=0.0))
        self.assertGreaterEqual(worst, -2.6 - 1e-9)
        self.assertLessEqual(best, 0.5 + 1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest test_stage2f.py -k "EpisodeSecondary" -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stage2f'` / `AttributeError: episode_secondary`.

- [ ] **Step 3: Implement `stage2f.py` primitives**

```python
"""Stage 2f: aligned-reward warm-start fine-tuning (route-memory gate, take 4).

Purpose (see docs/superpowers/specs/2026-09-17-stage2f-aligned-reward-design.md).
Stage 2e escaped 2d's optimization floor (recurrent gate 0.00 -> 0.516, zero
timeouts) and beat its ablation, but neither cleared the 0.90/0.80 gate. The
diagnosis: fitness (mean episode reward) did not track the gate metric -- the
optimizer could raise reward while arrivals fell. Stage 2f makes ARRIVAL COUNT the
primary CEM objective and demotes the bounded-net-progress shaping to a strictly
bounded (< 1) tie-breaker that can never outweigh a single arrival.

Only a PASS is conclusive, scoped to the diagonal waypoint-witnessed cross stratum
plus the cardinal regular/asym strata. A failure stays bounded and confounded
between reward alignment and scenario coverage; it never implicates the observation
interface. Does not touch the connectome or the frozen Stage 2/2b/2c/2d/2e
artifacts; no spent gate is rerun or replaced.
"""

from __future__ import annotations

import math

from street import MAX_SIM_TIME, run_street_episode
from stage2c import RecurrentController
from stage2d import _final_xy

# Frozen shaping coefficients (design choices; not tuned on the 2f gate). Arrival
# count is the integer primary term; these order the sub-1 tie-break only.
W_PROGRESS = 0.5
W_COLLIDE = 2.0
W_TIME = 0.1
SECONDARY_DIVISOR = 4.0


def episode_secondary(scenario, result) -> float:
    """Bounded per-episode shaping in [-2.6, 0.5]. Net progress uses the FINAL
    position (unfarmable). Privileged distances are used here only -- never as
    controller inputs."""
    d0 = math.hypot(scenario.target_x - scenario.start.x,
                    scenario.target_y - scenario.start.y)
    fx, fy = _final_xy(scenario, result)
    d_final = math.hypot(scenario.target_x - fx, scenario.target_y - fy)
    progress = (d0 - d_final) / d0 if d0 > 0.0 else 0.0
    progress = max(-1.0, min(1.0, progress))
    s = W_PROGRESS * progress
    s -= W_COLLIDE if result.outcome == "collision" else 0.0
    s -= W_TIME * (result.elapsed_time / MAX_SIM_TIME)
    return s


def aligned_fitness(esn, scenarios, layouts) -> float:
    """Arrival-count-primary fitness for the policy currently on `esn`:
    `arrivals + mean(episode_secondary) / SECONDARY_DIVISOR`. The mean-then-divide
    contribution lies in [-0.65, 0.125] (swing < 1), so one more arrival always
    wins."""
    controller = RecurrentController(esn)
    arrivals = 0
    secondary_total = 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        if result.outcome == "arrival":
            arrivals += 1
        secondary_total += episode_secondary(scenario, result)
    return arrivals + (secondary_total / len(scenarios)) / SECONDARY_DIVISOR
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest test_stage2f.py -k "EpisodeSecondary" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add stage2f.py test_stage2f.py
git commit -m "feat(stage2f): arrival-primary aligned fitness primitives"
```

---

### Task 2: extend the Stage 2e trainer/evaluator with a pluggable `fitness_fn`

**Files:**
- Modify: `stage2e.py` (`evaluate_policy`, `train_readout_by_reward_warmstart`; add `_mean_episode_reward`)
- Test: `test_stage2f.py`

**Interfaces:**
- Consumes: existing `stage2e.evaluate_policy`, `stage2e.train_readout_by_reward_warmstart`, `stage2e.warm_start_readout`; `stage2f.aligned_fitness`; `stage2d.CEMConfig`.
- Produces:
  - `stage2e._mean_episode_reward(esn, scenarios, layouts) -> float` (the default 2e fitness)
  - `stage2e.evaluate_policy(esn, scenarios, layouts, fitness_fn=None) -> dict` — adds `"fitness"` when `fitness_fn` given; default behavior unchanged
  - `stage2e.train_readout_by_reward_warmstart(esn, theta0, fitness_scenarios, layouts, cfg, fitness_fn=None) -> (esn, info)` — default uses `_mean_episode_reward` (2e semantics bit-for-bit); `info` gains `"warmstart_outcomes"`

- [ ] **Step 1: Write failing tests for the pluggable fitness_fn**

Add to `test_stage2f.py`:

```python
from street import initial_layouts
from evaluate_stage2 import load_scenarios
from evaluate_stage2d import select_fitness_scenarios
from stage2d import CEMConfig
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
)


class TestPluggableFitness(unittest.TestCase):
    def test_default_matches_2e_mean_reward(self):
        layouts = initial_layouts()
        fitness = select_fitness_scenarios(
            load_scenarios("runs/stage2d/train_split.json"))[:6]
        esn, _ = warm_start_readout(True, layouts)
        out = evaluate_policy(esn, fitness, layouts)          # no fitness_fn
        self.assertNotIn("fitness", out)                     # 2e shape preserved
        out2 = evaluate_policy(esn, fitness, layouts, fitness_fn=stage2f.aligned_fitness)
        self.assertIn("fitness", out2)
        # counts are reward-independent -> identical across the two calls
        self.assertEqual(out["arrivals"], out2["arrivals"])

    def test_trainer_uses_supplied_fitness_and_records_warmstart_outcomes(self):
        layouts = initial_layouts()
        fitness = select_fitness_scenarios(
            load_scenarios("runs/stage2d/train_split.json"))[:6]
        esn, theta0 = warm_start_readout(True, layouts)
        esn, info = train_readout_by_reward_warmstart(
            esn, theta0, fitness, layouts, CEMConfig(population=6, n_iter=2, seed=0),
            fitness_fn=stage2f.aligned_fitness)
        # warm-start guard uses the supplied (aligned) fitness
        self.assertGreaterEqual(info["best_fitness"], info["warmstart_fitness"] - 1e-9)
        self.assertIn("warmstart_outcomes", info)
        self.assertEqual(info["warmstart_outcomes"]["n"], len(fitness))
        # aligned fitness >= arrivals - 1 (secondary contribution is > -1)
        self.assertGreater(info["warmstart_fitness"],
                           info["warmstart_outcomes"]["arrivals"] - 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest test_stage2f.py -k "PluggableFitness" -q`
Expected: FAIL (`evaluate_policy() got an unexpected keyword argument 'fitness_fn'`).

- [ ] **Step 3: Add `_mean_episode_reward` and thread `fitness_fn` through**

In `stage2e.py`, add after the imports/constants block:

```python
def _mean_episode_reward(esn, scenarios, layouts) -> float:
    """The Stage 2e fitness: mean bounded-net-progress episode reward. Default so
    2e semantics are unchanged when no `fitness_fn` is supplied."""
    controller = RecurrentController(esn)
    total = 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        total += episode_reward(scenario, result)
    return total / len(scenarios)
```

Change `evaluate_policy` to accept `fitness_fn=None` and report it:

```python
def evaluate_policy(esn, scenarios, layouts, fitness_fn=None) -> dict:
    """Mean reward and arrival/collision/timeout counts for the policy currently
    on `esn`. When `fitness_fn` is given, also report `"fitness"` (the training
    objective's value); otherwise the 2e-shaped dict is returned unchanged."""
    controller = RecurrentController(esn)
    total = 0.0
    counts = {"arrival": 0, "collision": 0, "timeout": 0}
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        total += episode_reward(scenario, result)
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    out = {"mean_reward": total / len(scenarios), "n": len(scenarios),
           "arrivals": counts["arrival"], "collisions": counts["collision"],
           "timeouts": counts["timeout"]}
    if fitness_fn is not None:
        out["fitness"] = float(fitness_fn(esn, scenarios, layouts))
    return out
```

Change `train_readout_by_reward_warmstart` to accept and use `fitness_fn`:

```python
def train_readout_by_reward_warmstart(esn, theta0, fitness_scenarios, layouts,
                                      cfg: CEMConfig, fitness_fn=None):
    """Fine-tune `esn.W_out` by CEM from `theta0` against `fitness_fn` (default the
    Stage 2e mean episode reward). Uses the BEST-EVER candidate with a warm-start
    guard (never ships worse than theta0). Records the warm-start outcomes on the
    fitness set and the trained policy's outcomes. Returns `(esn, info)`."""
    fitness_fn = fitness_fn or _mean_episode_reward
    theta0 = np.asarray(theta0, dtype=float)
    d = theta0.size // 2

    def fitness(theta: np.ndarray) -> float:
        esn.W_out = theta.reshape(2, d)
        return fitness_fn(esn, fitness_scenarios, layouts)

    f0 = fitness(theta0)   # warm-start fitness -- never ship worse than this
    warmstart_outcomes = evaluate_policy(esn, fitness_scenarios, layouts, fitness_fn)
    _, info = cem_maximize(fitness, theta0.size, cfg, init_mu=theta0)
    if info["best"]["fitness"] >= f0:
        best_theta = np.asarray(info["best"]["theta"], dtype=float)
        info["improved_over_warmstart"] = True
    else:
        best_theta = theta0   # no perturbation beat the warm start; keep it
        info["improved_over_warmstart"] = False
    info["warmstart_fitness"] = float(f0)
    info["warmstart_outcomes"] = warmstart_outcomes
    info["best_fitness"] = float(max(info["best"]["fitness"], f0))
    esn.W_out = best_theta.reshape(2, d)
    info["best_theta"] = best_theta.tolist()
    info["best_outcomes"] = evaluate_policy(esn, fitness_scenarios, layouts, fitness_fn)
    return esn, info
```

Note: `evaluate_policy` currently computes `mean_reward` (2e episode_reward) plus counts; `best_outcomes`/`warmstart_outcomes` now also carry `"fitness"` (the aligned value when 2f passes `aligned_fitness`).

- [ ] **Step 4: Run the new tests AND the 2e regression suite**

Run: `python -m pytest test_stage2f.py -k "PluggableFitness" test_stage2e.py -q`
Expected: PASS (new tests pass; all existing `test_stage2e.py` tests stay green — the default path is byte-for-byte the old behavior plus one extra reward-independent `warmstart_outcomes` field, which 2e ignores).

- [ ] **Step 5: Commit**

```bash
git add stage2e.py test_stage2f.py
git commit -m "feat(stage2e): pluggable fitness_fn; persist warm-start outcomes"
```

---

### Task 3: `evaluate_stage2f.generate_expanded_cross` — diagonal, waypoint-witnessed cross

**Files:**
- Create: `evaluate_stage2f.py` (module header + `generate_expanded_cross` + constants)
- Test: `test_stage2f.py`

**Interfaces:**
- Consumes: `evaluate_stage2._waypoints`, `evaluate_stage2._scenario_key`; `evaluate_stage2b.is_outward_road_end`? (imported from `stage2b`); `stage2b.WaypointController`, `stage2b.is_outward_road_end`; `street.{StreetScenario, StreetCarState, MAX_SIM_TIME, initial_layouts, run_street_episode}`.
- Produces:
  - `evaluate_stage2f.DIAGONAL_HEADINGS = (math.pi/4, 3*math.pi/4, -math.pi/4, -3*math.pi/4)`
  - `evaluate_stage2f.generate_expanded_cross(seed, exclude, count) -> list[StreetScenario]` — all `cross`, diagonal heading, `dist ≥ 12`, outward-filtered, waypoint-witnessed, identity-disjoint from `exclude`; raises `SystemExit` if fewer than `count` witnessed candidates exist.

- [ ] **Step 1: Write failing tests for the expanded-cross generator**

Add to `test_stage2f.py`:

```python
from evaluate_stage2 import _scenario_key
from stage2b import WaypointController, is_outward_road_end
from street import run_street_episode


class TestExpandedCross(unittest.TestCase):
    def test_all_cross_diagonal_witnessed_and_disjoint(self):
        from evaluate_stage2f import DIAGONAL_HEADINGS, generate_expanded_cross
        layouts = initial_layouts()
        cross = generate_expanded_cross(seed=61, exclude=[], count=12)
        self.assertEqual(len(cross), 12)
        keys = {_scenario_key(s) for s in cross}
        self.assertEqual(len(keys), 12)                         # no internal dups
        for s in cross:
            self.assertEqual(s.layout, "cross")
            self.assertIn(round(s.start.heading, 6),
                          {round(h, 6) for h in DIAGONAL_HEADINGS})
            layout = layouts["cross"]
            self.assertFalse(is_outward_road_end(
                s.start.x, s.start.y, s.start.heading, layout))
            wc = WaypointController(s, layout); wc.reset()
            self.assertEqual(run_street_episode(s, layout, wc).outcome, "arrival")

    def test_exclude_is_respected(self):
        from evaluate_stage2f import generate_expanded_cross
        first = generate_expanded_cross(seed=61, exclude=[], count=6)
        more = generate_expanded_cross(seed=61, exclude=first, count=6)
        self.assertTrue({_scenario_key(s) for s in first}.isdisjoint(
            {_scenario_key(s) for s in more}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest test_stage2f.py -k "ExpandedCross" -q`
Expected: FAIL (`No module named 'evaluate_stage2f'`).

- [ ] **Step 3: Create `evaluate_stage2f.py` with the module header and generator**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest test_stage2f.py -k "ExpandedCross" -q`
Expected: PASS (2 tests). (The witness runs make this take a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add evaluate_stage2f.py test_stage2f.py
git commit -m "feat(stage2f): diagonal waypoint-witnessed expanded-cross generator"
```

---

### Task 4: `evaluate_stage2f` fresh-split builders + preregistration

**Files:**
- Modify: `evaluate_stage2f.py` (constants + `_load`, `build_fitness_split`, `build_gate_split`, `_freeze`)
- Test: `test_stage2f.py`

**Interfaces:**
- Consumes: `generate_expanded_cross`, `generate_stage2b_split`, `PRIOR_SPLIT_PATHS`, `_scenario_key`, `load_scenarios`, `save_scenarios`.
- Produces:
  - `evaluate_stage2f.STAGE2F_FITNESS_SEED = 60`, `STAGE2F_GATE_SEED = 61`
  - `evaluate_stage2f.EXCLUDE_PATHS` (corrected priors + spent 2d/2e splits)
  - `evaluate_stage2f.FITNESS_COUNTS = {"cross": 6, "regular": 40, "asymmetric": 40}`
  - `evaluate_stage2f.GATE_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}`
  - `evaluate_stage2f.build_fitness_split() -> list` (86)
  - `evaluate_stage2f.build_gate_split(fitness) -> list` (100, disjoint from fitness + excludes)
  - `evaluate_stage2f._freeze(path, builder) -> list`

- [ ] **Step 1: Write failing tests for the split builders**

Add to `test_stage2f.py`:

```python
class TestSplitBuilders(unittest.TestCase):
    def test_fitness_and_gate_counts_and_disjointness(self):
        from evaluate_stage2f import (
            EXCLUDE_PATHS, build_fitness_split, build_gate_split,
        )
        fitness = build_fitness_split()
        gate = build_gate_split(fitness)
        # counts
        self.assertEqual(len(fitness), 86)
        self.assertEqual(len(gate), 100)
        for split, n in ((fitness, {"cross": 6, "regular": 40, "asymmetric": 40}),
                         (gate, {"cross": 12, "regular": 44, "asymmetric": 44})):
            for lay, k in n.items():
                self.assertEqual(sum(1 for s in split if s.layout == lay), k)
        f_keys = {_scenario_key(s) for s in fitness}
        g_keys = {_scenario_key(s) for s in gate}
        self.assertEqual(len(f_keys), 86)
        self.assertEqual(len(g_keys), 100)
        self.assertTrue(f_keys.isdisjoint(g_keys))              # gate INTERSECT fitness = 0
        # disjoint from every excluded prior/spent split
        excl = []
        for p in EXCLUDE_PATHS:
            excl += load_scenarios(p)
        e_keys = {_scenario_key(s) for s in excl}
        self.assertTrue(f_keys.isdisjoint(e_keys))
        self.assertTrue(g_keys.isdisjoint(e_keys))
        # every cross scenario (both splits) uses a diagonal heading
        from evaluate_stage2f import DIAGONAL_HEADINGS
        diag = {round(h, 6) for h in DIAGONAL_HEADINGS}
        for s in fitness + gate:
            if s.layout == "cross":
                self.assertIn(round(s.start.heading, 6), diag)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest test_stage2f.py -k "SplitBuilders" -q`
Expected: FAIL (`cannot import name 'build_fitness_split'`).

- [ ] **Step 3: Add constants and builders to `evaluate_stage2f.py`**

Insert after `DIAGONAL_HEADINGS`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest test_stage2f.py -k "SplitBuilders" -q`
Expected: PASS. (Runs the witness filter twice; a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add evaluate_stage2f.py test_stage2f.py
git commit -m "feat(stage2f): fresh 86/100 split builders with preregistered seeds"
```

---

### Task 5: `evaluate_stage2f` orchestration — reproduction gate, one-shot run, interpretation, main

**Files:**
- Modify: `evaluate_stage2f.py` (`_closed_loop_breakdown`, `_reproduction_gate`, `_one_shot`, `_interpretation`, `_timing_smoke`, `main`)
- Test: `test_stage2f.py`

**Interfaces:**
- Consumes: builders/constants from Task 4; `warm_start_readout`, `evaluate_policy`, `train_readout_by_reward_warmstart`, `aligned_fitness`; `_breakdown`, `_gate`, `_run_per_scenario`, `RecurrentController`.
- Produces:
  - `evaluate_stage2f._reproduction_gate(recurrent, esn, repro_ref, layouts) -> dict`
  - `evaluate_stage2f._one_shot(recurrent, repro_ref, fitness, gate, layouts) -> dict`
  - `evaluate_stage2f._interpretation(recurrent, ablation) -> str`
  - `evaluate_stage2f.main(argv=None)` with `--smoke`, writing `runs/stage2f/{results,gate_results}.json`

- [ ] **Step 1: Write failing tests for the interpretation branches**

Add to `test_stage2f.py`:

```python
class TestInterpretation(unittest.TestCase):
    def _res(self, rate, passes):
        return {"overall_arrival_rate": rate, "passes_gate": passes}

    def test_pass_and_beats_ablation_is_scoped(self):
        from evaluate_stage2f import _interpretation
        msg = _interpretation(self._res(0.92, True), self._res(0.80, False))
        self.assertIn("witnessed", msg.lower())        # scoped, not a broad claim
        self.assertIn("dopamine", msg.lower())

    def test_both_fail_is_bounded_and_confounded(self):
        from evaluate_stage2f import _interpretation
        msg = _interpretation(self._res(0.55, False), self._res(0.40, False))
        self.assertIn("confounded", msg.lower())
        self.assertNotIn("interface is not", msg.lower())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest test_stage2f.py -k "Interpretation" -q`
Expected: FAIL (`cannot import name '_interpretation'`).

- [ ] **Step 3: Add the orchestration functions and `main`**

Append to `evaluate_stage2f.py`:

```python
def _closed_loop_breakdown(esn, scenarios, layouts):
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def _reproduction_gate(recurrent: bool, esn, repro_ref, layouts) -> dict:
    """Assert the warm start reproduces the known-policy fingerprint (arrivals==22
    on the 40 Stage 2d fitness scenarios) before optimizing. Coefficient-
    independent: uses the 2e episode-reward `mean_reward`, not the aligned fitness."""
    out = evaluate_policy(esn, repro_ref, layouts)
    exp = REPRO_REWARD[recurrent]
    ok = (out["arrivals"] == REPRO_ARRIVALS
          and abs(out["mean_reward"] - exp) <= REPRO_REWARD_TOL)
    if not ok:
        raise SystemExit(
            f"reproduction gate FAILED (recurrent={recurrent}): got "
            f"{out['arrivals']}/{out['n']} arrivals, reward {out['mean_reward']:.4f}; "
            f"expected {REPRO_ARRIVALS}, reward ~{exp}. Stage 2c readout not "
            f"reconstructed exactly.")
    return out


def _one_shot(recurrent: bool, repro_ref, fitness, gate, layouts) -> dict:
    esn, theta0 = warm_start_readout(recurrent, layouts)
    repro = _reproduction_gate(recurrent, esn, repro_ref, layouts)
    esn, info = train_readout_by_reward_warmstart(
        esn, theta0, fitness, layouts, CEM_BUDGET, fitness_fn=aligned_fitness)
    breakdown = _closed_loop_breakdown(esn, gate, layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    return {
        "passes_gate": passes, "overall_arrival_rate": overall,
        "by_layout_arrival_rate": per_layout, "breakdown": breakdown,
        "reproduction": repro,
        "warmstart_fitness": info["warmstart_fitness"],
        "warmstart_outcomes_on_fitness": info["warmstart_outcomes"],
        "best_fitness": info["best_fitness"],
        "improved_over_warmstart": info["improved_over_warmstart"],
        "best_outcomes_on_fitness": info["best_outcomes"],
        "best_theta": info["best_theta"],
        "train_history": info["history"],
    }


def _interpretation(recurrent: dict, ablation: dict) -> str:
    r = recurrent["overall_arrival_rate"]
    a = ablation["overall_arrival_rate"]
    lead = (f"Recurrent {r:.3f} vs memoryless ablation {a:.3f} (references: "
            f"waypoint witness {REF_WAYPOINT_GATE:.2f}, Stage 2b SM "
            f"{REF_STATE_MACHINE_GATE:.2f}, Stage 2c recurrent "
            f"{REF_STAGE2C_RECURRENT:.2f}, Stage 2d recurrent "
            f"{REF_STAGE2D_RECURRENT:.2f}, Stage 2e recurrent "
            f"{REF_STAGE2E_RECURRENT:.2f}).")
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " Arrival-primary aligned fitness, fine-tuning temporal "
                "memory over the observation interface, clears a fresh gate and "
                "beats its ablation. Scoped to this diagonal waypoint-witnessed "
                "cross stratum plus the cardinal regular/asym strata, the interface "
                "is not the bottleneck -- not a claim about cross scenarios outside "
                "the witnessed eligible set. Motivates the biologically grounded "
                "memory/action-selection work (dopamine as modulation/teaching, "
                "never goal bearing).")
    if recurrent["passes_gate"]:
        return (lead + " The warm-started recurrent policy clears the fresh gate but "
                "does not clearly beat its ablation, so aligned reward exploited the "
                "reactive policy state (not memory specifically); the interface is "
                "sufficient and memory is not required for it, over this stratum.")
    return (lead + " Aligned-reward fine-tuning did not clear the fresh gate. Per "
            "the pass-only-is-conclusive asymmetry this stays bounded and confounded "
            "between reward alignment and scenario coverage (and residually "
            "capacity/optimization); it does not implicate the interface. Report the "
            "recurrent-ablation gap and the collision/arrival breakdown; diagnose "
            "before any interface claim or dopamine/connectome learning.")


def _timing_smoke(fitness, layouts, n_evals: int = 3):
    import numpy as np
    esn, theta0 = warm_start_readout(True, layouts)
    d = theta0.size // 2
    esn.W_out = np.asarray(theta0).reshape(2, d)

    def one_eval():
        aligned_fitness(esn, fitness, layouts)

    one_eval()  # warm up
    t0 = time.perf_counter()
    for _ in range(n_evals):
        one_eval()
    per_eval = (time.perf_counter() - t0) / n_evals
    total = per_eval * CEM_BUDGET.population * CEM_BUDGET.n_iter * 2
    print(f"[smoke] {len(fitness)} scenarios/eval; {per_eval:.3f}s per fitness eval")
    print(f"[smoke] projected TOTAL (pop {CEM_BUDGET.population} x iters "
          f"{CEM_BUDGET.n_iter} x 2 models): {total/60:.1f} min")
    return per_eval, total


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2f aligned-reward warm-start.")
    parser.add_argument("--fitness-split", default="runs/stage2f/fitness_split.json")
    parser.add_argument("--gate-split", default="runs/stage2f/gate_split.json")
    parser.add_argument("--out", default="runs/stage2f/results.json")
    parser.add_argument("--gate-out", default="runs/stage2f/gate_results.json")
    parser.add_argument("--smoke", action="store_true",
                        help="timing-only smoke test; no optimization, no gate")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    fitness = _freeze(args.fitness_split, build_fitness_split)
    gate = _freeze(args.gate_split, lambda: build_gate_split(fitness))
    repro_ref = select_fitness_scenarios(load_scenarios(REPRO_REFERENCE_SPLIT))
    print(f"Stage 2f: fitness {len(fitness)} (6/40/40, diagonal cross), gate "
          f"{len(gate)} (12/44/44, diagonal cross). Repro ref {len(repro_ref)}.")

    if args.smoke:
        _timing_smoke(fitness, layouts)
        return

    print(f"Warm-start CEM {CEM_BUDGET} (recurrent, then ablation)...")
    t0 = time.perf_counter()
    recurrent = _one_shot(True, repro_ref, fitness, gate, layouts)
    ablation = _one_shot(False, repro_ref, fitness, gate, layouts)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min.")

    interpretation = _interpretation(recurrent, ablation)
    provenance = {p: sha256(p) for p in EXCLUDE_PATHS}
    provenance[args.fitness_split] = sha256(args.fitness_split)
    provenance[args.gate_split] = sha256(args.gate_split)
    strip = lambda d: {k: v for k, v in d.items()
                       if k not in ("train_history", "best_theta")}
    gate_payload = {
        "fitness_split": args.fitness_split, "gate_split": args.gate_split,
        "repro_reference_split": REPRO_REFERENCE_SPLIT,
        "provenance_sha256": provenance,
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "reservoir_config": RESERVOIR_CONFIG, "esn_seed": ESN_SEED,
        "cem_budget": vars(CEM_BUDGET), "warm_start": "stage2c_readout",
        "fitness_counts": FITNESS_COUNTS, "gate_counts": GATE_COUNTS,
        "cross_note": ("diagonal-heading waypoint-witnessed cross; NOT "
                       "orientation-comparable to prior stages; 0.80 cross gate "
                       "needs 10/12."),
        "recurrent": strip(recurrent),
        "memoryless_ablation": strip(ablation),
        "references": {"waypoint_witness": REF_WAYPOINT_GATE,
                       "state_machine": REF_STATE_MACHINE_GATE,
                       "stage2c_recurrent": REF_STAGE2C_RECURRENT,
                       "stage2d_recurrent": REF_STAGE2D_RECURRENT,
                       "stage2e_recurrent": REF_STAGE2E_RECURRENT},
        "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["recurrent"] = recurrent          # full, incl. best_theta + history
    results_payload["memoryless_ablation"] = ablation
    Path(args.out).write_text(json.dumps(results_payload, indent=2) + "\n")

    for name, res in (("Recurrent", recurrent), ("Ablation", ablation)):
        print(f"{name} gate:  overall={res['overall_arrival_rate']:.3f} "
              f"{res['by_layout_arrival_rate']} "
              f"{'PASS' if res['passes_gate'] else 'FAIL'} "
              f"(warm {res['warmstart_fitness']:.3f} -> best {res['best_fitness']:.3f})")
    print(f"Wrote {args.gate_out} and {args.out}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the interpretation tests AND the full 2f + 2e suites**

Run: `python -m pytest test_stage2f.py test_stage2e.py -q`
Expected: PASS (all 2f tests + all 2e regression tests green).

- [ ] **Step 5: Commit**

```bash
git add evaluate_stage2f.py test_stage2f.py
git commit -m "feat(stage2f): one-shot orchestration, scoped interpretation, main"
```

---

### Task 6: Freeze the fresh splits and run the timing smoke

**Files:**
- Create (generated artifacts): `runs/stage2f/fitness_split.json`, `runs/stage2f/gate_split.json`

**Interfaces:**
- Consumes: `evaluate_stage2f.main --smoke`.
- Produces: the two frozen split files (committed) and the projected wall-clock cost.

- [ ] **Step 1: Freeze the splits and project cost (one command)**

Run: `python -u evaluate_stage2f.py --smoke`
Expected: prints `fitness 86 ... gate 100 ...`, writes both split files, and prints `[smoke] projected TOTAL ... min`. Sanity: with 86 scenarios/eval the projection should be roughly ~2x the Stage 2e 107 min (order ~180–220 min).

- [ ] **Step 2: Verify the frozen splits pass every disjointness assertion**

Run: `python -m pytest test_stage2f.py -k "SplitBuilders or ExpandedCross" -q`
Expected: PASS (builders now short-circuit to the frozen files where present; disjointness holds).

- [ ] **Step 3: Record the split shas**

Run: `python -c "from evaluate_stage2 import sha256; print(sha256('runs/stage2f/fitness_split.json')); print(sha256('runs/stage2f/gate_split.json'))"`
Expected: two sha256 hex strings (note them for the README).

- [ ] **Step 4: Commit the frozen splits**

```bash
git add runs/stage2f/fitness_split.json runs/stage2f/gate_split.json
git commit -m "chore(stage2f): freeze fresh 86/100 splits (seeds 60/61)"
```

---

### Task 7: Execute the one-shot run

**Files:**
- Create (generated artifacts): `runs/stage2f/results.json`, `runs/stage2f/gate_results.json`

**Interfaces:**
- Consumes: the frozen splits from Task 6; `evaluate_stage2f.main`.
- Produces: the gate results for both models (the run is long — projected ~3–4 h; launch as a tracked background task, do not block).

- [ ] **Step 1: Launch the run in the background**

Run (tracked background task): `python -u evaluate_stage2f.py`
Expected: on completion, writes `runs/stage2f/{results,gate_results}.json` and prints both models' gate lines + the interpretation. The frozen splits are reused (deterministic, one-shot). Do NOT relaunch while it is running.

- [ ] **Step 2: Verify the outputs are well-formed**

Run: `python -c "import json; d=json.load(open('runs/stage2f/gate_results.json')); print(d['recurrent']['overall_arrival_rate'], d['recurrent']['passes_gate'], d['memoryless_ablation']['overall_arrival_rate']); print(d['recurrent']['reproduction']['arrivals'], d['memoryless_ablation']['reproduction']['arrivals'])"`
Expected: prints the two overall rates + pass flags, and `22 22` (reproduction fingerprint held for both models).

- [ ] **Step 3: Commit the results**

```bash
git add runs/stage2f/results.json runs/stage2f/gate_results.json
git commit -m "runs(stage2f): one-shot aligned-reward gate results"
```

---

### Task 8: Write `runs/stage2f/README.md` and update `HANDOFF.md`

**Files:**
- Create: `runs/stage2f/README.md`
- Modify: `HANDOFF.md` (append a Stage 2f entry)

**Interfaces:**
- Consumes: `runs/stage2f/gate_results.json` (the actual numbers).
- Produces: the bounded write-up + handoff entry.

- [ ] **Step 1: Write `runs/stage2f/README.md` from the actual results**

Mirror `runs/stage2e/README.md` structure, using the real numbers from `gate_results.json`:
- Headline verdict scoped by the pass-only asymmetry (PASS scoped to the diagonal witnessed cross + cardinal strata; or bounded FAILURE, confounded between reward alignment and scenario coverage — never an interface verdict).
- What changed vs 2e: arrival-primary fitness (one more arrival always wins; the 22-arrival/1.555 < 18-arrival/1.686 weighted-sum failure it fixes), 86-scenario fitness set, diagonal witnessed cross gate.
- Reproduction gate (arrivals==22 for both) + warm-start outcomes on the 86-set.
- Results table: overall / cross / regular / asym / warm→best fitness / pass, both models; arrival/collision/timeout counts.
- Provenance: fitness sha, gate sha (from Task 6 Step 3), fully exact-identity disjoint from the corrected priors + spent 2d/2e splits + each other; cross documented as diagonal/witnessed and NOT orientation-comparable; regular/asym trend-comparable. Link the spec + this plan.

- [ ] **Step 2: Append a Stage 2f entry to `HANDOFF.md`**

Add a short Stage 2f section: the aligned-fitness change and its motivation, the result (pass/fail + recurrent vs ablation vs the 2e 0.516 reference), the scoped verdict, and the diagnosed next step (bounded — no interface claim, no connectome/dopamine advance on a failure).

- [ ] **Step 3: Commit the write-up**

```bash
git add runs/stage2f/README.md HANDOFF.md
git commit -m "docs(stage2f): bounded write-up + handoff entry"
```

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task:
- Arrival-primary fitness + frozen shaping (spec §1) → Task 1 (+ ordering/bounds tests).
- More fitness scenarios / 86-set (spec §2) → Task 4 (`FITNESS_COUNTS`) + Task 5 (train on it).
- Held from 2e: warm start, frozen reservoir, matched ablation, CEM best-ever, `n_iter=25` unconditional, reproduction==22, persist warm-start outcomes on the 86-set (spec §3, review pt 4) → Task 2 (`warmstart_outcomes`, default preserves 2e) + Task 5 (`_reproduction_gate`, `_one_shot`).
- Fresh gate + expanded cross, preregistered seeds/order, disjointness asserts, counts 86/100 (spec "Fresh gate" + Preregistration) → Tasks 3–4.
- Waypoint-witnessed rename + scoped pass claim (review pts 2–3) → Task 3 (docstring/tests) + Task 5 (`_interpretation`).
- Selection discipline / no gate tuning (spec) → Global Constraints + frozen coefficients in Task 1.
- Files list (spec "Files") → `stage2f.py` (T1), `stage2e` extension (T2), `evaluate_stage2f.py` (T3–5), `test_stage2f.py` (T1–5), `runs/stage2f/*` (T6–8).
- Interpretation branches (spec) → Task 5 `_interpretation` + tests.

**2. Placeholder scan.** No TBD/TODO; every step has runnable code or an exact command.

**3. Type consistency.** `aligned_fitness(esn, scenarios, layouts)` matches the `fitness_fn` signature consumed by `evaluate_policy`/`train_readout_by_reward_warmstart` (Task 2) and passed in `_one_shot` (Task 5). `generate_expanded_cross(seed, exclude, count)` matches its callers in `build_fitness_split`/`build_gate_split`. `_scenario_key` tuple `(layout, x, y, heading, target_x, target_y)` matches the `("cross", a[0], a[1], h, b[0], b[1])` exclusion key. `episode_secondary(scenario, result)` and `_final_xy` reuse the Stage 2d shape. `info` keys produced in Task 2 (`warmstart_fitness`, `warmstart_outcomes`, `best_fitness`, `improved_over_warmstart`, `best_theta`, `best_outcomes`, `history`, `best`) are exactly those read in Task 5 `_one_shot`.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-17-stage2f-aligned-reward.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review each, fast iteration.

**2. Inline Execution** — execute tasks in this session via executing-plans, with batch execution and checkpoints.

Note: Tasks 6–7 produce frozen one-shot artifacts and a multi-hour run; those steps are executed once and cannot be redone without spending a fresh gate.

**Which approach?**
