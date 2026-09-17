"""Stage 2f tests: arrival-primary fitness + diagonal witnessed cross + fresh gate."""
from __future__ import annotations

import math
import unittest

import numpy as np

from street import (
    MAX_SIM_TIME, StreetCarState, StreetEpisodeResult, StreetScenario,
    initial_layouts, run_street_episode,
)
from evaluate_stage2 import load_scenarios, _scenario_key
from stage2b import WaypointController, is_outward_road_end
from evaluate_stage2d import select_fitness_scenarios
from stage2d import CEMConfig
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
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

    def test_default_path_outcome_dicts_have_no_fitness_key(self):
        # Regression: ensure the default path (fitness_fn=None) preserves 2e shape
        # with no "fitness" key in outcome dicts, even though we coalesce the fitness
        # function internally for the CEM objective.
        layouts = initial_layouts()
        fitness = select_fitness_scenarios(
            load_scenarios("runs/stage2d/train_split.json"))[:6]
        esn, theta0 = warm_start_readout(True, layouts)
        esn, info = train_readout_by_reward_warmstart(
            esn, theta0, fitness, layouts, CEMConfig(population=6, n_iter=2, seed=0))
        # no fitness_fn -> default path; both outcome dicts must remain 2e-shaped
        self.assertNotIn("fitness", info["warmstart_outcomes"])
        self.assertNotIn("fitness", info["best_outcomes"])


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
