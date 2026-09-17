"""Stage 2f tests: arrival-primary fitness + diagonal witnessed cross + fresh gate."""
from __future__ import annotations

import math
import unittest

import numpy as np

from street import (
    MAX_SIM_TIME, StreetCarState, StreetEpisodeResult, StreetScenario,
    initial_layouts,
)
from evaluate_stage2 import load_scenarios
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
