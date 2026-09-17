"""Stage 2d tests: the reward-trained same-observation baseline.

Cover the reward shaping (bounded net progress, arrival dominance, collision and
standing-still properties), the CEM optimizer (improves a toy objective,
deterministic), the reward-training pipeline, and split exact-identity
disjointness from every prior Stage 2/2b split.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from street import (
    MAX_SIM_TIME, StreetCarState, StreetEpisodeResult, StreetScenario,
    initial_layouts, run_street_episode,
)
from stage2c import EchoStateNetwork, N_FEATURES, RecurrentController
from stage2d import (
    CEMConfig, W_ARRIVE, W_COLLIDE, W_PROGRESS, W_TIME, cem_maximize,
    episode_reward, train_readout_by_reward,
)
from evaluate_stage2 import load_scenarios


def _scenario(start=(0.0, 0.0), target=(10.0, 0.0)):
    return StreetScenario("regular", StreetCarState(start[0], start[1], 0.0, 0.0),
                          target[0], target[1])


def _result(outcome, final_xy, elapsed):
    # A one-point trajectory whose last position is `final_xy` (record=True form).
    return StreetEpisodeResult(outcome, elapsed, 0.0,
                               [(final_xy[0], final_xy[1], 0.0, 0.0)])


class TestReward(unittest.TestCase):
    def test_bounded_net_progress_uses_final_not_closest(self):
        # A car that gets near the goal then leaves earns net progress by its
        # FINAL position, not its closest approach.
        s = _scenario(start=(0.0, 0.0), target=(10.0, 0.0))
        far_end = episode_reward(s, _result("timeout", (2.0, 0.0), MAX_SIM_TIME))
        # final at x=2 -> net progress (10-8)/10 = 0.2
        self.assertAlmostEqual(far_end, W_PROGRESS * 0.2 - W_TIME, places=6)

    def test_standing_still_below_forward_progress(self):
        s = _scenario(start=(0.0, 0.0), target=(10.0, 0.0))
        standing = episode_reward(s, _result("timeout", (0.0, 0.0), MAX_SIM_TIME))
        forward = episode_reward(s, _result("timeout", (5.0, 0.0), MAX_SIM_TIME))
        self.assertLess(standing, forward)

    def test_arrival_dominates_max_shaping(self):
        # Arrival bonus alone must exceed the maximum possible shaping (progress=1).
        self.assertGreater(W_ARRIVE, W_PROGRESS)
        s = _scenario()
        arrival = episode_reward(s, _result("arrival", (10.0, 0.0), 5.0))
        best_non_arrival = episode_reward(s, _result("timeout", (10.0, 0.0), 0.0))
        self.assertGreater(arrival, best_non_arrival)

    def test_collision_is_penalized(self):
        s = _scenario()
        collision = episode_reward(s, _result("collision", (5.0, 0.0), 10.0))
        timeout = episode_reward(s, _result("timeout", (5.0, 0.0), 10.0))
        self.assertAlmostEqual(timeout - collision, W_COLLIDE, places=6)

    def test_net_progress_is_bounded(self):
        # Driving well past / away from the goal cannot blow up the reward.
        s = _scenario(start=(0.0, 0.0), target=(10.0, 0.0))
        way_past = episode_reward(s, _result("timeout", (-50.0, 0.0), MAX_SIM_TIME))
        self.assertGreaterEqual(way_past, -W_PROGRESS - W_TIME - 1e-9)


class TestCEM(unittest.TestCase):
    def test_improves_quadratic(self):
        target = np.array([0.5, -0.3, 1.0])
        fitness = lambda x: -float(np.sum((x - target) ** 2))
        mu, info = cem_maximize(fitness, 3, CEMConfig(population=40, n_iter=30, seed=0))
        self.assertLess(float(np.sum((mu - target) ** 2)), 1e-2)
        self.assertGreater(info["history"][-1]["best"], info["history"][0]["best"])

    def test_deterministic(self):
        fitness = lambda x: -float(np.sum(x ** 2))
        cfg = CEMConfig(population=20, n_iter=5, seed=7)
        a, _ = cem_maximize(fitness, 4, cfg)
        b, _ = cem_maximize(fitness, 4, cfg)
        self.assertTrue(np.allclose(a, b))


class TestRewardTraining(unittest.TestCase):
    def test_pipeline_sets_readout_and_improves_reward(self):
        layouts = initial_layouts()
        scenarios = load_scenarios("runs/stage2b/sm_train_split.json")[:2]
        esn = EchoStateNetwork(n_reservoir=16, seed=0, recurrent=True)
        D = 1 + N_FEATURES + esn.n

        def mean_reward(controller):
            total = 0.0
            for s in scenarios:
                controller.reset()
                r = run_street_episode(s, layouts[s.layout], controller, record=True)
                total += episode_reward(s, r)
            return total / len(scenarios)

        esn.W_out = np.zeros((2, D))
        before = mean_reward(RecurrentController(esn))
        train_readout_by_reward(esn, scenarios, layouts,
                                CEMConfig(population=8, n_iter=4, seed=0))
        self.assertEqual(esn.W_out.shape, (2, D))
        after = mean_reward(RecurrentController(esn))
        self.assertGreaterEqual(after, before)


class TestControllerInterface(unittest.TestCase):
    def test_is_observation_only(self):
        params = list(inspect.signature(RecurrentController.__call__).parameters)
        self.assertEqual(params, ["self", "obs"])


class TestSplitDisjointness(unittest.TestCase):
    """Validate the FROZEN, spent Stage 2d splits as they are on disk -- including
    the documented overlap defect (the as-run exclusion omitted training.json and
    heldout.json). Codifying the exact overlap keeps it from being silently
    'repaired' by regeneration; the spent one-shot gate must never be replaced."""

    def _keys(self, path):
        from evaluate_stage2b import _scenario_key
        return {_scenario_key(s) for s in load_scenarios(path)}

    def test_frozen_splits_internal_and_clean_disjointness(self):
        gate = self._keys("runs/stage2d/gate_split.json")
        train = self._keys("runs/stage2d/train_split.json")
        self.assertEqual(len(gate), 100)   # 12/44/44
        self.assertEqual(len(train), 96)   # 8/44/44
        self.assertTrue(gate.isdisjoint(train))            # gate/train are clean
        for clean in ("runs/stage2/dev_split.json",
                      "runs/stage2b/gate_split.json",
                      "runs/stage2b/sm_train_split.json"):
            self.assertTrue(gate.isdisjoint(self._keys(clean)))
            self.assertTrue(train.isdisjoint(self._keys(clean)))

    def test_frozen_splits_known_overlap_defect(self):
        gate = self._keys("runs/stage2d/gate_split.json")
        train = self._keys("runs/stage2d/train_split.json")
        leaked = self._keys("runs/stage2/training.json") | \
            self._keys("runs/stage2/heldout.json")
        # Exact, documented leak from the as-run exclusion omission.
        self.assertEqual(len(gate & leaked), 8)   # 2 training + 6 heldout
        self.assertEqual(len(train & leaked), 6)   # 1 training + 5 heldout

    def test_corrected_prior_list_covers_the_leak(self):
        from evaluate_stage2d import PRIOR_SPLIT_PATHS
        self.assertIn("runs/stage2/training.json", PRIOR_SPLIT_PATHS)
        self.assertIn("runs/stage2/heldout.json", PRIOR_SPLIT_PATHS)


if __name__ == "__main__":
    unittest.main()
