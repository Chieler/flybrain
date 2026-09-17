"""Stage 2e tests: warm-start reward fine-tuning.

Cover the deterministic Stage 2c warm-start reconstruction (reproduces 22/40 and
reward ~1.1 for both models), the best-ever CEM extension (records the best
sampled candidate; warm-start guard so the trained policy never ships worse than
theta0), the persisted best readout, and exact-identity gate disjointness from
every prior split plus the spent Stage 2d splits and the fitness set.
"""

from __future__ import annotations

import unittest

import numpy as np

from street import initial_layouts
from evaluate_stage2 import load_scenarios
from evaluate_stage2b import _scenario_key
from evaluate_stage2d import select_fitness_scenarios
from stage2d import CEMConfig, cem_maximize
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
)


class TestWarmStartReconstruction(unittest.TestCase):
    def test_reproduces_stage2c_readout(self):
        layouts = initial_layouts()
        fitness = select_fitness_scenarios(
            load_scenarios("runs/stage2d/train_split.json"))
        for recurrent, expected_reward in ((True, 1.106), (False, 1.086)):
            esn, theta0 = warm_start_readout(recurrent, layouts)
            self.assertEqual(theta0.size, 150)
            out = evaluate_policy(esn, fitness, layouts)
            self.assertEqual(out["arrivals"], 22)          # exact reproduction
            self.assertAlmostEqual(out["mean_reward"], expected_reward, delta=0.02)

    def test_warm_start_is_deterministic(self):
        _, a = warm_start_readout(True)
        _, b = warm_start_readout(True)
        self.assertTrue(np.allclose(a, b))


class TestCEMBestEver(unittest.TestCase):
    def test_records_best_ever_candidate(self):
        target = np.array([0.5, -0.3, 1.0])
        fitness = lambda x: -float(np.sum((x - target) ** 2))
        mu, info = cem_maximize(fitness, 3, CEMConfig(population=40, n_iter=20, seed=0))
        self.assertIn("best", info)
        self.assertEqual(np.asarray(info["best"]["theta"]).shape, (3,))
        # Best-ever == the max per-iteration best, and its theta re-scores to it.
        self.assertAlmostEqual(info["best"]["fitness"],
                               max(h["best"] for h in info["history"]))
        self.assertAlmostEqual(fitness(np.asarray(info["best"]["theta"])),
                               info["best"]["fitness"], places=9)

    def test_warm_start_mean_is_respected(self):
        # With sigma tiny and mu seeded far from optimum, samples cluster at init.
        fitness = lambda x: -float(np.sum(x ** 2))
        cfg = CEMConfig(population=30, n_iter=1, init_std=1e-6, seed=1)
        _, info = cem_maximize(fitness, 4, cfg, init_mu=np.array([5.0, 5.0, 5.0, 5.0]))
        self.assertTrue(np.allclose(info["best"]["theta"], 5.0, atol=1e-3))


class TestWarmStartTraining(unittest.TestCase):
    def test_never_ships_worse_than_warm_start(self):
        layouts = initial_layouts()
        fitness = select_fitness_scenarios(
            load_scenarios("runs/stage2d/train_split.json"))[:6]
        esn, theta0 = warm_start_readout(True, layouts)
        esn, info = train_readout_by_reward_warmstart(
            esn, theta0, fitness, layouts, CEMConfig(population=6, n_iter=2, seed=0))
        # The shipped policy's fitness is >= the warm-start fitness.
        self.assertGreaterEqual(info["best_fitness"], info["warmstart_fitness"] - 1e-9)
        self.assertEqual(len(info["best_theta"]), 150)
        self.assertIn("best_outcomes", info)
        # Persisted best readout reloads onto the esn equivalently.
        self.assertTrue(np.allclose(
            esn.W_out, np.asarray(info["best_theta"]).reshape(2, 75)))


class TestGateDisjointness(unittest.TestCase):
    def test_stage2e_gate_disjoint_from_all(self):
        from evaluate_stage2e import build_gate_split, EXCLUDE_PATHS, FITNESS_SPLIT

        gate = build_gate_split()
        self.assertEqual(len(gate), 95)   # 7/44/44
        gate_keys = {_scenario_key(s) for s in gate}
        self.assertEqual(len(gate_keys), len(gate))
        prior = []
        for p in EXCLUDE_PATHS:
            prior += load_scenarios(p)
        prior_keys = {_scenario_key(s) for s in prior}
        self.assertTrue(gate_keys.isdisjoint(prior_keys))
        # Disjoint from the fitness set (the 40 Stage 2d fitness scenarios).
        fitness = select_fitness_scenarios(load_scenarios(FITNESS_SPLIT))
        self.assertTrue(gate_keys.isdisjoint({_scenario_key(s) for s in fitness}))
        # Uses all 7 remaining eligible cross.
        self.assertEqual(sum(1 for s in gate if s.layout == "cross"), 7)


if __name__ == "__main__":
    unittest.main()
