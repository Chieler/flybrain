"""Stage 2c tests: the same-observation recurrent/history baseline.

Cover the observation-only interface discipline, reservoir determinism, the
matched memoryless ablation, ridge fitting, and closed-loop control validity.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from simulation import MAX_STEERING
from street import (
    Control, MAX_ACCEL, MAX_BRAKE, StreetObservation, initial_layouts,
    run_street_episode,
)
from stage2c import (
    EchoStateNetwork, N_FEATURES, RecurrentController, TrajectoryRecorder,
    observation_features,
)
from stage2b import WaypointController
from evaluate_stage2 import load_scenarios


def _obs(heading=0.0, goal_bearing=0.0, speed=1.0, ranges=(8.0, 8.0, 8.0, 8.0, 8.0)):
    return StreetObservation(heading, goal_bearing, speed, ranges)


class TestFeatures(unittest.TestCase):
    def test_is_observation_only(self):
        # The feature builder takes exactly one argument: the observation.
        params = list(inspect.signature(observation_features).parameters)
        self.assertEqual(params, ["obs"])

    def test_feature_dimension_and_normalization(self):
        f = observation_features(_obs(heading=math.pi / 2, goal_bearing=0.0,
                                      speed=3.0, ranges=(8.0, 4.0, 0.0, 8.0, 8.0)))
        self.assertEqual(f.shape, (N_FEATURES,))
        # sin/cos(heading=pi/2) = (1, 0); speed 3/MAX_SPEED = 1; ranges in [0,1].
        self.assertAlmostEqual(f[0], 1.0)
        self.assertAlmostEqual(f[1], 0.0)
        self.assertAlmostEqual(f[4], 1.0)
        self.assertTrue(np.all(f[5:] >= 0.0) and np.all(f[5:] <= 1.0))

    def test_angles_are_faithful_sin_cos_not_difference(self):
        # Two observations with the SAME (goal_bearing - heading) but different
        # absolute angles must produce different features -- proving no engineered
        # difference term was injected.
        a = observation_features(_obs(heading=0.0, goal_bearing=0.5))
        b = observation_features(_obs(heading=1.0, goal_bearing=1.5))
        self.assertFalse(np.allclose(a, b))


class TestReservoir(unittest.TestCase):
    def test_seed_is_deterministic(self):
        a = EchoStateNetwork(seed=3)
        b = EchoStateNetwork(seed=3)
        self.assertTrue(np.array_equal(a.W_in, b.W_in))
        self.assertTrue(np.array_equal(a.W, b.W))

    def test_spectral_radius_scaling(self):
        esn = EchoStateNetwork(n_reservoir=20, spectral_radius=0.7, seed=1)
        radius = float(np.max(np.abs(np.linalg.eigvals(esn.W))))
        self.assertAlmostEqual(radius, 0.7, places=6)

    def test_ablation_is_memoryless(self):
        # recurrent=False: the readout vector at a given observation must be
        # independent of the preceding observation history.
        esn = EchoStateNetwork(seed=2, recurrent=False)
        here = _obs(heading=0.3, goal_bearing=-0.4, ranges=(2.0, 5.0, 8.0, 1.0, 7.0))
        esn.reset()
        esn._readout_vec(observation_features(_obs(heading=1.2)))
        after_a = esn._readout_vec(observation_features(here))
        esn.reset()
        esn._readout_vec(observation_features(_obs(heading=-2.0, speed=0.1)))
        after_b = esn._readout_vec(observation_features(here))
        self.assertTrue(np.allclose(after_a, after_b))

    def test_recurrent_has_memory(self):
        # recurrent=True: the same observation after different histories yields a
        # different reservoir state.
        esn = EchoStateNetwork(seed=2, recurrent=True, spectral_radius=0.9)
        here = _obs(heading=0.3, goal_bearing=-0.4)
        esn.reset()
        esn._readout_vec(observation_features(_obs(heading=1.2)))
        after_a = esn._readout_vec(observation_features(here))
        esn.reset()
        esn._readout_vec(observation_features(_obs(heading=-2.0, speed=0.1)))
        after_b = esn._readout_vec(observation_features(here))
        self.assertFalse(np.allclose(after_a, after_b))


class TestRidgeFit(unittest.TestCase):
    def test_fits_linear_target_in_sample(self):
        # Ridge should reproduce a target that is a smooth function of features.
        esn = EchoStateNetwork(n_reservoir=40, seed=5, recurrent=False)
        rng = np.random.default_rng(0)
        feats = [rng.uniform(-1, 1, (30, N_FEATURES)) for _ in range(4)]
        targs = [np.stack([np.sin(f[:, 0]), np.cos(f[:, 1])], axis=1) for f in feats]
        esn.fit(feats, targs, ridge=1e-6)
        # In-sample prediction error is small.
        esn.reset()
        preds = np.array([esn.predict(u) for u in feats[0]])
        self.assertLess(np.mean((preds - targs[0]) ** 2), 1e-2)


class TestRecurrentController(unittest.TestCase):
    def test_is_observation_only(self):
        params = list(inspect.signature(RecurrentController.__call__).parameters)
        self.assertEqual(params, ["self", "obs"])

    def test_returns_clamped_control(self):
        esn = EchoStateNetwork(n_reservoir=16, seed=7)
        # Force a huge readout to check clamping.
        esn.W_out = np.full((2, 1 + N_FEATURES + esn.n), 1e6)
        ctrl = RecurrentController(esn)
        ctrl.reset()
        out = ctrl(_obs())
        self.assertIsInstance(out, Control)
        self.assertLessEqual(out.steering, MAX_STEERING)
        self.assertGreaterEqual(out.steering, -MAX_STEERING)
        self.assertLessEqual(out.acceleration, MAX_ACCEL)
        self.assertGreaterEqual(out.acceleration, -MAX_BRAKE)

    def test_reset_clears_reservoir_state(self):
        esn = EchoStateNetwork(seed=1).fit(
            [np.zeros((3, N_FEATURES))], [np.zeros((3, 2))], ridge=1.0)
        ctrl = RecurrentController(esn)
        ctrl.reset()
        ctrl(_obs(heading=1.0))
        state_after_step = esn.x.copy()
        ctrl.reset()
        self.assertTrue(np.array_equal(esn.x, np.zeros(esn.n)))
        self.assertFalse(np.array_equal(state_after_step, np.zeros(esn.n)))


class TestBehaviorCloningPipeline(unittest.TestCase):
    def test_recorder_harvests_teacher_demo(self):
        layouts = initial_layouts()
        scenarios = load_scenarios("runs/stage2b/sm_train_split.json")
        scenario = next(s for s in scenarios if s.layout in layouts)
        layout = layouts[scenario.layout]
        rec = TrajectoryRecorder(WaypointController(scenario, layout))
        rec.reset()
        result = run_street_episode(scenario, layout, rec)
        self.assertGreater(len(rec.features), 0)
        self.assertEqual(len(rec.features), len(rec.targets))
        self.assertEqual(rec.features[0].shape, (N_FEATURES,))
        self.assertIn(result.outcome, {"arrival", "collision", "timeout"})

    def test_imitates_teacher_in_sample(self):
        # Train the recurrent readout on a handful of teacher demos and check the
        # in-sample action error is small (pipeline learns to reproduce actions).
        layouts = initial_layouts()
        scenarios = load_scenarios("runs/stage2b/sm_train_split.json")[:6]
        feats, targs = [], []
        for scenario in scenarios:
            layout = layouts[scenario.layout]
            rec = TrajectoryRecorder(WaypointController(scenario, layout))
            rec.reset()
            if run_street_episode(scenario, layout, rec).outcome == "arrival":
                feats.append(np.array(rec.features))
                targs.append(np.array(rec.targets))
        self.assertGreater(len(feats), 0)
        esn = EchoStateNetwork(n_reservoir=48, seed=0, recurrent=True).fit(
            feats, targs, ridge=1e-4)
        esn.reset()
        preds = np.array([esn.predict(u) for u in feats[0]])
        self.assertLess(np.mean((preds[:, 0] - targs[0][:, 0]) ** 2), 0.05)


if __name__ == "__main__":
    unittest.main()
