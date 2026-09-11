"""Small deterministic checks for the Stage 1 arena + neural runtime slice."""

import math
import unittest

import numpy as np
import scipy.sparse as sp

import simulation as sim
import brain as br
import evaluate as ev


class TestArena(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(sim.wrap_angle(3 * math.pi), -math.pi)   # range [-pi, pi)
        self.assertAlmostEqual(sim.wrap_angle(-3 * math.pi), -math.pi)
        self.assertAlmostEqual(sim.wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(sim.wrap_angle(math.pi / 2), math.pi / 2)

    def test_straight_movement(self):
        s = sim.CarState(0.0, 0.0, 0.0)
        for _ in range(10):
            s = sim.bicycle_step(s, 0.0, 0.02)
        self.assertAlmostEqual(s.x, sim.SPEED * 0.02 * 10, places=6)
        self.assertAlmostEqual(s.y, 0.0, places=9)
        self.assertAlmostEqual(s.heading, 0.0, places=9)

    def test_left_turn_increases_heading(self):
        s = sim.bicycle_step(sim.CarState(0, 0, 0), 0.3, 0.02)
        self.assertGreater(s.heading, 0.0)

    def test_right_turn_decreases_heading(self):
        s = sim.bicycle_step(sim.CarState(0, 0, 0), -0.3, 0.02)
        self.assertLess(s.heading, 0.0)

    def test_swept_arrival_not_skipped(self):
        # One big step overshoots the target but the swept path passes through it.
        sc = sim.Scenario(sim.CarState(0, 0, 0), 5.0, 0.0, timeout=30.0)
        r = sim.run_episode(sc, lambda o: 0.0, dt=5.0)  # speed*dt = 10 units
        self.assertEqual(r.outcome, "arrival")
        self.assertLess(r.elapsed_time, 5.0)

    def test_swept_boundary_failure(self):
        sc = sim.Scenario(sim.CarState(24.0, 0.0, 0.0), 100.0, 0.0, timeout=30.0)
        r = sim.run_episode(sc, lambda o: 0.0, dt=1.0)  # steps past +x wall
        self.assertEqual(r.outcome, "boundary")

    def test_start_on_target_is_arrival(self):
        sc = sim.Scenario(sim.CarState(0, 0, 0), 0.0, 0.0)
        r = sim.run_episode(sc, lambda o: 0.0)
        self.assertEqual(r.outcome, "arrival")
        self.assertEqual(r.elapsed_time, 0.0)


class TestScenarios(unittest.TestCase):
    def test_deterministic_generation(self):
        a = ev.generate_scenarios(5, seed=1)
        b = ev.generate_scenarios(5, seed=1)
        self.assertEqual([s.target_x for s in a], [s.target_x for s in b])
        self.assertEqual([s.start.heading for s in a], [s.start.heading for s in b])

    def test_targets_inside_arena(self):
        for s in ev.generate_scenarios(50, seed=7):
            self.assertLessEqual(abs(s.target_x) + sim.TARGET_RADIUS, sim.ARENA_BOUND)
            self.assertLessEqual(abs(s.target_y) + sim.TARGET_RADIUS, sim.ARENA_BOUND)

    def test_conventional_baseline_navigable(self):
        # Verifies the car/target setup is reachable before evaluating a brain.
        sc = ev.generate_scenarios(20, seed=3)
        summary = ev.evaluate_controller(sc, ev.conventional_baseline())
        self.assertGreaterEqual(summary.arrivals, 19)


class TestBrain(unittest.TestCase):
    def _chain_graph(self):
        # Edge i->j lives at row j, col i (postsynaptic rows). Chain 0->1->2.
        W = sp.csr_array(np.array([[0, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0]], dtype=np.float32))
        input_map = {"heading": (np.array([0]), np.array([0.0], dtype=np.float32))}
        output_map = {"left": np.array([2]), "right": np.array([1])}
        return br.Brain(W, input_map, output_map, br.RateParams(recurrent_gain=1.0))

    def test_directed_propagation(self):
        b = self._chain_graph()
        b.reset()
        stim = np.zeros(3, dtype=np.float32)
        stim[0] = 1.0
        for _ in range(50):
            b.step(stim)
        # Activity must have flowed 0 -> 1 -> 2 along the recorded direction.
        self.assertGreater(b.activity[1], 0.0)
        self.assertGreater(b.activity[2], 0.0)
        self.assertTrue(np.all(np.isfinite(b.activity)))
        self.assertTrue(np.all(b.activity <= 1.0 + 1e-6))

    def test_reset_zeroes_activity(self):
        b = self._chain_graph()
        b.step(np.ones(3, dtype=np.float32))
        self.assertGreater(b.activity.sum(), 0.0)
        b.reset()
        self.assertEqual(b.activity.sum(), 0.0)

    def test_encode_aligned_cue_is_max(self):
        b = self._chain_graph()
        stim = b.encode(heading=0.0, goal_bearing=0.0)  # preferred angle 0 -> cos=1
        self.assertAlmostEqual(stim[0], b.params.input_gain, places=5)

    def test_row_normalization(self):
        W = sp.csr_array(np.array([[0, 2, 2], [0, 0, 0], [3, 0, 0]], dtype=np.float32))
        Wn = br.normalize_rows(W).toarray()
        self.assertAlmostEqual(np.abs(Wn[0]).sum(), 1.0, places=5)  # nonempty -> 1
        self.assertEqual(np.abs(Wn[1]).sum(), 0.0)                  # empty stays 0
        self.assertAlmostEqual(np.abs(Wn[2]).sum(), 1.0, places=5)

    def test_adapter_symmetry(self):
        a = br.Adapter(gain=2.0, bias=0.0)
        self.assertEqual(a(0.5, 0.5), 0.0)      # equal pools -> no steering
        self.assertGreater(a(0.8, 0.2), 0.0)    # more left -> positive
        self.assertLess(a(0.2, 0.8), 0.0)


if __name__ == "__main__":
    unittest.main()
