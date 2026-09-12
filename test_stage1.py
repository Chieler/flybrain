"""Small deterministic checks for the Stage 1 arena + neural runtime slice."""

import json
import math
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

import simulation as sim
import brain as br
import evaluate as ev
import prepare_connectome as prep


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

    def test_timeout_does_not_overshoot_final_step(self):
        sc = sim.Scenario(sim.CarState(0, 0, 0), 100.0, 0.0, timeout=0.01)
        r = sim.run_episode(sc, lambda o: 0.0, dt=0.02)
        self.assertEqual(r.outcome, "timeout")
        self.assertAlmostEqual(r.elapsed_time, 0.01)
        self.assertAlmostEqual(r.path_length, sim.SPEED * 0.01)

    def test_swept_event_records_interpolated_heading(self):
        steering = 0.5
        sc = sim.Scenario(sim.CarState(0, 0, 0), 0.03, 0.0,
                          target_radius=0.005, timeout=1.0)
        r = sim.run_episode(sc, lambda o: steering, dt=0.02, record=True)
        t = r.elapsed_time / 0.02
        expected = sim.SPEED / sim.WHEELBASE * math.tan(steering) * 0.02 * t
        self.assertEqual(r.outcome, "arrival")
        self.assertAlmostEqual(r.trajectory[-1][2], expected)


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

    def test_transmitter_sign_policy(self):
        self.assertEqual(prep.transmitter_sign("acetylcholine", 0.9), 1)
        self.assertEqual(prep.transmitter_sign("gaba", 0.9), -1)
        self.assertEqual(prep.transmitter_sign("glutamate", 0.6), -1)
        self.assertEqual(prep.transmitter_sign("acetylcholine", 0.4), 0)  # low conf
        self.assertEqual(prep.transmitter_sign("unclear", 0.9), 0)
        self.assertEqual(prep.transmitter_sign("dopamine", 0.9), 0)       # modulatory
        self.assertEqual(prep.transmitter_sign(None, None), 0)

    def test_source_filenames_match_download_urls(self):
        for filename, url in prep.SOURCES.values():
            self.assertEqual(filename, url.rsplit("/", 1)[-1])

    def test_adapter_symmetry(self):
        a = br.Adapter(gain=2.0, bias=0.0)
        self.assertEqual(a(0.5, 0.5), 0.0)      # equal pools -> no steering
        self.assertGreater(a(0.8, 0.2), 0.0)    # more left -> positive
        self.assertLess(a(0.2, 0.8), 0.0)


def _angle_tuned_brain():
    """Synthetic 2-neuron brain, no recurrence: goal cue drives left/right pops.

    Neuron 0 (left output) prefers +pi/2, neuron 1 (right output) prefers -pi/2,
    so a goal to the left excites left>right and steering goes positive.
    """
    W = sp.csr_array(np.zeros((2, 2), dtype=np.float32))
    input_map = {"goal": (np.array([0, 1]),
                          np.array([math.pi / 2, -math.pi / 2], dtype=np.float32))}
    output_map = {"left": np.array([0]), "right": np.array([1])}
    return br.Brain(W, input_map, output_map, br.RateParams())


def _settle(controller, heading, goal_bearing, steps=60):
    obs = sim.NeuralObservation(heading=heading, goal_bearing=goal_bearing)
    s = 0.0
    for _ in range(steps):
        s = controller(obs)
    return s


class TestNeuralSteering(unittest.TestCase):
    """Phase 3: cue->brain->steering coupling checks (synthetic graph)."""

    def test_left_right_symmetry(self):
        b = _angle_tuned_brain()
        ctrl = br.NeuralController(b, br.Adapter(gain=4.0, bias=0.0))
        b.reset(); left = _settle(ctrl, 0.0, +0.8)
        b.reset(); right = _settle(ctrl, 0.0, -0.8)
        self.assertGreater(left, 0.0)               # goal left -> steer left
        self.assertAlmostEqual(left, -right, places=5)  # mirror bearing -> mirror steer

    def test_monotonic_in_bearing(self):
        b = _angle_tuned_brain()
        ctrl = br.NeuralController(b, br.Adapter(gain=0.5, bias=0.0))  # keep off the clip
        b.reset(); near = _settle(ctrl, 0.0, 0.2)
        b.reset(); far = _settle(ctrl, 0.0, 0.8)
        b.reset(); zero = _settle(ctrl, 0.0, 0.0)
        self.assertAlmostEqual(zero, 0.0, places=5)
        self.assertLess(near, far)

    def test_adapter_has_no_geometry_shortcut(self):
        # With outputs held fixed, steering must be invariant to any geometry:
        # the only channel from the world to the adapter is the two pooled rates.
        b = _angle_tuned_brain()
        ctrl = br.NeuralController(b, br.Adapter(gain=4.0, bias=0.0))
        b.outputs = lambda: (0.7, 0.2)  # freeze the only legal channel
        steers = {_settle(ctrl, h, g)
                  for h in (-2.0, 0.0, 2.0) for g in (-2.0, 0.0, 1.0, 3.0)}
        self.assertEqual(len(steers), 1)  # geometry changed, steering did not


class TestControls(unittest.TestCase):
    def test_shuffle_preserves_source_stats(self):
        W = sp.csr_array(np.array([[0, -2, 0, 3],
                                   [1, 0, 0, 0],
                                   [0, 4, 0, -5],
                                   [0, 0, 6, 0]], dtype=np.float32))
        C0 = sp.csc_array(W)
        C1 = sp.csc_array(ev.shuffle_connectivity(W, seed=7))
        self.assertEqual(C1.shape, C0.shape)
        for c in range(W.shape[1]):  # per source: same out-degree + weight multiset
            d0 = sorted(C0.data[C0.indptr[c]:C0.indptr[c + 1]])
            d1 = sorted(C1.data[C1.indptr[c]:C1.indptr[c + 1]])
            self.assertEqual(d0, d1)

    def test_cue_withheld_kills_steering(self):
        b = _angle_tuned_brain()
        full = br.NeuralController(b, br.Adapter(gain=4.0, bias=0.0))
        b.reset(); driven = _settle(full, 0.0, 0.8)
        ctrl, reset = ev.cue_withheld_controller(b, br.Adapter(gain=4.0, bias=0.0))
        reset(); withheld = _settle(ctrl, 0.0, 0.8)
        self.assertGreater(driven, 0.0)
        self.assertAlmostEqual(withheld, 0.0, places=5)

    def test_pathway_silenced_collapses_to_bias(self):
        b = _angle_tuned_brain()
        ctrl, reset = ev.pathway_silenced_controller(b, br.Adapter(gain=4.0, bias=0.0))
        reset(); s = _settle(ctrl, 0.0, 0.8)
        self.assertAlmostEqual(s, 0.0, places=6)  # outputs clamped -> only bias

    def test_calibrate_picks_from_declared_grid(self):
        b = _angle_tuned_brain()
        sc = ev.generate_scenarios(8, seed=2)
        gain, bias, grid = ev.calibrate_adapter(sc, ev.neural_factory(b),
                                                gains=(1.0, 4.0), biases=(0.0, 0.05))
        self.assertIn(gain, (1.0, 4.0))
        self.assertIn(bias, (0.0, 0.05))
        self.assertEqual(len(grid), 4)  # full budget reported, not expanded

    def test_load_checkpoint_reads_nested_calibration_artifact(self):
        with tempfile.NamedTemporaryFile(mode="w") as f:
            json.dump({"adapter": {"gain": 32, "bias": -0.05},
                       "rate_params": {"neural_dt": 0.01, "tau": 0.05}}, f)
            f.flush()
            checkpoint = ev.load_checkpoint(f.name)
        self.assertEqual(checkpoint["gain"], 32)
        self.assertEqual(checkpoint["bias"], -0.05)
        self.assertEqual(checkpoint["params"].neural_dt, 0.01)


if __name__ == "__main__":
    unittest.main()
