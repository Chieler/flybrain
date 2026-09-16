# test_stage2.py
import math
import unittest

import street as st
import brain as br
import evaluate_stage2 as ev2
from simulation import MAX_STEERING


class TestStreetGeometry(unittest.TestCase):
    def test_initial_layouts_are_fixed_and_connected_grids(self):
        layouts = st.initial_layouts()
        self.assertEqual(set(layouts), {"cross", "regular", "asymmetric"})
        self.assertEqual(layouts["cross"].x_roads, (0.0,))
        self.assertEqual(layouts["regular"].y_roads, (-10.0, 10.0))
        self.assertEqual(layouts["asymmetric"].x_roads, (-12.0, 7.0))

    def test_forward_sensor_hits_building(self):
        layout = st.make_grid_layout("test", (0.0,), (0.0,), road_width=6.0)
        state = st.StreetCarState(0.0, 0.0, math.pi / 4, 0.0)
        obs = st.observe_street(state, st.StreetScenario("test", state, 0.0, 10.0), layout)
        self.assertAlmostEqual(obs.ranges[2], 3.0 * math.sqrt(2), places=5)

    def test_sensor_range_is_clipped(self):
        layout = st.make_grid_layout("test", (0.0,), (0.0,), road_width=6.0)
        state = st.StreetCarState(0.0, 0.0, 0.0, 0.0)
        obs = st.observe_street(state, st.StreetScenario("test", state, 0.0, 10.0), layout)
        self.assertEqual(obs.ranges[2], st.SENSOR_RANGE)


class TestStreetEpisode(unittest.TestCase):
    def test_acceleration_and_braking_are_bounded(self):
        state = st.StreetCarState(0.0, 0.0, 0.0, 1.0)
        faster = st.street_step(state, st.Control(0.0, 100.0), 0.5)
        slower = st.street_step(state, st.Control(0.0, -100.0), 0.5)
        self.assertEqual(faster.speed, 2.0)
        self.assertEqual(slower.speed, 0.0)

    def test_swept_building_collision_terminates(self):
        layout = st.make_grid_layout("test", (0.0,), (0.0,), road_width=6.0)
        start = st.StreetCarState(0.0, 0.0, math.pi / 4, 3.0)
        scenario = st.StreetScenario("test", start, 0.0, 20.0)
        result = st.run_street_episode(scenario, layout,
            lambda obs: st.Control(0.0, 0.0), dt=2.0)
        self.assertEqual(result.outcome, "collision")
        self.assertLess(result.elapsed_time, 2.0)

    def test_target_hit_wins_when_first(self):
        layout = st.make_grid_layout("test", (0.0,), (0.0,), road_width=6.0)
        start = st.StreetCarState(0.0, -5.0, math.pi / 2, 2.0)
        scenario = st.StreetScenario("test", start, 0.0, -4.0, target_radius=0.1)
        result = st.run_street_episode(scenario, layout,
            lambda obs: st.Control(0.0, 0.0), dt=1.0, record=True)
        self.assertEqual(result.outcome, "arrival")
        self.assertAlmostEqual(result.path_length, 0.9)
        self.assertEqual(len(result.trajectory[-1]), 4)


class TestStage2Adapter(unittest.TestCase):
    def test_equal_openings_preserve_neural_steering(self):
        adapter = br.Stage2Adapter(brain_gain=2.0, bias=0.0,
                                   avoidance_gain=1.0, brake_distance=4.0)
        control = adapter(0.7, 0.2, (8.0, 8.0, 8.0, 8.0, 8.0), 1.0)
        self.assertAlmostEqual(control.steering, MAX_STEERING)

    def test_adapter_steers_toward_open_side(self):
        adapter = br.Stage2Adapter(brain_gain=0.0, bias=0.0,
                                   avoidance_gain=0.5, brake_distance=4.0)
        control = adapter(0.0, 0.0, (8.0, 8.0, 8.0, 1.0, 1.0), 1.0)
        self.assertGreater(control.steering, 0.0)

    def test_near_forward_wall_commands_braking(self):
        adapter = br.Stage2Adapter(brain_gain=0.0, bias=0.0,
                                   avoidance_gain=0.0, brake_distance=4.0)
        control = adapter(0.0, 0.0, (8.0, 8.0, 0.5, 8.0, 8.0), 2.0)
        self.assertLess(control.acceleration, 0.0)

    def test_stage2_adapter_has_no_geometry_argument(self):
        import inspect
        self.assertEqual(list(inspect.signature(br.Stage2Adapter.__call__).parameters),
                         ["self", "left", "right", "ranges", "speed"])


class TestStage2Recovery(unittest.TestCase):
    """Opt-in commit-and-hold recovery behavior (HANDOFF Stage 2 step #2)."""

    def _adapter(self, **kw):
        params = dict(brain_gain=2.0, bias=0.0, avoidance_gain=0.0,
                      brake_distance=4.0, commit_distance=2.0,
                      release_distance=4.0, recover_speed=1.0)
        params.update(kw)
        return br.Stage2Adapter(**params)

    def test_disabled_by_default_reproduces_plain_behavior(self):
        # No commit params -> recovery off -> near-wall response is plain braking,
        # never a forced max turn.
        plain = br.Stage2Adapter(brain_gain=0.0, bias=0.0, avoidance_gain=0.0,
                                 brake_distance=4.0)
        control = plain(0.0, 0.0, (8.0, 8.0, 0.5, 8.0, 8.0), 2.0)
        self.assertEqual(plain._committed, 0)
        self.assertAlmostEqual(control.steering, 0.0)
        self.assertLess(control.acceleration, 0.0)

    def test_commits_toward_more_open_side_below_threshold(self):
        adapter = self._adapter()
        # Forward blocked (0.5 < commit_distance 2.0); left rays open, right closed.
        control = adapter(0.0, 0.0, (8.0, 8.0, 0.5, 1.0, 1.0), 1.0)
        self.assertEqual(adapter._committed, 1)  # committed left
        self.assertAlmostEqual(control.steering, MAX_STEERING)

    def test_hold_overrides_goal_pull_until_lane_reopens(self):
        adapter = self._adapter()
        adapter(0.0, 0.0, (8.0, 8.0, 0.5, 1.0, 1.0), 1.0)  # commit left
        # Strong rightward neural push, but forward (3.0) is still inside the
        # hysteresis band (commit 2.0 < 3.0 < release 4.0): the hold wins.
        held = adapter(0.0, 1.0, (8.0, 8.0, 3.0, 8.0, 8.0), 1.0)
        self.assertEqual(adapter._committed, 1)
        self.assertAlmostEqual(held.steering, MAX_STEERING)

    def test_releases_only_past_release_distance(self):
        adapter = self._adapter()
        adapter(0.0, 0.0, (8.0, 8.0, 0.5, 1.0, 1.0), 1.0)  # commit left
        adapter(0.0, 1.0, (8.0, 8.0, 3.0, 8.0, 8.0), 1.0)  # still held
        # Forward 5.0 > release 4.0 -> release; steering follows the neural push
        # (left<right -> negative steer).
        freed = adapter(0.0, 1.0, (8.0, 8.0, 5.0, 8.0, 8.0), 1.0)
        self.assertEqual(adapter._committed, 0)
        self.assertLess(freed.steering, 0.0)

    def test_reset_clears_commit_state(self):
        adapter = self._adapter()
        adapter(0.0, 0.0, (8.0, 8.0, 0.5, 1.0, 1.0), 1.0)  # commit
        self.assertEqual(adapter._committed, 1)
        adapter.reset()
        self.assertEqual(adapter._committed, 0)

    def test_neural_controller_reset_clears_adapter_state(self):
        brain = __import__("test_stage1")._angle_tuned_brain()
        adapter = self._adapter()
        controller = br.Stage2NeuralController(brain, adapter)
        adapter._committed = 1
        controller.reset()
        self.assertEqual(adapter._committed, 0)


class TestStage2GoalAwareRecovery(unittest.TestCase):
    """Goal-aware commit (HANDOFF Stage 2 step #4): when trapped, commit toward
    the side the brain/compass already wants (sign of the neural steer term),
    not merely the locally-more-open side. Uses only the pooled rates already
    passed in -- no geometry argument."""

    def _adapter(self, **kw):
        params = dict(brain_gain=2.0, bias=0.0, avoidance_gain=0.0,
                      brake_distance=4.0, commit_distance=2.0,
                      release_distance=4.0, recover_speed=1.0, goal_clearance=0.3)
        params.update(kw)
        return br.Stage2Adapter(**params)

    def test_commits_toward_goal_side_even_when_other_side_more_open(self):
        # Brain wants left (left rate > right rate). Left lane is open enough
        # (0.375 >= goal_clearance 0.3) though the RIGHT side is more open.
        adapter = self._adapter()
        control = adapter(1.0, 0.0, (3.0, 3.0, 0.5, 8.0, 8.0), 1.0)
        self.assertEqual(adapter._committed, 1)          # toward goal (left)
        self.assertAlmostEqual(control.steering, MAX_STEERING)

    def test_falls_back_to_open_side_when_goal_side_is_blocked(self):
        # Brain wants left, but the left side is walled (0.0625 < 0.3): escape
        # toward the genuinely more-open right side instead.
        adapter = self._adapter()
        adapter(1.0, 0.0, (0.5, 0.5, 0.5, 8.0, 8.0), 1.0)
        self.assertEqual(adapter._committed, -1)

    def test_goal_direction_follows_sign_of_neural_term(self):
        # Brain wants right (right rate > left rate); right side open enough.
        adapter = self._adapter()
        adapter(0.0, 1.0, (8.0, 8.0, 0.5, 3.0, 3.0), 1.0)
        self.assertEqual(adapter._committed, -1)         # toward goal (right)

    def test_disabled_goal_clearance_reproduces_pure_openness_commit(self):
        # goal_clearance == 0 -> opt-in off -> commit toward the more-open side
        # (right), ignoring the brain's leftward preference.
        adapter = self._adapter(goal_clearance=0.0)
        adapter(1.0, 0.0, (3.0, 3.0, 0.5, 8.0, 8.0), 1.0)
        self.assertEqual(adapter._committed, -1)


class TestStage2NeuralController(unittest.TestCase):
    def test_bridge_passes_only_rates_ranges_and_speed_to_adapter(self):
        brain = __import__("test_stage1")._angle_tuned_brain()
        adapter = br.Stage2Adapter(1.0, 0.0, 0.0, 4.0)
        controller = br.Stage2NeuralController(brain, adapter)
        obs = st.StreetObservation(0.0, 0.8, 1.0, (8.0,) * 5)
        control = controller(obs)
        self.assertGreater(control.steering, 0.0)
        self.assertTrue(-st.MAX_BRAKE <= control.acceleration <= st.MAX_ACCEL)


class TestStage2Evaluation(unittest.TestCase):
    def test_frozen_split_is_reproducible_and_disjoint(self):
        a_train, a_test = ev2.generate_scenario_splits(seed=2)
        b_train, b_test = ev2.generate_scenario_splits(seed=2)
        self.assertEqual([ev2.scenario_to_dict(s) for s in a_train],
                         [ev2.scenario_to_dict(s) for s in b_train])
        self.assertEqual(len(a_train), 24)
        self.assertEqual(len(a_test), 100)
        self.assertTrue(set(s.label for s in a_train).isdisjoint(s.label for s in a_test))
        key = lambda s: (s.layout, s.start.x, s.start.y, s.target_x, s.target_y)
        self.assertTrue({key(s) for s in a_train}.isdisjoint(key(s) for s in a_test))
        self.assertEqual([sum(s.layout == name for s in a_train)
                          for name in ("cross", "regular", "asymmetric")], [8, 8, 8])
        self.assertEqual([sum(s.layout == name for s in a_test)
                          for name in ("cross", "regular", "asymmetric")], [12, 44, 44])

    def test_dev_split_is_fresh_relative_to_prior_sets(self):
        train, heldout = ev2.generate_scenario_splits(seed=2)
        dev = ev2.generate_dev_split(seed=7, exclude=train + heldout)
        self.assertEqual(len(dev), 100)
        self.assertEqual([sum(s.layout == name for s in dev)
                          for name in ("cross", "regular", "asymmetric")], [12, 44, 44])
        # No dev scenario is identical (route + heading) to any prior scenario.
        prior = {ev2._scenario_key(s) for s in train + heldout}
        self.assertTrue({ev2._scenario_key(s) for s in dev}.isdisjoint(prior))
        # regular/asymmetric routes are fully fresh; only cross reuses routes.
        prior_routes = {ev2._route_key(s) for s in train + heldout}
        reused = [s for s in dev if ev2._route_key(s) in prior_routes]
        self.assertTrue(all(s.layout == "cross" for s in reused))

    def test_summary_reports_every_outcome(self):
        layout = st.make_grid_layout("test", (0.0,), (0.0,))
        arrived = st.StreetScenario("test", st.StreetCarState(0, 0, 0, 0), 0, 0)
        summary = ev2.evaluate_controller([arrived], {"test": layout},
            lambda obs: st.Control(0.0, 0.0))
        self.assertEqual(summary.as_dict(), {
            "trials": 1, "arrivals": 1, "collisions": 0, "timeouts": 0,
            "arrival_rate": 1.0, "mean_arrival_time": 0.0,
            "mean_route_length": 0.0,
        })

    def test_calibration_exhausts_declared_twelve_candidates(self):
        train, _ = ev2.generate_scenario_splits(seed=2)
        made = []
        def factory(avoidance_gain, brake_distance):
            made.append((avoidance_gain, brake_distance))
            return (lambda obs: st.Control(0.0, 0.0)), None
        gain, distance, grid = ev2.calibrate_stage2_adapter(train[:1], factory)
        self.assertEqual(len(grid), 12)
        self.assertEqual(len(made), 12)
        self.assertIn(gain, ev2.AVOIDANCE_GAINS)
        self.assertIn(distance, ev2.BRAKE_DISTANCES)


class TestStage2Controls(unittest.TestCase):
    def test_factory_changes_only_declared_stage2_parameters(self):
        brain = __import__("test_stage1")._angle_tuned_brain()
        source = {"gain": 32.0, "bias": -0.05}
        controller, _ = ev2.stage2_factory(brain, source)(0.5, 4.0)
        self.assertEqual(controller.adapter.brain_gain, 32.0)
        self.assertEqual(controller.adapter.bias, -0.05)
        self.assertEqual(controller.adapter.avoidance_gain, 0.5)
        self.assertEqual(controller.adapter.brake_distance, 4.0)

    def test_sensor_only_ignores_neural_rates(self):
        adapter = br.Stage2Adapter(32.0, -0.05, 0.5, 4.0)
        control = ev2.sensor_only_controller(adapter)(
            st.StreetObservation(2.0, -2.0, 1.0, (8.0,) * 5))
        expected = br.Stage2Adapter(0.0, 0.0, 0.5, 4.0)(
            0.0, 0.0, (8.0,) * 5, 1.0)
        self.assertEqual(control, expected)

    def test_results_have_separate_adapter_and_neural_controls(self):
        expected = {"neural", "direct_compass", "sensor_only",
                    "neural_no_sensors", "goal_cue_withheld", "pfl3_silenced"}
        self.assertEqual(set(ev2.CONTROL_NAMES), expected)

    def test_checkpoint_records_exact_sources_and_learned_parameters(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name
                     for name in ("stage1.json", "training.json", "heldout.json")]
            for path in paths:
                path.write_text("{}")
            output = Path(directory) / "checkpoint.json"
            ev2.save_stage2_checkpoint(str(output), *(str(path) for path in paths),
                                       0.5, 4.0, [])
            checkpoint = json.loads(output.read_text())
            self.assertEqual(checkpoint["learned_parameters"],
                             ["avoidance_gain", "brake_distance"])
            self.assertEqual(checkpoint["stage1_checkpoint_sha256"], ev2.sha256(str(paths[0])))
            self.assertEqual(checkpoint["training_scenarios_sha256"], ev2.sha256(str(paths[1])))
            self.assertEqual(checkpoint["heldout_scenarios_sha256"], ev2.sha256(str(paths[2])))


class TestStage2Viewer(unittest.TestCase):
    def test_building_rect_has_positive_screen_size(self):
        import viewer
        rect = viewer._rect_to_screen(st.Rect(-5.0, -4.0, 2.0, 3.0))
        self.assertGreater(rect.width, 0)
        self.assertGreater(rect.height, 0)


if __name__ == "__main__":
    unittest.main()
