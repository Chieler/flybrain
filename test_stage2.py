# test_stage2.py
import math
import unittest

import street as st
import brain as br
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


class TestStage2NeuralController(unittest.TestCase):
    def test_bridge_passes_only_rates_ranges_and_speed_to_adapter(self):
        brain = __import__("test_stage1")._angle_tuned_brain()
        adapter = br.Stage2Adapter(1.0, 0.0, 0.0, 4.0)
        controller = br.Stage2NeuralController(brain, adapter)
        obs = st.StreetObservation(0.0, 0.8, 1.0, (8.0,) * 5)
        control = controller(obs)
        self.assertGreater(control.steering, 0.0)
        self.assertTrue(-st.MAX_BRAKE <= control.acceleration <= st.MAX_ACCEL)


if __name__ == "__main__":
    unittest.main()
