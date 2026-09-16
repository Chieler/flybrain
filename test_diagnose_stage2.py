"""Checks for the Stage 2 per-episode steering diagnostic.

These cover only the pure-geometry, evaluator-only classification helpers, so
they run without loading the real connectome.
"""

import math
import unittest

from diagnose_stage2 import (
    classify_scenario, line_of_sight_blocked, TURN_ERROR_THRESHOLD,
)
from street import StreetCarState, StreetScenario, make_grid_layout


class TestLineOfSight(unittest.TestCase):
    def setUp(self):
        # A single central building spanning roughly [-3, 3] on both axes
        # (cross layout: one road at x=0 and y=0, road width 6).
        self.layout = make_grid_layout("cross", (0.0,), (0.0,))

    def test_segment_through_building_is_blocked(self):
        # Diagonal across the arena passes through the central block.
        self.assertTrue(line_of_sight_blocked(-20.0, -20.0, 20.0, 20.0, self.layout))

    def test_segment_along_open_road_is_clear(self):
        # Straight down the x=0 road never enters a building.
        self.assertFalse(line_of_sight_blocked(0.0, -20.0, 0.0, 20.0, self.layout))


class TestScenarioClassification(unittest.TestCase):
    def setUp(self):
        self.layout = make_grid_layout("cross", (0.0,), (0.0,))

    def _scenario(self, sx, sy, heading, tx, ty):
        return StreetScenario("cross", StreetCarState(sx, sy, heading, 0.0), tx, ty)

    def test_clear_straight_shot_needs_no_turn(self):
        # Facing straight up the open road toward a target dead ahead.
        s = self._scenario(0.0, -20.0, math.pi / 2, 0.0, 20.0)
        cls = classify_scenario(s, self.layout)
        self.assertFalse(cls["line_of_sight_blocked"])
        self.assertFalse(cls["turn_required"])
        self.assertAlmostEqual(cls["initial_bearing_error"], 0.0, places=6)

    def test_blocked_line_of_sight_forces_turn(self):
        # Target across the central building: LOS blocked -> turn required.
        s = self._scenario(-20.0, -20.0, math.radians(45.0), 20.0, 20.0)
        cls = classify_scenario(s, self.layout)
        self.assertTrue(cls["line_of_sight_blocked"])
        self.assertTrue(cls["turn_required"])

    def test_large_bearing_error_forces_turn_even_with_clear_los(self):
        # Clear road ahead but the car is pointed the wrong way.
        s = self._scenario(0.0, -20.0, -math.pi / 2, 0.0, 20.0)
        cls = classify_scenario(s, self.layout)
        self.assertFalse(cls["line_of_sight_blocked"])
        self.assertGreater(abs(cls["initial_bearing_error"]), TURN_ERROR_THRESHOLD)
        self.assertTrue(cls["turn_required"])


if __name__ == "__main__":
    unittest.main()
