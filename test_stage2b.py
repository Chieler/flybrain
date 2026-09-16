"""Stage 2b architecture experiment: two cheap observation-boundary baselines.

- WaypointController: evaluator-only solvability floor (may use geometry).
- ObservationStateMachine: observation-only FOLLOW/TURN state machine that must
  detect lateral openings and begin turns *before* becoming trapped, using only
  heading, goal_bearing, ranges, and speed.

These are a separately named experiment; they must not be merged into or replace
the 33/100 Stage 2 result.
"""

import inspect
import math
import unittest

import stage2b
from simulation import MAX_STEERING
from street import (
    StreetCarState, StreetObservation, StreetScenario, initial_layouts,
    make_grid_layout, run_street_episode,
)


class TestRoutePlanning(unittest.TestCase):
    def setUp(self):
        self.layout = make_grid_layout("cross", (0.0,), (0.0,))

    def test_route_turns_at_intersection_when_los_blocked(self):
        # Start west on the y=0 road, target north on the x=0 road: a straight
        # start->target segment crosses the central block, so the road route must
        # pass through the (0,0) intersection.
        route = stage2b.plan_route((-21.0, 0.0), (0.0, 21.0), self.layout)
        self.assertEqual(route[0], (-21.0, 0.0))
        self.assertEqual(route[-1], (0.0, 21.0))
        self.assertIn((0.0, 0.0), route)          # turns at the intersection
        # every hop is axis-aligned (travels along a road, never diagonally)
        for a, b in zip(route, route[1:]):
            self.assertTrue(math.isclose(a[0], b[0]) or math.isclose(a[1], b[1]))

    def test_straight_route_needs_no_intermediate_node(self):
        route = stage2b.plan_route((0.0, -21.0), (0.0, 21.0), self.layout)
        self.assertEqual(route, [(0.0, -21.0), (0.0, 21.0)])


class TestWaypointController(unittest.TestCase):
    def test_arrives_on_a_turn_required_route(self):
        layout = make_grid_layout("cross", (0.0,), (0.0,))
        # Facing north up the x=0 road; target east on the y=0 road -> right turn
        # at the origin is required.
        scenario = StreetScenario(
            "cross", StreetCarState(0.0, -21.0, math.pi / 2, 0.0), 21.0, 0.0)
        ctrl = stage2b.WaypointController(scenario, layout)
        ctrl.reset()
        result = run_street_episode(scenario, layout, ctrl)
        self.assertEqual(result.outcome, "arrival")


class TestEligibleStratum(unittest.TestCase):
    def setUp(self):
        self.layout = make_grid_layout("cross", (0.0,), (0.0,))

    def test_outward_facing_road_end_is_flagged(self):
        # North endpoint of the x=0 road, facing further north (outward).
        self.assertTrue(stage2b.is_outward_road_end(0.0, 21.0, math.pi / 2, self.layout))

    def test_inward_and_perpendicular_starts_are_eligible(self):
        # Same endpoint facing back down the road (inward) -> eligible.
        self.assertFalse(stage2b.is_outward_road_end(0.0, 21.0, -math.pi / 2, self.layout))
        # An intersection start is never a road-end.
        self.assertFalse(stage2b.is_outward_road_end(0.0, 0.0, math.pi / 2, self.layout))

    def test_generated_split_forbids_outward_road_end(self):
        import evaluate_stage2b as ev
        layouts = initial_layouts()
        split = ev.generate_stage2b_split(seed=21, counts={"cross": 5, "regular": 5,
                                                           "asymmetric": 5})
        for s in split:
            self.assertFalse(
                stage2b.is_outward_road_end(s.start.x, s.start.y, s.start.heading,
                                            layouts[s.layout]),
                f"{s.label} is an outward road-end start")


class TestObservationStateMachine(unittest.TestCase):
    def _obs(self, heading, goal_bearing, ranges, speed=1.5):
        return StreetObservation(heading, goal_bearing, speed, ranges)

    def test_is_observation_only(self):
        # No geometry argument: the controller takes exactly (self, obs).
        self.assertEqual(
            list(inspect.signature(stage2b.ObservationStateMachine.__call__).parameters),
            ["self", "obs"])

    def test_straight_corridor_goes_forward(self):
        sm = stage2b.ObservationStateMachine()
        sm.reset()
        # Clear ahead, side walls symmetric at the road half-width, goal dead
        # ahead: near-zero steering and positive acceleration.
        control = sm(self._obs(0.0, 0.0, (3.0, 4.2, 8.0, 4.2, 3.0), 1.0))
        self.assertAlmostEqual(control.steering, 0.0, places=2)
        self.assertGreater(control.acceleration, 0.0)

    def test_arms_then_commits_turn_toward_goal_when_that_side_opens(self):
        sm = stage2b.ObservationStateMachine(turn_delay=1.0)
        sm.reset()
        # Goal is to the left (+90deg) and the left side has opened, but forward
        # is still long: arm the turn without committing yet.
        sm(self._obs(0.0, math.pi / 2, (8.0, 8.0, 8.0, 4.2, 3.0), 1.5))
        self.assertEqual(sm._armed, 1)
        self.assertEqual(sm.mode, stage2b.ObservationStateMachine.FOLLOW)
        # Forward has dropped past turn_delay (intersection entered): commit left.
        control = sm(self._obs(0.0, math.pi / 2, (8.0, 8.0, 6.0, 4.2, 3.0), 1.5))
        self.assertEqual(sm.mode, stage2b.ObservationStateMachine.TURN_LEFT)
        self.assertGreater(control.steering, 0.0)

    def test_does_not_turn_toward_an_opening_that_is_away_from_goal(self):
        sm = stage2b.ObservationStateMachine()
        sm.reset()
        # Left side opens but the goal is straight ahead (err~0): keep following,
        # do not chase the opening.
        control = sm(self._obs(0.0, 0.0, (8.0, 8.0, 8.0, 4.2, 3.0), 1.5))
        self.assertEqual(sm.mode, stage2b.ObservationStateMachine.FOLLOW)

    def test_turns_when_forward_blocked_even_without_goal_offset(self):
        sm = stage2b.ObservationStateMachine()
        sm.reset()
        # Wall dead ahead, goal ahead, only the right side is open: must turn
        # right rather than drive into the wall.
        control = sm(self._obs(0.0, 0.0, (3.0, 3.0, 1.0, 8.0, 8.0), 1.0))
        self.assertEqual(sm.mode, stage2b.ObservationStateMachine.TURN_RIGHT)
        self.assertLess(control.steering, 0.0)

    def test_arrives_on_a_straight_shot(self):
        layout = make_grid_layout("cross", (0.0,), (0.0,))
        scenario = StreetScenario(
            "cross", StreetCarState(0.0, -21.0, math.pi / 2, 0.0), 0.0, 21.0)
        sm = stage2b.ObservationStateMachine()
        sm.reset()
        result = run_street_episode(scenario, layout, sm)
        self.assertEqual(result.outcome, "arrival")


if __name__ == "__main__":
    unittest.main()
