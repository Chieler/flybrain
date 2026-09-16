"""Stage 2b architecture experiment: two cheap observation-boundary baselines.

Purpose (see HANDOFF Stage 2b): localize what the Stage 2 street-navigation
failure actually is, using two controllers scored on the same 90%/80% gate.

- `WaypointController` -- an EVALUATOR-ONLY solvability *witness*. It is
  explicitly allowed to use privileged layout + route + target + pose
  information (declared, like the Stage 1 conventional baseline): it plans a road
  route (Dijkstra) and follows it with pure pursuit, dead-reckoning its own pose
  from the known start using the exact simulator integrator. Its success on a
  scenario witnesses that the vehicle and layout admit a solution there; its
  failure would NOT prove a scenario unsolvable (a different route or controller
  might still succeed).

- `ObservationStateMachine` -- an OBSERVATION-ONLY FOLLOW/TURN state machine. It
  sees exactly `heading, goal_bearing, speed, ranges` (no geometry; the
  `__call__` signature is `(self, obs)`, asserted in `test_stage2b`). It carries
  short-horizon control state (mode, an armed turn, a target heading) but no
  route or topological memory. Unlike the three Stage 2 reactive variants, it
  detects a lateral opening and *begins the turn before becoming trapped*,
  rather than only escaping after forward blockage.

Neither controller is neural; this experiment does not touch the connectome or
the frozen Stage 2 artifacts.
"""

from __future__ import annotations

import heapq
import math

from simulation import MAX_STEERING, PHYSICS_DT, wrap_angle
from street import (
    ARENA_BOUND, CRUISE_SPEED, Control, MAX_ACCEL, MAX_BRAKE, SENSOR_RANGE,
    StreetLayout, _segment_rect_entry,
)

# Road-endpoint waypoints sit this far in from the arena edge (matches the
# evaluator's scenario waypoints).
_EDGE = ARENA_BOUND - 4.0


def is_outward_road_end(x: float, y: float, heading: float,
                        layout: StreetLayout) -> bool:
    """True if the start sits at a road's arena-edge endpoint and faces outward
    (toward the edge).

    Such a car must reverse to reach any interior target but has no lane room to
    U-turn (full-lock turn radius 1.73 sweeps 3.46 laterally in a 6-wide road).
    This is a post-hoc characterization of the 16 waypoint failures on the
    seed-7 dev split, used to define the Stage 2b eligible stratum -- NOT a claim
    that these starts are physically impossible.
    """
    def hclose(a: float, b: float) -> bool:
        return abs(wrap_angle(a - b)) < 1e-6
    for xr in layout.x_roads:
        if math.isclose(x, xr):
            if math.isclose(y, _EDGE) and hclose(heading, math.pi / 2):
                return True
            if math.isclose(y, -_EDGE) and hclose(heading, -math.pi / 2):
                return True
    for yr in layout.y_roads:
        if math.isclose(y, yr):
            if math.isclose(x, _EDGE) and hclose(heading, 0.0):
                return True
            if math.isclose(x, -_EDGE) and hclose(heading, math.pi):
                return True
    return False


def _los_blocked(ax: float, ay: float, bx: float, by: float,
                 layout: StreetLayout) -> bool:
    """True if the straight segment enters any building interior."""
    return any(_segment_rect_entry(ax, ay, bx, by, rect) is not None
               for rect in layout.buildings)


def _road_nodes(layout: StreetLayout) -> list[tuple[float, float]]:
    """Road-centerline nodes: every intersection plus each road's two edge ends."""
    nodes = {(x, y) for x in layout.x_roads for y in layout.y_roads}
    nodes |= {(x, -_EDGE) for x in layout.x_roads}
    nodes |= {(x, _EDGE) for x in layout.x_roads}
    nodes |= {(-_EDGE, y) for y in layout.y_roads}
    nodes |= {(_EDGE, y) for y in layout.y_roads}
    return sorted(nodes)


def plan_route(start: tuple[float, float], target: tuple[float, float],
               layout: StreetLayout) -> list[tuple[float, float]]:
    """Shortest road route start -> target as a waypoint polyline (Dijkstra).

    Nodes are road-centerline points; two nodes are connected only by an
    axis-aligned, collision-free segment, so every hop runs along a road.
    Evaluator-only: uses layout geometry by design.
    """
    nodes = set(_road_nodes(layout)) | {start, target}
    nodes = sorted(nodes)
    adj: dict[tuple[float, float], list[tuple[float, float, float]]] = {n: [] for n in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if (math.isclose(a[0], b[0]) or math.isclose(a[1], b[1])) and \
                    not _los_blocked(a[0], a[1], b[0], b[1], layout):
                w = math.dist(a, b)
                adj[a].append((w, b[0], b[1]))
                adj[b].append((w, a[0], a[1]))
    # Dijkstra
    dist = {n: math.inf for n in nodes}
    prev: dict[tuple[float, float], tuple[float, float] | None] = {n: None for n in nodes}
    dist[start] = 0.0
    pq = [(0.0, start[0], start[1])]
    while pq:
        d, x, y = heapq.heappop(pq)
        node = (x, y)
        if d > dist[node]:
            continue
        if node == target:
            break
        for w, nx, ny in adj[node]:
            nd = d + w
            if nd < dist[(nx, ny)]:
                dist[(nx, ny)] = nd
                prev[(nx, ny)] = node
                heapq.heappush(pq, (nd, nx, ny))
    if dist[target] == math.inf:
        return [start, target]  # disconnected; let pure pursuit try directly
    route, node = [], target
    while node is not None:
        route.append(node)
        node = prev[node]
    route.reverse()
    return route


def _accel_to(desired_speed: float, speed: float) -> float:
    return max(-MAX_BRAKE, min(MAX_ACCEL, 2.0 * (desired_speed - speed)))


def _proj_param(p, a, b) -> float:
    """Clamped projection parameter of p onto segment a->b (0 at a, 1 at b)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return 0.0
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom
    return max(0.0, min(1.0, t))


class WaypointController:
    """Evaluator-only pure-pursuit route follower (solvability witness).

    Follows the planned polyline with a fixed look-ahead ("carrot") point and
    slows into turns, so its arcs stay inside the road corridor rather than
    cutting building corners.
    """

    def __init__(self, scenario, layout: StreetLayout, lookahead: float = 3.0,
                 steer_gain: float = 2.5, turn_speed: float = 1.0,
                 turn_slow_dist: float = 4.0):
        self.start = (scenario.start.x, scenario.start.y)
        self.start_heading = scenario.start.heading
        self.route = plan_route(self.start, (scenario.target_x, scenario.target_y),
                                layout)
        self.lookahead = lookahead
        self.steer_gain = steer_gain
        self.turn_speed = turn_speed
        self.turn_slow_dist = turn_slow_dist
        self.reset()

    def reset(self) -> None:
        self._pos = self.start
        self._prev_heading = self.start_heading
        self._seg = 0
        self._first = True

    def _advance_segment(self) -> None:
        # Move to the segment the car has progressed onto.
        while self._seg < len(self.route) - 2 and \
                _proj_param(self._pos, self.route[self._seg],
                            self.route[self._seg + 1]) >= 1.0:
            self._seg += 1

    def _carrot(self) -> tuple[float, float]:
        # Point a look-ahead distance beyond the projection, walked along the
        # remaining polyline.
        seg = self._seg
        t = _proj_param(self._pos, self.route[seg], self.route[seg + 1])
        a, b = self.route[seg], self.route[seg + 1]
        base = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        remaining = self.lookahead
        cur = base
        k = seg
        while k < len(self.route) - 1:
            nxt = self.route[k + 1]
            d = math.dist(cur, nxt)
            if d >= remaining:
                f = remaining / d if d else 1.0
                return (cur[0] + (nxt[0] - cur[0]) * f, cur[1] + (nxt[1] - cur[1]) * f)
            remaining -= d
            cur = nxt
            k += 1
        return self.route[-1]

    def _turn_ahead(self) -> bool:
        # True if an interior route vertex within turn_slow_dist bends the path.
        for k in range(self._seg + 1, len(self.route) - 1):
            v = self.route[k]
            if math.dist(self._pos, v) > self.turn_slow_dist:
                break
            a, c = self.route[k - 1], self.route[k + 1]
            # collinear vertices do not bend
            if not (math.isclose(a[0], c[0]) or math.isclose(a[1], c[1])):
                return True
        return False

    def __call__(self, obs) -> Control:
        if not self._first:
            # Dead-reckon with the exact simulator update: the displacement used
            # this step's post-acceleration speed (now obs.speed) and the prior
            # heading.
            self._pos = (self._pos[0] + obs.speed * math.cos(self._prev_heading) * PHYSICS_DT,
                         self._pos[1] + obs.speed * math.sin(self._prev_heading) * PHYSICS_DT)
        self._first = False
        self._advance_segment()
        cx, cy = self._carrot()
        err = wrap_angle(math.atan2(cy - self._pos[1], cx - self._pos[0]) - obs.heading)
        steering = max(-MAX_STEERING, min(MAX_STEERING, self.steer_gain * err))
        speed = self.turn_speed if (self._turn_ahead() or abs(err) > math.radians(20.0)) \
            else CRUISE_SPEED
        self._prev_heading = obs.heading
        return Control(steering, _accel_to(speed, obs.speed))


class ObservationStateMachine:
    """Observation-only FOLLOW / TURN state machine.

    Sees exactly heading, goal_bearing, speed, ranges. It follows the current
    road corridor (centering on the side rays), and at an intersection -- where a
    side ray opens past a cross street -- commits a 90 degree turn onto the road
    that heads toward the goal, *before* becoming trapped. A committed turn is
    delayed until the intersection is entered (`turn_delay` forward advance) so
    the fixed-radius arc lands centered in the crossing lane instead of clipping
    the inner corner.

    Thresholds default from the arena geometry (road width 6 -> half-width 3,
    SENSOR_RANGE 8); they are exposed as constructor arguments so the design can
    be tuned on the Stage 2b *training* split only (never the gate split).
    """

    FOLLOW, TURN_LEFT, TURN_RIGHT = "FOLLOW", "TURN_LEFT", "TURN_RIGHT"

    # Defaults are the best configuration from a 144-point sweep on the Stage 2b
    # TRAINING split (runs/stage2b/sm_train_split.json); the gate split was never
    # inspected during tuning.
    def __init__(self, opening: float = 5.0, forward_block: float = 3.0,
                 align: float = math.radians(35.0), turn_speed: float = 1.0,
                 heading_tol: float = math.radians(10.0), corridor_clear: float = 5.0,
                 kc1: float = 0.06, kc2: float = 0.03, k_turn: float = 3.0,
                 cruise: float = CRUISE_SPEED, forward_slow: float = 6.0,
                 speed_floor: float = 0.5, trigger45: bool = False,
                 turn_delay: float = 0.0):
        self.opening = opening
        self.forward_block = forward_block
        self.align = align
        self.turn_speed = turn_speed
        self.heading_tol = heading_tol
        self.corridor_clear = corridor_clear
        self.kc1 = kc1
        self.kc2 = kc2
        self.k_turn = k_turn
        self.cruise = cruise
        self.forward_slow = forward_slow
        self.speed_floor = speed_floor
        self.trigger45 = trigger45          # gate openings on the 45deg ray
        self.turn_delay = turn_delay        # forward advance before committing
        self.reset()

    def reset(self) -> None:
        self.mode = self.FOLLOW
        self._target_heading = None
        self._armed = 0                     # pending turn side, awaiting entry
        self._arm_forward = None            # forward range when the turn armed

    def __call__(self, obs) -> Control:
        l90, l45, forward, r45, r90 = obs.ranges
        err = wrap_angle(obs.goal_bearing - obs.heading)
        left_side = l45 if self.trigger45 else l90
        right_side = r45 if self.trigger45 else r90
        left_open = left_side > self.opening
        right_open = right_side > self.opening
        forward_blocked = forward < self.forward_block

        # --- executing a committed turn ---
        if self.mode != self.FOLLOW:
            th_err = wrap_angle(self._target_heading - obs.heading)
            if abs(th_err) > self.heading_tol or forward < self.corridor_clear:
                steering = max(-MAX_STEERING, min(MAX_STEERING, self.k_turn * th_err))
                return Control(steering, _accel_to(self.turn_speed, obs.speed))
            self.mode = self.FOLLOW          # aligned + corridor acquired

        # --- choose a turn side (goalward opening, or forced by a wall) ---
        turn_dir = 0
        if abs(err) > self.align:
            if err > 0 and left_open:
                turn_dir = 1
            elif err < 0 and right_open:
                turn_dir = -1
        if turn_dir == 0 and forward_blocked:
            if left_open and (err >= 0 or not right_open):
                turn_dir = 1
            elif right_open:
                turn_dir = -1
            elif left_open:
                turn_dir = 1
            else:
                turn_dir = 1 if err >= 0 else -1     # dead-end: rotate goalward

        # --- arm, then commit once the intersection is entered ---
        if turn_dir != 0:
            if self._armed != turn_dir:
                self._armed = turn_dir
                self._arm_forward = forward
            entered = (self._arm_forward - forward) >= self.turn_delay
            if forward_blocked or entered:
                self.mode = self.TURN_LEFT if turn_dir > 0 else self.TURN_RIGHT
                self._target_heading = wrap_angle(obs.heading + turn_dir * math.pi / 2)
                self._armed = 0
                return Control(turn_dir * MAX_STEERING,
                               _accel_to(self.turn_speed, obs.speed))
        else:
            self._armed = 0

        # --- FOLLOW: center on the corridor (zeroes offset and heading skew) ---
        steering = self.kc1 * (l90 - r90) + self.kc2 * (l45 - r45)
        steering = max(-MAX_STEERING, min(MAX_STEERING, steering))
        speed = self.cruise * max(self.speed_floor, min(1.0, forward / self.forward_slow))
        if self._armed != 0:                 # approaching a turn: ease off
            speed = min(speed, self.turn_speed)
        return Control(steering, _accel_to(speed, obs.speed))
