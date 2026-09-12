# street.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from simulation import (
    CAR_RADIUS, MAX_STEERING, PHYSICS_DT, TARGET_RADIUS,
    _earliest_hit_t, _lerp_state, wrap_angle,
)

ARENA_BOUND = 25.0
ROAD_WIDTH = 6.0
SENSOR_RANGE = 8.0
SENSOR_ANGLES = (math.pi / 2, math.pi / 4, 0.0, -math.pi / 4, -math.pi / 2)
CRUISE_SPEED = 2.0
MAX_SPEED = 3.0
MAX_ACCEL = 2.0
MAX_BRAKE = 4.0
WHEELBASE = 1.0
MAX_SIM_TIME = 45.0


@dataclass(frozen=True)
class Rect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def expanded(self, amount: float) -> "Rect":
        return Rect(self.min_x - amount, self.min_y - amount,
                    self.max_x + amount, self.max_y + amount)


@dataclass(frozen=True)
class StreetLayout:
    name: str
    x_roads: tuple[float, ...]
    y_roads: tuple[float, ...]
    buildings: tuple[Rect, ...]


@dataclass(frozen=True)
class StreetCarState:
    x: float
    y: float
    heading: float
    speed: float


@dataclass(frozen=True)
class StreetScenario:
    layout: str
    start: StreetCarState
    target_x: float
    target_y: float
    target_radius: float = TARGET_RADIUS
    timeout: float = MAX_SIM_TIME
    label: str = ""


@dataclass(frozen=True)
class StreetObservation:
    heading: float
    goal_bearing: float
    speed: float
    ranges: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class Control:
    steering: float
    acceleration: float


@dataclass
class StreetEpisodeResult:
    outcome: str
    elapsed_time: float
    path_length: float
    trajectory: list[tuple[float, float, float, float]] | None = field(default=None)


StreetController = Callable[[StreetObservation], Control]


def make_grid_layout(name: str, x_roads: tuple[float, ...],
                     y_roads: tuple[float, ...], road_width: float = ROAD_WIDTH) -> StreetLayout:
    half = road_width / 2
    x_edges = (-ARENA_BOUND,) + tuple(v for x in x_roads for v in (x - half, x + half)) + (ARENA_BOUND,)
    y_edges = (-ARENA_BOUND,) + tuple(v for y in y_roads for v in (y - half, y + half)) + (ARENA_BOUND,)
    buildings = tuple(
        Rect(x_edges[i], y_edges[j], x_edges[i + 1], y_edges[j + 1])
        for i in range(0, len(x_edges) - 1, 2)
        for j in range(0, len(y_edges) - 1, 2)
    )
    return StreetLayout(name, x_roads, y_roads, buildings)


def initial_layouts() -> dict[str, StreetLayout]:
    return {
        "cross": make_grid_layout("cross", (0.0,), (0.0,)),
        "regular": make_grid_layout("regular", (-10.0, 10.0), (-10.0, 10.0)),
        "asymmetric": make_grid_layout("asymmetric", (-12.0, 7.0), (-8.0, 11.0)),
    }


def _segment_rect_entry(ax: float, ay: float, bx: float, by: float,
                        rect: Rect) -> float | None:
    lo, hi = 0.0, 1.0
    for a, d, r0, r1 in ((ax, bx - ax, rect.min_x, rect.max_x),
                         (ay, by - ay, rect.min_y, rect.max_y)):
        if d == 0.0:
            if not r0 <= a <= r1:
                return None
            continue
        t0, t1 = (r0 - a) / d, (r1 - a) / d
        if t0 > t1:
            t0, t1 = t1, t0
        lo, hi = max(lo, t0), min(hi, t1)
        if lo > hi:
            return None
    return lo if 0.0 <= lo <= 1.0 else None


def _ray_distance(x: float, y: float, angle: float, layout: StreetLayout) -> float:
    bx = x + SENSOR_RANGE * math.cos(angle)
    by = y + SENSOR_RANGE * math.sin(angle)
    hits = [1.0]
    hits += [t for rect in layout.buildings
             if (t := _segment_rect_entry(x, y, bx, by, rect)) is not None]
    boundary = Rect(-ARENA_BOUND, -ARENA_BOUND, ARENA_BOUND, ARENA_BOUND)
    for a, d, low, high in ((x, bx - x, boundary.min_x, boundary.max_x),
                            (y, by - y, boundary.min_y, boundary.max_y)):
        if d > 0:
            hits.append((high - a) / d)
        elif d < 0:
            hits.append((low - a) / d)
    return SENSOR_RANGE * min(t for t in hits if 0.0 <= t <= 1.0)


def observe_street(state: StreetCarState, scenario: StreetScenario,
                   layout: StreetLayout) -> StreetObservation:
    bearing = math.atan2(scenario.target_y - state.y, scenario.target_x - state.x)
    ranges = tuple(_ray_distance(state.x, state.y, state.heading + offset, layout)
                   for offset in SENSOR_ANGLES)
    return StreetObservation(state.heading, bearing, state.speed, ranges)


def street_step(state: StreetCarState, control: Control, dt: float) -> StreetCarState:
    steering = max(-MAX_STEERING, min(MAX_STEERING, control.steering))
    acceleration = max(-MAX_BRAKE, min(MAX_ACCEL, control.acceleration))
    speed = max(0.0, min(MAX_SPEED, state.speed + acceleration * dt))
    x = state.x + speed * math.cos(state.heading) * dt
    y = state.y + speed * math.sin(state.heading) * dt
    heading = wrap_angle(state.heading + speed / WHEELBASE * math.tan(steering) * dt)
    return StreetCarState(x, y, heading, speed)


def _collision_t(a: StreetCarState, b: StreetCarState,
                 layout: StreetLayout) -> float | None:
    hits = [t for rect in layout.buildings
            if (t := _segment_rect_entry(a.x, a.y, b.x, b.y,
                                         rect.expanded(CAR_RADIUS))) is not None]
    safe = ARENA_BOUND - CAR_RADIUS
    if abs(a.x) >= safe or abs(a.y) >= safe:
        hits.append(0.0)
    for p, q in ((a.x, b.x), (a.y, b.y)):
        if q > safe:
            hits.append((safe - p) / (q - p))
        elif q < -safe:
            hits.append((-safe - p) / (q - p))
    valid = [t for t in hits if 0.0 <= t <= 1.0]
    return min(valid) if valid else None


def run_street_episode(scenario: StreetScenario, layout: StreetLayout,
                       controller: StreetController, dt: float = PHYSICS_DT,
                       record: bool = False) -> StreetEpisodeResult:
    state, elapsed, distance = scenario.start, 0.0, 0.0
    trajectory = [(state.x, state.y, state.heading, state.speed)] if record else None
    if _earliest_hit_t(state.x, state.y, state.x, state.y,
                       scenario.target_x, scenario.target_y, scenario.target_radius) == 0.0:
        return StreetEpisodeResult("arrival", 0.0, 0.0, trajectory)
    cap = min(scenario.timeout, MAX_SIM_TIME)
    while elapsed < cap:
        step_dt = min(dt, cap - elapsed)
        nxt = street_step(state, controller(observe_street(state, scenario, layout)), step_dt)
        arrival_t = _earliest_hit_t(state.x, state.y, nxt.x, nxt.y,
                                    scenario.target_x, scenario.target_y,
                                    scenario.target_radius)
        collision_t = _collision_t(state, nxt, layout)
        event_t = arrival_t if arrival_t is not None and (
            collision_t is None or arrival_t <= collision_t) else collision_t
        if event_t is not None:
            outcome = "arrival" if event_t == arrival_t else "collision"
            base = _lerp_state(state, nxt, event_t)
            hit = StreetCarState(base.x, base.y, base.heading,
                                 state.speed + (nxt.speed - state.speed) * event_t)
            distance += math.hypot(hit.x - state.x, hit.y - state.y)
            elapsed += step_dt * event_t
            if trajectory is not None:
                trajectory.append((hit.x, hit.y, hit.heading, hit.speed))
            return StreetEpisodeResult(outcome, elapsed, distance, trajectory)
        distance += math.hypot(nxt.x - state.x, nxt.y - state.y)
        state, elapsed = nxt, elapsed + step_dt
        if trajectory is not None:
            trajectory.append((state.x, state.y, state.heading, state.speed))
    return StreetEpisodeResult("timeout", elapsed, distance, trajectory)
