# Stage 2 Small Street Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible road following and intersection-choice experiments on small street grids while keeping the map hidden from the controller.

**Architecture:** Preserve the Stage 1 simulator and controller contracts, and add a separate `street.py` runtime for axis-aligned buildings, five local range sensors, variable speed, and swept collision termination. Extend the existing neural bridge with one declared Stage 2 adapter that receives only PFL3 left/right rates, local ranges, and current speed; freeze scenarios before calibration and evaluate the adapter and neural contribution with separate controls.

**Tech Stack:** Python 3.12–3.14, standard-library dataclasses/JSON/random/unittest, NumPy, SciPy CSR, and the already-installed Pygame viewer.

**Spec:** [`ROADMAP.md`](../../../ROADMAP.md), especially “Stage 2 — Small street grid” and “Shared constraints”; Stage 1 integrity rules in [`HANDOFF.md`](../../../HANDOFF.md) remain in force.

## Global Constraints

- CPU-first, sparse computation; measure performance before adding acceleration infrastructure.
- Use the entire prepared MaleCNS v1.0 graph, including optic lobes; never silently reduce it.
- The controller receives heading, destination bearing, current speed, and five local range readings, but never a layout, building rectangles, target coordinates, route, waypoint, or shortest-path result.
- Only `avoidance_gain` and `brake_distance` are calibrated in Stage 2. Stage 1 neural dynamics and PFL3 adapter `gain`/`bias` remain frozen from `runs/stage1/checkpoint.json`.
- Training and held-out scenarios are disjoint, seeded, serialized under `runs/stage2/`, and hashed in the Stage 2 checkpoint.
- The fixed initial layouts are connected rectilinear grids. Blocked streets, dead ends, detours, unseen layouts, and route-efficiency claims belong to Stage 3.
- Report arrival, collision, and timeout numerators with the trial denominator; report arrival time and route length among successful trials only.
- A direct-compass controller is an explicitly labeled non-neural baseline. Neural interventions reuse the selected adapter without recalibration.
- Measured wiring, assumed rate dynamics, artificial sensors/motor mapping, and calibrated parameters must remain separately labeled.
- Do not claim the Stage 2 exit condition from implementation checks; it requires the frozen real-graph held-out run.

---

## Scope and deliberate simplifications

Stage 2 adds three fixed, connected layouts: a cross grid with road centerlines at `x=0, y=0`, a regular grid at `x=-10,10, y=-10,10`, and an asymmetric grid at `x=-12,7, y=-8,11`. Every road spans the arena, so authored waypoint pairs are reachable without adding a route planner. Buildings are the rectangular cells between road corridors; their visual complement is the road surface.

The car uses a conservative square collision footprint with half-width `CAR_RADIUS`. Sweeping the car center against rectangles expanded by that radius is exact for this declared footprint and prevents tunneling. Five rays at relative angles `(+90°, +45°, 0°, -45°, -90°)` report distance to the nearest building or arena edge, clipped to `SENSOR_RANGE = 8.0` and normalized to `[0, 1]` before reaching the adapter.

The Stage 2 adapter is intentionally small:

```text
left_open  = mean(left_90, left_45)
right_open = mean(right_45, right_90)
steering   = clip(stage1_gain * (pfl3_left - pfl3_right) + stage1_bias
                  + avoidance_gain * (left_open - right_open), ±MAX_STEERING)
desired_speed = CRUISE_SPEED * min(1, forward_range / brake_distance)
                * max(0.35, 1 - abs(steering) / MAX_STEERING)
acceleration = clip(2 * (desired_speed - speed), -MAX_BRAKE, MAX_ACCEL)
```

Only `avoidance_gain ∈ {0.0, 0.25, 0.5, 1.0}` and `brake_distance ∈ {2.0, 4.0, 6.0}` change during Stage 2 calibration: twelve candidates over twenty-four frozen training scenarios. This grid is small enough to audit and run against the real graph; expand it only after reporting that all twelve declared candidates fail.

## File map

| File | Responsibility |
| --- | --- |
| `street.py` | Fixed grid construction, building collision geometry, local sensors, variable-speed bicycle dynamics, observations, and Stage 2 episode loop. |
| `brain.py` | `Stage2Adapter` and `Stage2NeuralController`; no street geometry. |
| `evaluate_stage2.py` | Frozen scenario generation, calibration, metrics, controls, checkpoint/result JSON, and CLI. |
| `test_stage2.py` | One deterministic standard-library test module for Stage 2 behavior and integrity boundaries. |
| `viewer.py` | Draw selected Stage 2 layout/buildings and replay a recorded Stage 2 trajectory. |
| `runs/stage2/training.json` | Twenty-four frozen calibration scenarios, eight per layout. |
| `runs/stage2/heldout.json` | One hundred frozen evaluation scenarios disjoint from training. |
| `runs/stage2/checkpoint.json` | Source hashes, frozen parameters, twelve calibration summaries, and selected adapter values. |
| `runs/stage2/results.json` | Held-out summaries for the selected system and every declared control. |
| `README.md`, `SETUP.md`, `HANDOFF.md` | Stage 2 commands, status, provenance boundary, and honest result interpretation. |

### Task 1: Street geometry, sensors, and episode dynamics

**Files:**
- Create: `street.py`
- Create: `test_stage2.py`

**Interfaces:**
- Consumes: `simulation.wrap_angle`, `simulation._earliest_hit_t`, `simulation._lerp_state`, `simulation.CAR_RADIUS`, `simulation.TARGET_RADIUS`, `simulation.MAX_STEERING`, and `simulation.PHYSICS_DT`.
- Produces: `Rect`, `StreetLayout`, `StreetCarState`, `StreetScenario`, `StreetObservation`, `Control`, `StreetEpisodeResult`, `initial_layouts()`, `observe_street()`, `street_step()`, and `run_street_episode()`.

- [ ] **Step 1: Write the failing fixed-layout and sensor checks**

```python
# test_stage2.py
import math
import unittest

import street as st


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
```

- [ ] **Step 2: Run the new checks and confirm the module is absent**

Run: `python3 -m unittest test_stage2.TestStreetGeometry -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'street'`.

- [ ] **Step 3: Add the Stage 2 data contracts and fixed layout builder**

```python
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
```

The doubled edge tuples deliberately alternate building cells and road corridors. Keep this builder in `street.py`; do not add a geometry package or a layout class hierarchy.

- [ ] **Step 4: Implement slab intersection and five local ray readings**

```python
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
```

- [ ] **Step 5: Add failing acceleration, arrival, and collision checks**

```python
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
```

- [ ] **Step 6: Run the episode checks and confirm the missing functions fail**

Run: `python3 -m unittest test_stage2.TestStreetEpisode -v`

Expected: FAIL because `street_step` and `run_street_episode` are not defined.

- [ ] **Step 7: Implement variable-speed dynamics and earliest-event termination**

```python
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
```

- [ ] **Step 8: Run Stage 1 and Stage 2 checks**

Run: `python3 -m unittest test_stage1 test_stage2 -q`

Expected: PASS. Fix numeric expectations only if the implementation and declared geometry disagree by floating-point tolerance, never by changing the collision or observation contract.

- [ ] **Step 9: Commit the independently runnable street simulator**

```bash
git add street.py test_stage2.py
git commit -m "feat(stage2): add street grid simulation"
```

### Task 2: Minimal neural/sensor motor adapter

**Files:**
- Modify: `brain.py`
- Modify: `test_stage2.py`

**Interfaces:**
- Consumes: `Brain.encode(heading: float, goal_bearing: float) -> np.ndarray`, `Brain.step(stimulus)`, `Brain.outputs() -> tuple[float, float]`, `street.Control`, `street.StreetObservation`, `street.SENSOR_RANGE`, `street.CRUISE_SPEED`, `street.MAX_ACCEL`, and `street.MAX_BRAKE`.
- Produces: `Stage2Adapter.__call__(left: float, right: float, ranges: tuple[float, ...], speed: float) -> Control` and `Stage2NeuralController.__call__(obs: StreetObservation) -> Control`.

- [ ] **Step 1: Write failing adapter behavior and boundary tests**

```python
import brain as br
from simulation import MAX_STEERING


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
```

- [ ] **Step 2: Run the adapter checks and verify they fail for the missing type**

Run: `python3 -m unittest test_stage2.TestStage2Adapter -v`

Expected: FAIL with `AttributeError: module 'brain' has no attribute 'Stage2Adapter'`.

- [ ] **Step 3: Add the adapter and neural bridge without changing Stage 1 classes**

```python
# brain.py, after NeuralController
@dataclass
class Stage2Adapter:
    brain_gain: float
    bias: float
    avoidance_gain: float
    brake_distance: float
    cruise_speed: float = 2.0
    max_steering: float = math.radians(30.0)

    def __call__(self, left: float, right: float, ranges: tuple[float, ...],
                 speed: float):
        from street import Control, MAX_ACCEL, MAX_BRAKE, SENSOR_RANGE
        normalized = tuple(r / SENSOR_RANGE for r in ranges)
        left_open = sum(normalized[:2]) / 2
        right_open = sum(normalized[-2:]) / 2
        steering = self.brain_gain * (left - right) + self.bias
        steering += self.avoidance_gain * (left_open - right_open)
        steering = max(-self.max_steering, min(self.max_steering, steering))
        desired = self.cruise_speed * min(1.0, ranges[2] / self.brake_distance)
        desired *= max(0.35, 1.0 - abs(steering) / self.max_steering)
        acceleration = max(-MAX_BRAKE, min(MAX_ACCEL, 2.0 * (desired - speed)))
        return Control(steering, acceleration)


class Stage2NeuralController:
    def __init__(self, brain: Brain, adapter: Stage2Adapter, neural_updates: int = 2):
        self.brain = brain
        self.adapter = adapter
        self.neural_updates = neural_updates

    def reset(self) -> None:
        self.brain.reset()

    def __call__(self, obs):
        stimulus = self.brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(self.neural_updates):
            self.brain.step(stimulus)
        left, right = self.brain.outputs()
        return self.adapter(left, right, obs.ranges, obs.speed)
```

The local import prevents `brain.py` from importing street geometry at module load while keeping `Control` as the single motor command type. Do not add a generic policy interface; both existing concrete controllers are sufficient.

- [ ] **Step 4: Add a synthetic end-to-end bridge check**

```python
class TestStage2NeuralController(unittest.TestCase):
    def test_bridge_passes_only_rates_ranges_and_speed_to_adapter(self):
        brain = __import__("test_stage1")._angle_tuned_brain()
        adapter = br.Stage2Adapter(1.0, 0.0, 0.0, 4.0)
        controller = br.Stage2NeuralController(brain, adapter)
        obs = st.StreetObservation(0.0, 0.8, 1.0, (8.0,) * 5)
        control = controller(obs)
        self.assertGreater(control.steering, 0.0)
        self.assertTrue(-st.MAX_BRAKE <= control.acceleration <= st.MAX_ACCEL)
```

- [ ] **Step 5: Run all deterministic checks**

Run: `python3 -m unittest test_stage1 test_stage2 -q`

Expected: PASS with all Stage 1 behavior unchanged.

- [ ] **Step 6: Commit the adapter slice**

```bash
git add brain.py test_stage2.py
git commit -m "feat(stage2): add neural street adapter"
```

### Task 3: Frozen reachable scenarios and calibration

**Files:**
- Create: `evaluate_stage2.py`
- Modify: `test_stage2.py`
- Create: `runs/stage2/training.json`
- Create: `runs/stage2/heldout.json`

**Interfaces:**
- Consumes: `street.initial_layouts()`, `street.run_street_episode()`, `brain.Stage2Adapter`, `brain.Stage2NeuralController`, and the Stage 1 checkpoint shape accepted by `evaluate.load_checkpoint()`.
- Produces: `scenario_to_dict()`, `scenario_from_dict()`, `save_scenarios()`, `load_scenarios()`, `generate_scenario_splits()`, `StreetSummary`, `evaluate_controller()`, `evaluate_breakdown()`, and `calibrate_stage2_adapter()`.

- [ ] **Step 1: Write failing split, serialization, metric, and calibration checks**

```python
import evaluate_stage2 as ev2


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
```

- [ ] **Step 2: Run the evaluation checks and confirm the module is absent**

Run: `python3 -m unittest test_stage2.TestStage2Evaluation -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_stage2'`.

- [ ] **Step 3: Implement deterministic, reachable scenario creation and JSON I/O**

```python
# evaluate_stage2.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from street import (
    ARENA_BOUND, MAX_SIM_TIME, SENSOR_RANGE, Control, StreetCarState, StreetScenario,
    StreetEpisodeResult, StreetLayout, initial_layouts, run_street_episode,
)

AVOIDANCE_GAINS = (0.0, 0.25, 0.5, 1.0)
BRAKE_DISTANCES = (2.0, 4.0, 6.0)


def _waypoints(layout: StreetLayout) -> list[tuple[float, float]]:
    edge = ARENA_BOUND - 4.0
    points = [(x, y) for x in layout.x_roads for y in layout.y_roads]
    points += [(x, -edge) for x in layout.x_roads] + [(x, edge) for x in layout.x_roads]
    points += [(-edge, y) for y in layout.y_roads] + [(edge, y) for y in layout.y_roads]
    return sorted(set(points))


def generate_scenario_splits(seed: int = 2):
    rng = random.Random(seed)
    layouts = initial_layouts()
    headings = (0.0, math.pi / 2, math.pi, -math.pi / 2)
    heldout_counts = {"cross": 12, "regular": 44, "asymmetric": 44}
    training, heldout = [], []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        pairs = [(start, target) for start in points for target in points
                 if start != target and math.dist(start, target) >= 12.0]
        rng.shuffle(pairs)
        train_pairs = pairs[:8]
        test_pairs = pairs[8:8 + heldout_counts[name]]
        training += [StreetScenario(
            name, StreetCarState(*start, rng.choice(headings), 0.0), *target,
            timeout=MAX_SIM_TIME, label=f"train-{name}-{i:03d}",
        ) for i, (start, target) in enumerate(train_pairs)]
        heldout += [StreetScenario(
            name, StreetCarState(*start, rng.choice(headings), 0.0), *target,
            timeout=MAX_SIM_TIME, label=f"heldout-{name}-{i:03d}",
        ) for i, (start, target) in enumerate(test_pairs)]
    return training, heldout


def scenario_to_dict(s: StreetScenario) -> dict:
    return {
        "layout": s.layout,
        "start": {"x": s.start.x, "y": s.start.y,
                  "heading": s.start.heading, "speed": s.start.speed},
        "target_x": s.target_x, "target_y": s.target_y,
        "target_radius": s.target_radius, "timeout": s.timeout, "label": s.label,
    }


def scenario_from_dict(data: dict) -> StreetScenario:
    start = data["start"]
    return StreetScenario(data["layout"], StreetCarState(**start),
                          data["target_x"], data["target_y"],
                          data["target_radius"], data["timeout"], data["label"])


def save_scenarios(path: str, scenarios: list[StreetScenario]) -> None:
    Path(path).write_text(json.dumps([scenario_to_dict(s) for s in scenarios], indent=2) + "\n")


def load_scenarios(path: str) -> list[StreetScenario]:
    return [scenario_from_dict(data) for data in json.loads(Path(path).read_text())]
```

The split samples unique ordered start/destination pairs on connected full-length roads, then assigns one seeded cardinal heading to each pair. It reserves eight training pairs per layout and held-out counts of 12/44/44; the smaller cross grid has only twenty unique ordered waypoint pairs, so this uses all of them without train/test overlap.

- [ ] **Step 4: Implement exact Stage 2 summaries and the fixed calibration budget**

```python
@dataclass(frozen=True)
class StreetSummary:
    trials: int
    arrivals: int
    collisions: int
    timeouts: int
    arrival_rate: float
    mean_arrival_time: float | None
    mean_route_length: float | None

    def as_dict(self) -> dict:
        return self.__dict__


def _run_controller(scenarios: list[StreetScenario],
                    layouts: dict[str, StreetLayout], controller, reset=None):
    results = []
    for scenario in scenarios:
        if reset is not None:
            reset()
        result = run_street_episode(scenario, layouts[scenario.layout], controller)
        results.append((scenario.layout, result))
    return results


def _summarize(results: list[StreetEpisodeResult]) -> StreetSummary:
    arrivals = [r for r in results if r.outcome == "arrival"]
    n = len(results)
    mean = lambda values: sum(values) / len(values) if values else None
    return StreetSummary(
        n, len(arrivals), sum(r.outcome == "collision" for r in results),
        sum(r.outcome == "timeout" for r in results), len(arrivals) / n if n else 0.0,
        mean([r.elapsed_time for r in arrivals]), mean([r.path_length for r in arrivals]),
    )


def evaluate_controller(scenarios: list[StreetScenario],
                        layouts: dict[str, StreetLayout], controller,
                        reset=None) -> StreetSummary:
    return _summarize([result for _, result in
                       _run_controller(scenarios, layouts, controller, reset)])


def evaluate_breakdown(scenarios: list[StreetScenario],
                       layouts: dict[str, StreetLayout], controller,
                       reset=None) -> dict:
    rows = _run_controller(scenarios, layouts, controller, reset)
    return {
        "overall": _summarize([result for _, result in rows]).as_dict(),
        "by_layout": {
            name: _summarize([result for layout, result in rows if layout == name]).as_dict()
            for name in layouts
        },
    }


def calibrate_stage2_adapter(scenarios, make_controller,
                             avoidance_gains=AVOIDANCE_GAINS,
                             brake_distances=BRAKE_DISTANCES):
    layouts, grid, best = initial_layouts(), [], None
    for gain in avoidance_gains:
        for distance in brake_distances:
            controller, reset = make_controller(gain, distance)
            summary = evaluate_controller(scenarios, layouts, controller, reset)
            row = {"avoidance_gain": gain, "brake_distance": distance,
                   **summary.as_dict()}
            grid.append(row)
            route = summary.mean_route_length if summary.mean_route_length is not None else math.inf
            key = (summary.arrivals, -summary.collisions, -summary.timeouts,
                   -route, -gain, -distance)
            if best is None or key > best[0]:
                best = (key, gain, distance)
    return best[1], best[2], grid
```

Selection maximizes arrivals, then minimizes collisions, timeouts, successful route length, avoidance gain, and brake distance in that order. The last two tie-breaks prefer the smaller intervention.

- [ ] **Step 5: Generate and inspect the frozen scenario artifacts**

Add this CLI branch:

```python
if args.freeze_scenarios:
    train, heldout = generate_scenario_splits(seed=args.seed)
    Path("runs/stage2").mkdir(parents=True, exist_ok=True)
    save_scenarios("runs/stage2/training.json", train)
    save_scenarios("runs/stage2/heldout.json", heldout)
```

Add parser flags `--freeze-scenarios` and `--seed` with default `2`, then run:

```bash
python3 evaluate_stage2.py --freeze-scenarios
python3 -c "import evaluate_stage2 as e; a=e.load_scenarios('runs/stage2/training.json'); b=e.load_scenarios('runs/stage2/heldout.json'); assert len(a)==24 and len(b)==100 and {x.label for x in a}.isdisjoint(x.label for x in b)"
```

Expected: both assertions pass, and every serialized scenario names `cross`, `regular`, or `asymmetric`.

- [ ] **Step 6: Run all checks**

Run: `python3 -m unittest test_stage1 test_stage2 -q`

Expected: PASS.

- [ ] **Step 7: Commit scenarios and calibration mechanics**

```bash
git add evaluate_stage2.py test_stage2.py runs/stage2/training.json runs/stage2/heldout.json
git commit -m "feat(stage2): freeze street evaluation scenarios"
```

### Task 4: Real-graph checkpoint and separate adapter/neural controls

**Files:**
- Modify: `evaluate_stage2.py`
- Modify: `test_stage2.py`
- Create after the real run: `runs/stage2/checkpoint.json`
- Create after the real run: `runs/stage2/results.json`

**Interfaces:**
- Consumes: `evaluate.load_checkpoint(path)`, `brain.Brain.load(data_dir, params)`, `brain.Stage2Adapter`, `brain.Stage2NeuralController`, Stage 1 `cue_withheld_controller` behavior, and the frozen Stage 2 JSON files.
- Produces: `stage2_factory()`, `sensor_only_controller()`, `direct_compass_controller()`, `run_stage2_controls()`, `save_stage2_checkpoint()`, and CLI modes `--calibrate` and `--controls`.

- [ ] **Step 1: Write failing integrity and control-accounting checks**

```python
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
```

- [ ] **Step 2: Run the controls checks and confirm missing interfaces fail**

Run: `python3 -m unittest test_stage2.TestStage2Controls -v`

Expected: FAIL because `stage2_factory`, `sensor_only_controller`, and `CONTROL_NAMES` are absent.

- [ ] **Step 3: Add controller factories and labeled non-neural baselines**

```python
from brain import Brain, Stage2Adapter, Stage2NeuralController
from evaluate import load_checkpoint
from simulation import MAX_STEERING, wrap_angle

CONTROL_NAMES = (
    "neural", "direct_compass", "sensor_only", "neural_no_sensors",
    "goal_cue_withheld", "pfl3_silenced",
)


def stage2_factory(brain: Brain, stage1_checkpoint: dict):
    def make(avoidance_gain: float, brake_distance: float):
        adapter = Stage2Adapter(stage1_checkpoint["gain"], stage1_checkpoint["bias"],
                                avoidance_gain, brake_distance)
        controller = Stage2NeuralController(brain, adapter)
        return controller, controller.reset
    return make


def sensor_only_controller(adapter: Stage2Adapter):
    local = Stage2Adapter(0.0, 0.0, adapter.avoidance_gain,
                          adapter.brake_distance, adapter.cruise_speed)
    return lambda obs: local(0.0, 0.0, obs.ranges, obs.speed)


def direct_compass_controller(adapter: Stage2Adapter):
    local = Stage2Adapter(1.0, 0.0, adapter.avoidance_gain,
                          adapter.brake_distance, adapter.cruise_speed)
    def controller(obs):
        error = wrap_angle(obs.goal_bearing - obs.heading)
        signal = max(-MAX_STEERING, min(MAX_STEERING, 4.0 * error))
        return local(signal, 0.0, obs.ranges, obs.speed)
    return controller
```

`direct_compass` is evaluator-only and explicitly reads angular error. It must never be used as a fallback in `Stage2NeuralController`.

- [ ] **Step 4: Add fixed-adapter neural interventions**

```python
def _controlled_neural(brain: Brain, adapter: Stage2Adapter,
                       withhold_goal: bool = False, silence_pfl3: bool = False,
                       ranges_override: tuple[float, ...] | None = None):
    goal = brain.input_map.get("goal")
    goal_indices = goal[0] if goal is not None else None
    outputs = list(brain.output_map["left"]) + list(brain.output_map["right"])
    def controller(obs):
        stimulus = brain.encode(obs.heading, obs.goal_bearing)
        if withhold_goal and goal_indices is not None:
            stimulus[goal_indices] = 0.0
        for _ in range(2):
            brain.step(stimulus)
            if silence_pfl3:
                brain.activity[outputs] = 0.0
        left, right = brain.outputs()
        ranges = obs.ranges if ranges_override is None else ranges_override
        return adapter(left, right, ranges, obs.speed)
    return controller, brain.reset
```

Keep the selected `avoidance_gain` and `brake_distance` unchanged for both interventions. This isolates neural contribution from adapter retraining.

- [ ] **Step 5: Implement the six-way held-out evaluation**

```python
def run_stage2_controls(data_dir: str, stage1_checkpoint: dict,
                        stage2_checkpoint: dict,
                        heldout: list[StreetScenario]) -> dict:
    layouts = initial_layouts()
    brain = Brain.load(data_dir, stage1_checkpoint["params"])
    adapter = Stage2Adapter(stage1_checkpoint["gain"], stage1_checkpoint["bias"],
                            stage2_checkpoint["avoidance_gain"],
                            stage2_checkpoint["brake_distance"])
    full = Stage2NeuralController(brain, adapter)
    controllers = {
        "neural": (full, full.reset),
        "direct_compass": (direct_compass_controller(adapter), None),
        "sensor_only": (sensor_only_controller(adapter), None),
        "neural_no_sensors": _controlled_neural(
            brain, adapter, ranges_override=(SENSOR_RANGE,) * 5),
        "goal_cue_withheld": _controlled_neural(brain, adapter, withhold_goal=True),
        "pfl3_silenced": _controlled_neural(brain, adapter, silence_pfl3=True),
    }
    return {name: evaluate_breakdown(heldout, layouts, controller, reset)
            for name, (controller, reset) in controllers.items()}
```

Interpretation boundaries:

- `neural` versus `sensor_only` and `pfl3_silenced` tests whether the frozen neural readout contributes beyond local wall following.
- `neural` versus `neural_no_sensors` tests the calibrated local sensor adapter; the latter substitutes five maximum-range readings, disabling both avoidance steering and range-based braking.
- `direct_compass` shows what the same local adapter can do with privileged angular error; it is not a neural result.
- `goal_cue_withheld` tests the artificial destination-cue pathway without changing the adapter.

- [ ] **Step 6: Save auditable checkpoint provenance**

```python
def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_stage2_checkpoint(path: str, stage1_path: str, training_path: str,
                           heldout_path: str, gain: float, distance: float,
                           grid: list[dict]) -> None:
    payload = {
        "stage1_checkpoint": stage1_path,
        "stage1_checkpoint_sha256": sha256(stage1_path),
        "training_scenarios": training_path,
        "training_scenarios_sha256": sha256(training_path),
        "heldout_scenarios": heldout_path,
        "heldout_scenarios_sha256": sha256(heldout_path),
        "learned_parameters": ["avoidance_gain", "brake_distance"],
        "avoidance_gain": gain,
        "brake_distance": distance,
        "selection": "max arrivals, min collisions, min timeouts, min successful route length, min gain, min distance",
        "candidates": grid,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
```

Add this provenance check:

```python
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
```

- [ ] **Step 7: Add calibration and controls CLI modes**

Use these exact commands and defaults:

```bash
python3 evaluate_stage2.py --calibrate \
  --data data/malecns-v1.0 \
  --stage1-checkpoint runs/stage1/checkpoint.json \
  --training runs/stage2/training.json \
  --heldout runs/stage2/heldout.json \
  --checkpoint runs/stage2/checkpoint.json

python3 evaluate_stage2.py --controls \
  --data data/malecns-v1.0 \
  --stage1-checkpoint runs/stage1/checkpoint.json \
  --checkpoint runs/stage2/checkpoint.json \
  --heldout runs/stage2/heldout.json \
  --out runs/stage2/results.json
```

Calibration must load the Stage 1 checkpoint through `evaluate.load_checkpoint`, call `calibrate_stage2_adapter`, and write the full twelve-row grid. Controls must verify every recorded SHA-256 before running and refuse a mismatch with a nonzero exit. Print progress after each candidate/control because a real-graph episode is expensive; do not parallelize the shared mutable `Brain`.

- [ ] **Step 8: Run deterministic tests, then the real calibration and held-out suite**

Run:

```bash
python3 -m unittest test_stage1 test_stage2 -q
python3 evaluate_stage2.py --calibrate --data data/malecns-v1.0 --stage1-checkpoint runs/stage1/checkpoint.json --training runs/stage2/training.json --heldout runs/stage2/heldout.json --checkpoint runs/stage2/checkpoint.json
python3 evaluate_stage2.py --controls --data data/malecns-v1.0 --stage1-checkpoint runs/stage1/checkpoint.json --checkpoint runs/stage2/checkpoint.json --heldout runs/stage2/heldout.json --out runs/stage2/results.json
```

Expected: unit tests PASS; calibration reports exactly `12 × 24 = 288` episodes; controls report six 100-trial overall summaries plus per-layout summaries derived from the same episodes. Record observed wall time and peak resident memory beside results in `HANDOFF.md`. If the held-out neural result is poor, commit it unchanged and state that Stage 2 has not met its exit condition.

- [ ] **Step 9: Commit the control harness and reproducible real results**

```bash
git add evaluate_stage2.py test_stage2.py runs/stage2/checkpoint.json runs/stage2/results.json
git commit -m "feat(stage2): evaluate street controls"
```

### Task 5: Replay and documentation

**Files:**
- Modify: `viewer.py`
- Modify: `test_stage2.py`
- Modify: `README.md`
- Modify: `SETUP.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes: `street.initial_layouts()`, `street.run_street_episode(record=True)`, `evaluate_stage2.load_scenarios()`, and the Stage 2 checkpoint/control construction from Task 4.
- Produces: viewer `--stage {1,2}`, building rendering, Stage 2 replay, and documented reproduction/status commands.

- [ ] **Step 1: Generalize the world-to-screen transform without changing Stage 1 output**

Keep `_to_screen()` and `_car_triangle()` unchanged. Add a rectangle helper and check its y-axis conversion:

```python
# viewer.py
def _rect_to_screen(rect):
    left, bottom = _to_screen(rect.min_x, rect.min_y)
    right, top = _to_screen(rect.max_x, rect.max_y)
    return pygame.Rect(left, top, right - left, bottom - top)
```

```python
# test_stage2.py
class TestStage2Viewer(unittest.TestCase):
    def test_building_rect_has_positive_screen_size(self):
        import viewer
        rect = viewer._rect_to_screen(st.Rect(-5.0, -4.0, 2.0, 3.0))
        self.assertGreater(rect.width, 0)
        self.assertGreater(rect.height, 0)
```

- [ ] **Step 2: Add a Stage 2 replay path and buildings**

Add parser options:

```python
p.add_argument("--stage", choices=(1, 2), type=int, default=1)
p.add_argument("--stage2-checkpoint", default="runs/stage2/checkpoint.json")
p.add_argument("--frames", type=int, help="Stop after this many replay frames (smoke tests).")
```

When `--stage 2`, load the selected frozen scenario, build the selected neural or direct-compass controller using `evaluate_stage2`, call `run_street_episode(..., record=True)`, and draw every building before the target/path/car:

```python
BUILDING = (48, 50, 58)
ROAD = (82, 82, 90)

screen.fill(BG)
pygame.draw.rect(screen, ROAD, (MARGIN, MARGIN, WIN - 2 * MARGIN, WIN - 2 * MARGIN))
for building in layout.buildings:
    pygame.draw.rect(screen, BUILDING, _rect_to_screen(building))
```

Count rendered frames separately from the trajectory index and set `running = False` after `args.frames` frames when that option is present.

Keep replay decoupled from neural computation: compute the entire trajectory first, then animate it. Do not add interactive map editing; that belongs to the Stage 3 visitor experiment.

- [ ] **Step 3: Smoke-check Stage 2 rendering headlessly**

Run:

```bash
SDL_VIDEODRIVER=dummy python3 viewer.py --stage 2 --controller baseline \
  --scenarios runs/stage2/heldout.json --index 0 --frames 2
```

Expected: the process initializes Pygame, draws two frames without a coordinate or tuple-unpacking error, and exits. The default interactive replay still holds the final frame until quit.

- [ ] **Step 4: Update project documentation with measured facts**

In `README.md`, change the title/status to cover Stage 1 and Stage 2, add the two evaluator commands from Task 4, add the Stage 2 viewer command, and summarize the six held-out results directly from `runs/stage2/results.json` with denominators.

In `SETUP.md`, add only the Stage 2 commands; dependencies and connectome preparation are unchanged.

In `HANDOFF.md`, add:

- The Stage 2 file roles and observation/control boundary.
- The five sensor angles/range and conservative collision-footprint assumption.
- The two calibrated parameters, twelve-candidate budget, source artifact hashes, and selected values.
- Arrival/collision/timeout counts, successful arrival time/length, wall time, and peak memory for all six controls.
- A literal `Stage 2 exit condition: MET` only if the evidence supports “reliable” behavior across all three layouts; otherwise use `NOT MET` and state the observed failure mode.

- [ ] **Step 5: Run the full verification**

Run:

```bash
python3 -m unittest test_stage1 test_stage2 -q
python3 -c "import json; r=json.load(open('runs/stage2/results.json')); assert set(r)=={'neural','direct_compass','sensor_only','neural_no_sensors','goal_cue_withheld','pfl3_silenced'}; assert all(v['overall']['trials']==100 and set(v['by_layout'])=={'cross','regular','asymmetric'} for v in r.values())"
git diff --check
```

Expected: tests PASS, the result schema assertion passes, and `git diff --check` prints nothing.

- [ ] **Step 6: Commit the replay and factual documentation**

```bash
git add viewer.py test_stage2.py README.md SETUP.md HANDOFF.md
git commit -m "docs(stage2): add street replay and results"
```

## Completion gate

Before calling Stage 2 complete, verify all of the following from committed artifacts:

- `python3 -m unittest test_stage1 test_stage2 -q` passes.
- The frozen split contains 24 training and 100 held-out scenarios with disjoint labels.
- The checkpoint hashes the unchanged Stage 1 checkpoint and both frozen scenario files.
- The checkpoint lists exactly `avoidance_gain` and `brake_distance` as learned parameters and contains all twelve candidate summaries.
- The held-out output contains all six controls with 100-trial overall denominators and separate summaries for all three layouts.
- No controller signature accepts a layout, rectangle, target coordinate, route, or waypoint.
- Documentation distinguishes neural results, learned adapter effects, direct-compass performance, and negative findings.
- `git status --short` is clean after the final commit.

Skipped: a route planner, learned neural weights, a new dependency, arbitrary road editing, blocked streets, and unseen-layout claims. Add those only under a separately scoped Stage 3 plan.
