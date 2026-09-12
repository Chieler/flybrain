"""Open-arena car simulation for Stage 1.

World coordinates: +x east, +y north, counterclockwise-positive angles.
Screen conversion happens only in the viewer.

Boundary: the simulator owns target geometry (needed for termination) and the
car kinematics. Controllers receive only a NeuralObservation and must never see
target coordinates or arena geometry. See ROADMAP.md / plan for the contract.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- Physical parameters (exposed in saved configuration) ---
WHEELBASE = 1.0          # units
SPEED = 2.0              # units/s, fixed
MAX_STEERING = math.radians(30.0)  # radians
CAR_RADIUS = 0.5         # units
TARGET_RADIUS = 1.5      # units
ARENA_BOUND = 25.0       # units, arena is [-25, 25] on both axes
PHYSICS_DT = 0.02        # s
NEURAL_UPDATES_PER_PHYSICS = 2
MAX_SIM_TIME = 30.0      # s, hard episode cap


@dataclass
class CarState:
    x: float
    y: float
    heading: float  # radians, world frame


@dataclass
class Scenario:
    start: CarState
    target_x: float
    target_y: float
    target_radius: float = TARGET_RADIUS
    timeout: float = MAX_SIM_TIME
    label: str = ""


@dataclass
class NeuralObservation:
    heading: float       # current car heading, radians
    goal_bearing: float  # bearing to target, radians (world frame)


@dataclass
class EpisodeResult:
    outcome: str            # "arrival" | "boundary" | "timeout"
    elapsed_time: float     # simulated seconds
    path_length: float      # units travelled
    return_: Optional[float] = None  # filled by the evaluator (owns rewards)
    trajectory: Optional[list] = field(default=None)  # list[(x, y, heading)]


# Controller contract: sees only the observation, returns a steering angle.
Controller = Callable[[NeuralObservation], float]


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def bicycle_step(state: CarState, steering: float, dt: float) -> CarState:
    """Kinematic bicycle model, fixed speed."""
    steering = max(-MAX_STEERING, min(MAX_STEERING, steering))
    x = state.x + SPEED * math.cos(state.heading) * dt
    y = state.y + SPEED * math.sin(state.heading) * dt
    heading = wrap_angle(state.heading + SPEED / WHEELBASE * math.tan(steering) * dt)
    return CarState(x, y, heading)


def observe(state: CarState, scenario: Scenario) -> NeuralObservation:
    goal_bearing = math.atan2(scenario.target_y - state.y, scenario.target_x - state.x)
    return NeuralObservation(heading=state.heading, goal_bearing=goal_bearing)


def _earliest_hit_t(ax, ay, bx, by, cx, cy, r) -> Optional[float]:
    """Smallest t in [0,1] where the segment A->B is within r of C, else None."""
    # Already inside at start.
    if (ax - cx) ** 2 + (ay - cy) ** 2 <= r * r:
        return 0.0
    dx, dy = bx - ax, by - ay
    fa = dx * dx + dy * dy
    if fa == 0.0:
        return None
    fb = 2.0 * (dx * (ax - cx) + dy * (ay - cy))
    fc = (ax - cx) ** 2 + (ay - cy) ** 2 - r * r
    disc = fb * fb - 4.0 * fa * fc
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t = (-fb - sq) / (2.0 * fa)
    if 0.0 <= t <= 1.0:
        return t
    return None


def _earliest_exit_t(ax, ay, bx, by, bound) -> Optional[float]:
    """Smallest t in [0,1] where the car center leaves the safe inner box."""
    if abs(ax) >= bound or abs(ay) >= bound:
        return 0.0
    dx, dy = bx - ax, by - ay
    best = None
    for a, d in ((ax, dx), (ay, dy)):
        if d > 0 and a + d > bound:
            t = (bound - a) / d
        elif d < 0 and a + d < -bound:
            t = (-bound - a) / d
        else:
            continue
        if 0.0 <= t <= 1.0 and (best is None or t < best):
            best = t
    return best


def _lerp_state(a: CarState, b: CarState, t: float) -> CarState:
    return CarState(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                    wrap_angle(a.heading + wrap_angle(b.heading - a.heading) * t))


def run_episode(
    scenario: Scenario,
    controller: Controller,
    dt: float = PHYSICS_DT,
    record: bool = False,
) -> EpisodeResult:
    """Advance the car under `controller` until termination.

    Swept checks against the target and boundary make sure a discrete step
    cannot skip arrival or wall contact; the earliest event wins.
    """
    state = scenario.start
    safe_bound = ARENA_BOUND - CAR_RADIUS
    elapsed = 0.0
    path_length = 0.0
    trajectory = [(state.x, state.y, state.heading)] if record else None

    # Exact-target start counts as arrival before any cue/step.
    if _earliest_hit_t(state.x, state.y, state.x, state.y,
                       scenario.target_x, scenario.target_y, scenario.target_radius) == 0.0:
        return EpisodeResult("arrival", 0.0, 0.0, trajectory=trajectory)

    cap = min(scenario.timeout, MAX_SIM_TIME)
    while elapsed < cap:
        step_dt = min(dt, cap - elapsed)
        obs = observe(state, scenario)  # recompute cues each physics update
        steering = controller(obs)
        nxt = bicycle_step(state, steering, step_dt)

        t_hit = _earliest_hit_t(state.x, state.y, nxt.x, nxt.y,
                                scenario.target_x, scenario.target_y, scenario.target_radius)
        t_exit = _earliest_exit_t(state.x, state.y, nxt.x, nxt.y, safe_bound)

        event = None
        t_ev = None
        if t_hit is not None and (t_exit is None or t_hit <= t_exit):
            event, t_ev = "arrival", t_hit
        elif t_exit is not None:
            event, t_ev = "boundary", t_exit

        if event is not None:
            hit = _lerp_state(state, nxt, t_ev)
            path_length += math.hypot(hit.x - state.x, hit.y - state.y)
            elapsed += step_dt * t_ev
            if record:
                trajectory.append((hit.x, hit.y, hit.heading))
            return EpisodeResult(event, elapsed, path_length, trajectory=trajectory)

        path_length += math.hypot(nxt.x - state.x, nxt.y - state.y)
        state = nxt
        elapsed += step_dt
        if record:
            trajectory.append((state.x, state.y, state.heading))

    return EpisodeResult("timeout", elapsed, path_length, trajectory=trajectory)


def _benchmark(seconds: float) -> None:
    """Time the sim loop with a trivial controller (no neural graph loaded)."""
    scenario = Scenario(CarState(-20.0, 0.0, 0.0), 100.0, 0.0, timeout=seconds)
    steps = int(seconds / PHYSICS_DT)
    start = time.perf_counter()
    run_episode(scenario, lambda obs: 0.0, record=False)
    wall = time.perf_counter() - start
    print(f"sim loop: {steps} steps, {seconds:.0f} sim-s in {wall:.4f} wall-s "
          f"(ratio {seconds / wall:.0f}x). Note: physics only, no neural graph.")


def _neural_benchmark(data_dir: str, seconds: float) -> None:
    """Phase 2 neural benchmark: graph counts + memory, per-step wall time, and
    the 10 ms vs 5 ms neural-timestep comparison on the real graph.

    Matvec cost is independent of neural_dt, so both timesteps share a per-step
    cost; the realtime ratio differs only because a finer dt needs more steps
    per simulated second. The dynamics comparison settles the same fixed cue at
    each dt and reports how far the pooled outputs and steering diverge — a check
    that the 10 ms default is not coarser than the model can tolerate.
    """
    import brain as br  # local import: keeps the physics module free of brain

    b = br.Brain.load(data_dir)
    W = b.W
    mem_mb = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1e6
    print(f"graph: {b.n} neurons, {W.nnz} edges, W {mem_mb:.1f} MB "
          f"(+ activity {b.activity.nbytes / 1e6:.2f} MB)")

    stim = b.encode(0.0, 0.8)  # fixed off-axis goal cue
    for _ in range(5):
        b.step(stim)           # warm caches before timing

    # Per-step matvec cost (dt-independent).
    n_steps = 200
    b.reset()
    start = time.perf_counter()
    for _ in range(n_steps):
        b.step(stim)
    per_step = (time.perf_counter() - start) / n_steps
    print(f"neural step: {per_step * 1e3:.2f} ms/step over the {W.nnz}-edge matvec")

    # 10 ms vs 5 ms: same per-step cost, ratio scales with dt; dynamics compared.
    out = {}
    for dt in (0.010, 0.005):
        b.params.neural_dt = dt
        b.reset()
        steps = int(seconds / dt)           # steps to cover `seconds` of sim time
        for _ in range(steps):
            b.step(stim)
        left, right = b.outputs()
        ratio = dt / per_step               # >1 faster than realtime
        out[dt] = (left, right)
        print(f"  dt={dt * 1e3:4.0f} ms: {steps} steps/{seconds:.0f}s, "
              f"ratio {ratio:.2f}x realtime, pooled L/R = {left:.5f}/{right:.5f}, "
              f"steer signal (L-R) = {left - right:+.6f}")
    (l10, r10), (l5, r5) = out[0.010], out[0.005]
    print(f"  10ms vs 5ms divergence: |dL|={abs(l10 - l5):.2e} "
          f"|dR|={abs(r10 - r5):.2e} |d(L-R)|={abs((l10 - r10) - (l5 - r5)):.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 car simulation (headless).")
    p.add_argument("--benchmark", action="store_true",
                   help="Time the physics loop (no neural graph in this slice).")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--data", help="Prepared connectome dir; enables the neural benchmark.")
    args = p.parse_args()
    if args.benchmark:
        if args.data:
            _neural_benchmark(args.data, args.seconds)
        else:
            _benchmark(args.seconds)
    else:
        p.print_help()
