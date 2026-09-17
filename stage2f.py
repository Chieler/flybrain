"""Stage 2f: aligned-reward warm-start fine-tuning (route-memory gate, take 4).

Purpose (see docs/superpowers/specs/2026-09-17-stage2f-aligned-reward-design.md).
Stage 2e escaped 2d's optimization floor (recurrent gate 0.00 -> 0.516, zero
timeouts) and beat its ablation, but neither cleared the 0.90/0.80 gate. The
diagnosis: fitness (mean episode reward) did not track the gate metric -- the
optimizer could raise reward while arrivals fell. Stage 2f makes ARRIVAL COUNT the
primary CEM objective and demotes the bounded-net-progress shaping to a strictly
bounded (< 1) tie-breaker that can never outweigh a single arrival.

Only a PASS is conclusive, scoped to the diagonal waypoint-witnessed cross stratum
plus the cardinal regular/asym strata. A failure stays bounded and confounded
between reward alignment and scenario coverage; it never implicates the observation
interface. Does not touch the connectome or the frozen Stage 2/2b/2c/2d/2e
artifacts; no spent gate is rerun or replaced.
"""

from __future__ import annotations

import math

from street import MAX_SIM_TIME, run_street_episode
from stage2c import RecurrentController
from stage2d import _final_xy

# Frozen shaping coefficients (design choices; not tuned on the 2f gate). Arrival
# count is the integer primary term; these order the sub-1 tie-break only.
W_PROGRESS = 0.5
W_COLLIDE = 2.0
W_TIME = 0.1
SECONDARY_DIVISOR = 4.0


def episode_secondary(scenario, result) -> float:
    """Bounded per-episode shaping in [-2.6, 0.5]. Net progress uses the FINAL
    position (unfarmable). Privileged distances are used here only -- never as
    controller inputs."""
    d0 = math.hypot(scenario.target_x - scenario.start.x,
                    scenario.target_y - scenario.start.y)
    fx, fy = _final_xy(scenario, result)
    d_final = math.hypot(scenario.target_x - fx, scenario.target_y - fy)
    progress = (d0 - d_final) / d0 if d0 > 0.0 else 0.0
    progress = max(-1.0, min(1.0, progress))
    s = W_PROGRESS * progress
    s -= W_COLLIDE if result.outcome == "collision" else 0.0
    s -= W_TIME * (result.elapsed_time / MAX_SIM_TIME)
    return s


def aligned_fitness(esn, scenarios, layouts) -> float:
    """Arrival-count-primary fitness for the policy currently on `esn`:
    `arrivals + mean(episode_secondary) / SECONDARY_DIVISOR`. The mean-then-divide
    contribution lies in [-0.65, 0.125] (swing < 1), so one more arrival always
    wins."""
    controller = RecurrentController(esn)
    arrivals = 0
    secondary_total = 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        if result.outcome == "arrival":
            arrivals += 1
        secondary_total += episode_secondary(scenario, result)
    return arrivals + (secondary_total / len(scenarios)) / SECONDARY_DIVISOR
