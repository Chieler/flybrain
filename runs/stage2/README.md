# Stage 2 — street grid: results and the direct-compass screening gate

This directory holds the Stage 2 artifacts. Stage 2 puts the corrected Stage 1
car into a fixed street grid with buildings, five local range sensors, and
variable speed. The controller sees only `heading`, `goal_bearing`, `speed`, and
five normalized ranges — never a layout, building rectangle, target coordinate,
route, or waypoint (`Stage2Adapter.__call__(self, left, right, ranges, speed)`
carries no geometry argument, asserted in `test_stage2.py`).

## Original held-out result (preserved)

`checkpoint.json` + `results.json` — the frozen 2026-09-15 real-graph run of the
stateless potential-field adapter over 100 held-out scenarios:

| control | arrivals/100 | collisions | timeouts |
|---|---:|---:|---:|
| neural | **33** | 65 | 2 |
| direct compass (baseline) | 41 | 58 | 1 |
| sensor only | 19 | 81 | 0 |
| neural, no sensors | 7 | 93 | 0 |
| goal cue withheld | 14 | 82 | 4 |
| PFL3 silenced | 14 | 86 | 0 |

The connectome signal contributes (neural beats no-sensors and both neural
ablations), but the adapter does not yet drive reliably: the Stage 2 exit
condition is **NOT MET**. This 33/100 is the original Stage 2 result and stays
as-is.

## Post-run mechanism work and the screening gate (2026-09-16)

Following the HANDOFF "next steps", three sensor-only recovery mechanisms were
added to `Stage2Adapter` and screened on a **cheap direct-compass controller**
(no connectome) against a predeclared reliability gate, *before* spending any
further multi-hour neural run:

1. **Memoryless "veer to open side"** — the original stateless rule.
2. **One-bit commit-and-hold** — when forward clearance drops below
   `commit_distance`, commit to the more-open side and hold a decisive turn
   (with hysteresis) until the forward lane reopens past `release_distance`.
3. **Goal-aware commit** — commit toward the side the brain/compass already
   wants (the sign of the neural steer term), provided that side is at least
   `goal_clearance` open; otherwise fall back to the most-open side. This reads
   no geometry — the goal direction comes from the pooled rates already passed
   in, so the no-geometry guardrail is unchanged.

### Predeclared bracket and gate

- **Controller:** `direct_compass` — reads angular error only, no connectome.
- **Dev split:** `dev_split.json` — 100 fresh scenarios (12 cross / 44 regular /
  44 asymmetric), disjoint by scenario identity from the seed-2 training and
  held-out sets (only the single-intersection `cross` layout reuses routes, with
  new headings, because its route pool is exhausted).
  - `dev_split.json` sha256: `1a3682705aa39d9d2d4acb5c3bd8465f3f1276064fceadc587395e302bb2c3db`
- **Gate:** `arrival_rate >= 0.90` overall **and** `>= 0.80` in every layout.
- **Declared grid (64 candidates):**
  `avoidance_gain ∈ {1.0, 1.5, 2.0, 3.0}` × `commit_distance ∈ {1.5, 2.5}` ×
  `release_margin ∈ {1.5, 3.0}` × `recover_speed ∈ {1.0}` ×
  `goal_clearance ∈ {0.0, 0.3, 0.5, 0.7}`, with `brake_distance = 6.0` fixed
  (inert lever, per the diagnosis). `goal_clearance = 0.0` reproduces the
  pure-openness commit inside the same grid as a baseline.
- **Artifacts:** `dev_bracket.json` (step #3, before goal-awareness),
  `dev_bracket_goalaware.json` (full 64-candidate grid).
  - `dev_bracket_goalaware.json` sha256:
    `02882af93f672fb8e802c6a0ff9a730b87a5c52d8386f9326c28b638b8d57aa7`
- **Reproduce:**
  `python evaluate_stage2.py --bracket --dev-split runs/stage2/dev_split.json --bracket-out runs/stage2/dev_bracket_goalaware.json`

### Result: 0/64 candidates pass the gate

| | overall | cross / regular / asym |
|---|---:|---|
| best overall (`goal_clearance = 0.0`, pure openness) | **0.74** | 0.83 / 0.75 / 0.70 |
| best goal-aware (`goal_clearance = 0.7`) | 0.73 | 0.83 / 0.77 / 0.66 |

Mean overall arrival rate by clearance: `0.0 → 0.611`, `0.3 → 0.595`,
`0.5 → 0.609`, `0.7 → 0.639`. Goal-awareness did not raise the ceiling; every
`goal_clearance > 0` config was at or below the pure-openness baseline. In a
trap the goal direction and the escape direction are in tension precisely
because a building sits between the car and the goal, so "prefer the goal side"
drives into the blocking wall.

## What this does and does not establish

- **Ceiling on these variants:** all three mechanisms top out at **74%** on the
  direct-compass screen and none clears the 90%/80% gate. The expensive neural
  run stays **blocked**.
- **`direct_compass` is a screening baseline, not a mathematical upper bound on
  the connectome.** It is a cheap, idealized goal signal used to reject
  mechanisms quickly; it does not bound what the real graph could do.
- **This does not prove every map-free reactive controller is exhausted.** All
  three variants trigger only *after* forward blockage — they test escape
  behavior, not proactive intersection navigation (detecting a lateral opening
  and beginning a turn before becoming trapped). A controller of that latter
  class has not been tested here.

Follow-on work (waypoint solvability baseline + an observation-only
FOLLOW/TURN state machine) is tracked as a separate **Stage 2b** experiment and
must not be merged into or replace this 33/100 Stage 2 result.
