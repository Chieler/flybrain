# Stage 2b — observation-boundary baselines

A separately named experiment to localize the Stage 2 street-navigation failure.
**Non-neural**; it does not touch the connectome or the frozen Stage 2 artifacts,
and must not be merged into or replace the 33/100 Stage 2 result.

Files: `stage2b.py`, `evaluate_stage2b.py`, `test_stage2b.py`.

## Two baselines

- **`WaypointController`** — evaluator-only solvability **witness**. Explicitly
  allowed privileged layout + route + target + pose information (declared, like
  the Stage 1 conventional baseline): it **Dijkstra**-plans a road route and
  follows it with look-ahead pure pursuit, dead-reckoning its own pose from the
  known start with the exact simulator integrator. Its success on a scenario
  witnesses that scenario is solvable; a failure would **not** prove a scenario
  unsolvable (a different route or controller might still succeed).
- **`ObservationStateMachine`** — observation-only FOLLOW / TURN_LEFT / TURN_RIGHT
  machine. Sees exactly `heading, goal_bearing, speed, ranges` (no geometry;
  `__call__(self, obs)`, asserted in `test_stage2b`). It carries short-horizon
  control state (mode, an armed turn, a target heading) but **no route or
  topological memory**. Detects a lateral opening and begins a turn *before*
  becoming trapped, unlike the Stage 2 escape-only variants. Thresholds default
  from arena geometry (road width 6, sensor range 8) and were tuned on the
  Stage 2b **training** split only (never the gate split).

## Full seed-7 dev split (`results.json`, preserved)

| controller | full split | eligible stratum |
|---|---:|---:|
| waypoint | **84/100** | **84/84 = 100%** |
| observation SM | **32/100** | **32/84 = 38.1%** (per layout 60.0 / 36.8 / 33.3) |

## Diagnosis of the 16 waypoint failures (post-hoc, not an impossibility claim)

Every one of the 16 waypoint failures is an **outward-facing road-end start**: a
car at a road's arena-edge endpoint (`(x, ±21)` on an x-road, `(±21, y)` on a
y-road) facing *outward*. `stage2b.is_outward_road_end` flags exactly those 16
(precision and recall 1.0). Such a start must reverse to reach any interior
target, but the full-lock turn radius (1.73) sweeps 3.46 laterally in a 6-wide
lane, so a naive route follower cannot U-turn in place there.

This is a lane-width vs turn-radius interaction — **not** a claim that these
starts are physically impossible. Reversed starts that are *not* outward
road-ends all succeed (10/10). The eligible stratum excludes **only** outward
road-end starts.

## Fresh frozen splits (forbid outward road-end starts)

Disjoint by scenario identity from the inspected seed-7 dev split and from each
other (verified: all pairwise intersections 0).

| file | n | seed | role | sha256 (prefix) |
|---|---:|---:|---|---|
| `gate_split.json` | 100 | 21 | one-shot gate / confirmation | `3f7863a68261f5ee` |
| `sm_train_split.json` | 100 | 22 | SM tuning only | `8df212c4b2700591` |

**Waypoint witness on `gate_split.json`: 100% (1.0 / 1.0 / 1.0) — PASSES 90%/80%.**
Solvability of the eligible stratum is established.

Reproduce: `python evaluate_stage2b.py --freeze-splits` (freezes both splits and
checks the waypoint gate), `python evaluate_stage2b.py` (full seed-7 dev split
baselines + eligible stratum).

## Step 5 result — SM tuned on training, evaluated once on the gate

`ObservationStateMachine` was reworked (corridor centering on both the 90 and 45
degree side rays; commit a 90 degree turn at a goalward opening) and tuned with a
**144-point sweep on `sm_train_split.json` only** (a grid over opening/forward_block/turn_speed/trigger45/turn_delay). The
inspected seed-7 dev split and the gate split were never used for tuning. Best
training config (now the `ObservationStateMachine` defaults): `opening=5.0,
forward_block=3.0, turn_speed=1.0, trigger45=False, turn_delay=0.0` — training
arrival **0.70**.

**One-shot evaluation on `gate_split.json`** (`gate_results.json`, sha256
`b190f38fdbd834ac…`):

| controller | overall | cross / regular / asym | collisions |
|---|---:|---|---:|
| waypoint (privileged map/route/pose) | **1.00** PASS | 1.00 / 1.00 / 1.00 | 0 |
| observation-only SM | **0.62** fail | 0.92 / 0.50 / 0.66 | 38 |

**Reading (bounded).** The waypoint witness reaching 100% shows the vehicle and
layouts are solvable on the eligible stratum — the task is not impossible. The
best of this **144-configuration SM family** — a tuned observation-only
controller *without* route/topological memory — reaches only 62% and fails the
gate, degrading with intersection count (single-intersection `cross` 92% ->
multi-intersection `regular` 50%), consistent with greedy per-intersection turn
choices made without route memory. The waypoint controller also has privileged
layout, route, target, and pose information, so the 100%-vs-62% gap does **not**
by itself localize the deficit to memory — the two differ in map, route, and pose
as well. What the result does: it **establishes solvability** and **motivates
route/history memory as the next hypothesis**.

**Next gate (before any dopamine/connectome learning).** Run a same-observation
**recurrent / history** baseline — same `heading, goal_bearing, speed, ranges`,
still no map and no pose, but with learned temporal state — against this frozen
gate split. That directly tests whether route memory explains the 62% gap: if the
recurrent baseline clears 90%/80%, memory (not the interface) is the missing
piece; if it does not, the interface itself is implicated. Only after that
consider biologically grounded learning, with dopamine as a teaching/modulatory
signal — **never** as goal bearing.
