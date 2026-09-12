# Handoff — Flybrain Stage 1

For an agent picking this up on another machine. Read `SETUP.md` first to get
running, then this for what is done, what remains, and the rules that must not
be broken.

## What this is

A car drives toward a target in an open 2D arena. Steering comes from a bounded
rate-model neural network whose weights are the **mapped MaleCNS v1.0**
connectome (male fruit-fly brain). Plan: `docs/superpowers/plans/2026-09-11-stage-1-driving.md`.

## Non-negotiable constraints (scientific integrity)

These come from the plan. Violating them invalidates the result.

- **Never fabricate neuron IDs or preferred angles.** Preferred angles are
  derived only from anatomy (PB glomerulus / FB column index parsed from the
  `instance` string). No sorting-by-ID, no made-up tuning.
- **Keep the four layers distinct:** measured wiring vs. assumed dynamics vs.
  artificial interfaces vs. learned params. The manifest tags each.
- **No silent graph reduction.** Missing a perf budget triggers a *documented*
  optimization decision, never a quiet subsample.
- **No hard-coded turn** in the sensory encoder.
- **The adapter/controller must NEVER see target geometry** — only the two
  pooled PFL3 rates. `Adapter.__call__(left, right)` takes exactly those.
- An explicitly artificial mapping is allowed only as a **declared** experiment,
  never an unannounced fallback.
- Report failures and denominators honestly.

## Layout

| File | Role |
|------|------|
| `simulation.py` | Arena, bicycle kinematics, swept target/boundary termination. Owns geometry. Controllers see only `NeuralObservation(heading, goal_bearing)`. |
| `brain.py` | `RateParams`, `Brain` (sparse rate model, `.load()`, `.encode()`, `.step()`, `.outputs()`), `Adapter`, `NeuralController`. |
| `prepare_connectome.py` | Feather → CSR pipeline. Transmitter signs, retained-neuron set, anatomy-derived preferred angles, manifest. |
| `evaluate.py` | Rewards, scenario generation, `conventional_baseline`, `evaluate_controller`. |
| `test_stage1.py` | 26 deterministic tests. |
| `viewer.py` | Phase 5 pygame replay of a recorded episode (baseline or neural). |
| `README.md` / `ROADMAP.md` | Scope, honesty boundary, phase map. |
| `data/` | gitignored — see SETUP.md to reproduce. |

## Model, briefly

- Dynamics: `drive = clip(baseline + recurrent_gain*(W@activity) + stimulus, 0, 1)`;
  `activity += (neural_dt/tau)*(drive - activity)`.
- `W` is CSR postsynaptic-rows / presynaptic-columns, so `W@activity` propagates
  in the recorded direction. Rows normalized by total absolute incoming weight,
  one global `recurrent_gain`.
- Transmitter sign: acetylcholine +1; gaba/glutamate −1; unclear/modulatory/
  low-confidence 0 (confidence ≥ 0.5). Glutamate sign is an approximation.
- Cue tuning: `max(0, cos(angle - preferred)) * input_gain`.
- Interface: EPG → heading, FC2 → goal, PFL3 L/R → steering.

## Status

**Done ✓**

- Phase 1: connectome downloaded, converted, validated; anatomy-based preferred
  angles derived and documented.
- Phase 2: arena + neural runtime; real-graph dynamics stable and bounded.
- Phase 3: left/right symmetry, monotonicity, and the adapter geometry-shortcut
  guard are checked in `test_stage1.py` (`TestNeuralSteering`). Full rotational
  invariance is NOT expected — the compass tuning is absolute, not relative.
- Phase 4 harness **code-complete** in `evaluate.py`, unit-tested on synthetic
  graphs (`TestControls`): `calibrate_adapter` (declared grid, tie-break lower
  |bias| then smaller gain), controls (`zero`/`random`), interventions
  (`cue_withheld`, `pathway_silenced`), `shuffle_connectivity` (per-source
  identity/sign/weight-multiset preserved), checkpoint I/O, and
  `run_controls`. Running against the real graph is blocked only on `data/` +
  a calibration checkpoint (see below).
- Phase 3 (partial): steering signal verified **directionally correct** —
  `(left − right)` is monotonic and sign-correct in goal bearing:

  ```
  goal_bearing  left    right   left-right
  -1.50        0.0395  0.0469  -0.00740
  -0.75        0.0376  0.0426  -0.00500
  +0.00        0.0424  0.0387  +0.00374
  +0.75        0.0509  0.0406  +0.01035
  +1.50        0.0526  0.0411  +0.01150
  ```

  Goal to the left → positive steer, matches adapter convention. Small
  zero-bearing offset (+0.004) the adapter `bias` absorbs.

**Blocked / open**

- **Compute wall.** Neural step ~13.4 ms, ~0.75× realtime, memory-bandwidth
  bound on the ~20M-edge matvec. Full calibration grid ≈ 9 h; Phase 4 controls
  ≈ tens of hours. Needs a decision (optimize matvec / reduced labeled first
  pass / long background run) — NOT silent graph reduction.

## Remaining work

- **Phase 3:** calibration completed (gain 32, bias -0.05; 7 arrivals on 32
  frozen calibration scenarios); checkpoint and example trajectories are under
  `runs/stage1/`.
- **Phase 2 leftovers: DONE.** `simulation.py --benchmark --seconds 60 --data <dir>`
  runs the neural benchmark on the real graph. Measured (MaleCNS v1.0):
  191,148 neurons, 19,924,788 edges, W 240.6 MB (+0.76 MB activity);
  13.4 ms/neural step. At 10 ms timestep 0.75× realtime, at 5 ms 0.37×.
  10 ms vs 5 ms steady-state divergence ~1.5e-8 (float32 noise), steering
  identical — **10 ms default confirmed adequate**, 5 ms doubles cost for no
  accuracy gain.
- **Phase 4 (run-only):** once the checkpoint lands, run
  `evaluate.py --controls --data <dir> --checkpoint checkpoint.json
  --scenarios heldout.json`. Harness (neural + baseline + zero/random +
  cue-withheld + pathway-silenced + shuffled×3-recalibrated) is written and
  unit-tested; only execution against the real graph remains.
- **Phase 5 (partial):** `viewer.py` (pygame 2.6.1, pinned in requirements.txt)
  replays a recorded episode in a window; rendering is decoupled from neural
  compute (`run_episode(record=True)` first, then animate). Screen y-flip lives
  only in the viewer. Runs with no download via the labeled baseline:
  `python viewer.py` (generates a scenario). Neural replay:
  `python viewer.py --controller neural --data <dir> --checkpoint checkpoint.json`.
  Transform + record path smoke-checked headless (SDL_VIDEODRIVER=dummy).

## Stage 2 — street grid

Plan: `docs/superpowers/plans/2026-09-12-stage-2-street-grid.md`. Separate
runtime; Stage 1 contracts unchanged.

**Files**

| File | Stage 2 role |
|------|------|
| `street.py` | Fixed grid construction, building collision geometry, five local range sensors, variable-speed bicycle dynamics, swept-collision episode loop. Owns all Stage 2 geometry. |
| `brain.py` | `Stage2Adapter` (PFL3 rates + ranges + speed → `Control`) and `Stage2NeuralController`. No street geometry at import (local import inside `__call__`). |
| `evaluate_stage2.py` | Frozen seeded splits, calibration grid, exact summaries, six controls, checkpoint provenance, `--freeze-scenarios/--calibrate/--controls`. |
| `test_stage2.py` | Deterministic Stage 2 behavior + integrity-boundary checks. |
| `viewer.py` | `--stage 2` draws buildings/road and replays a recorded 4-tuple trajectory. |
| `runs/stage2/` | `training.json` (24), `heldout.json` (100) committed; `checkpoint.json` + `results.json` produced by the real-graph run. |

**Observation / control boundary.** The controller receives heading, destination
bearing, current speed, and five local range readings — never a layout, building
rectangle, target coordinate, route, or waypoint. `Stage2Adapter.__call__(self,
left, right, ranges, speed)` carries no geometry argument (asserted in
`test_stage2`).

**Sensors & collision.** Five rays at relative angles `(+90°, +45°, 0°, −45°,
−90°)`, distance to nearest building or arena edge, clipped to
`SENSOR_RANGE = 8.0` and normalized to `[0,1]` before the adapter. Collision uses
a conservative square footprint of half-width `CAR_RADIUS`: the car center is
swept against rectangles expanded by that radius — exact for this declared
footprint and tunnel-proof.

**Calibrated parameters.** Only `avoidance_gain ∈ {0.0, 0.25, 0.5, 1.0}` and
`brake_distance ∈ {2.0, 4.0, 6.0}` — 12 candidates over 24 frozen training
scenarios. Stage 1 neural dynamics and the PFL3 adapter `gain`/`bias` stay frozen
from `runs/stage1/checkpoint.json`. The Stage 2 checkpoint hashes (SHA-256) the
Stage 1 checkpoint and both frozen scenario files; `--controls` refuses a hash
mismatch.

**Controls (all reuse the selected adapter, no recalibration):** `neural`,
`direct_compass` (labeled non-neural baseline, reads angular error),
`sensor_only`, `neural_no_sensors` (ranges forced to max), `goal_cue_withheld`,
`pfl3_silenced`.

**Results & exit condition — PENDING.** One full-graph episode ≈ 47 s;
calibration (288) + controls (600) ≈ 11–12 h, run separately. Until
`runs/stage2/checkpoint.json` and `runs/stage2/results.json` are committed:
**Stage 2 exit condition: NOT MET** (held-out real-graph evidence not yet
produced). Fill the selected `avoidance_gain`/`brake_distance`, the six-control
arrival/collision/timeout counts with denominators, successful arrival
time/length, wall time, and peak memory here once the run lands. Do not claim
MET without reliable behavior across all three layouts.

## Git

- `master`: Phases 1–2 slice merged.
- `stage1-prepare-connectome`: connectome pipeline + anatomy angle mapping
  (this handoff lives here; merge to master when integrating).

Reproduce data with SETUP.md — nothing under `data/` is versioned.
