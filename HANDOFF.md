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

- **Phase 3 finish (run-only):** calibration grid running in cloud →
  `checkpoint.json` (`{gain, bias, params, data_dir}`, written by
  `evaluate.py --calibrate`). Then produce untrained vs. calibrated example
  trajectories. Symmetry / geometry-shortcut checks: **done** (above).
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
- **Phase 5: DONE.** `viewer.py` (pygame 2.6.1, pinned in requirements.txt)
  replays a recorded episode in a window; rendering is decoupled from neural
  compute (`run_episode(record=True)` first, then animate). Screen y-flip lives
  only in the viewer. Runs with no download via the labeled baseline:
  `python viewer.py` (generates a scenario). Neural replay:
  `python viewer.py --controller neural --data <dir> --checkpoint checkpoint.json`.
  Transform + record path smoke-checked headless (SDL_VIDEODRIVER=dummy).

## Git

- `master`: Phases 1–2 slice merged.
- `stage1-prepare-connectome`: connectome pipeline + anatomy angle mapping
  (this handoff lives here; merge to master when integrating).

Reproduce data with SETUP.md — nothing under `data/` is versioned.
