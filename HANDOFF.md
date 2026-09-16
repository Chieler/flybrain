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
  `run_controls`.
- **Phase 4 real-graph controls DONE ✓** — full run over 100 held-out scenarios
  written to `runs/stage1/results.json`. Wall clock ~18.5 h (8 workers,
  `--jobs 8`; includes a mid-run sleep interval — the run survived it intact).

  | control | trials | arriv | bound | tmout | mean_return |
  |---|---:|---:|---:|---:|---:|
  | neural (real connectome) | 100 | **20** | 40 | 40 | −0.403 |
  | conventional (labeled baseline) | 100 | 100 | 0 | 0 | +0.947 |
  | zero | 100 | 7 | 93 | 0 | −0.989 |
  | random | 100 | 6 | 94 | 0 | −1.008 |
  | cue_withheld | 100 | 6 | 75 | 19 | −0.870 |
  | pathway_silenced | 100 | 5 | 95 | 0 | −1.048 |
  | shuffled_seed1 (recalibrated) | 100 | 7 | 93 | 0 | −0.988 |
  | shuffled_seed2 (recalibrated) | 100 | 6 | 94 | 0 | −1.011 |
  | shuffled_seed3 (recalibrated) | 100 | 6 | 66 | 28 | −0.786 |

  Read: the real connectome (20/100) beats every control, all of which sit at
  the ~5–7/100 chance floor. Withholding the goal cue, silencing PFL3, and
  shuffling connectivity (identity/sign/weight-multiset preserved **and**
  re-calibrated per seed) each collapse to chance — the *specific wiring* carries
  the steering signal, not the interfaces or a tuned readout. Conventional
  baseline 100/100 confirms the scenarios are solvable, so 20% is the connectome's
  honest ceiling, not the task's. Modest and mean-return-negative: evidence of
  directed steering, **not** competent driving.
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

- **Compute wall (Stage 1: resolved by long background run).** Neural step
  ~13.4 ms, ~0.75× realtime, memory-bandwidth bound on the ~20M-edge matvec.
  Phase 4 controls ran ~18.5 h wall at `--jobs 8` (bandwidth-bound, so ~20–45%
  CPU/worker, sub-linear scaling). No graph reduction. NOTE: `run_controls` has
  **no mid-run checkpoint** — `results.json` is written only at the very end, so
  a kill/restart discards the whole run. Keep the machine on AC or under
  `caffeinate -s` for the next long run.
- **Stage 2 real-graph run: complete, exit condition not met.** Calibration and
  all six 100-episode controls are in `runs/stage2/`; see the diagnosis below.

## Stage 1 improvement plan: exceed 20/100

### 2026-09-14 root cause: EPG left/right phase mapping (supersedes the plan below)

**Verdict: B — the project is not shown fundamentally unsound; the failed
Stage 1b decode was operating on a misregistered heading input.** The
The original `prepare_connectome.py` manifest mapped PB glomerulus `i` to
`2π(i−1)/8` on both sides. That “identical L/R copies” convention is not a
global offset/handedness that the adapter can absorb. The two PB halves wrap
the ellipsoid body in opposite directions, and the two medial EPG glomeruli
occupy adjacent wedges separated by 22.5°. See the anatomical mapping in
[Wolff et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4407839/) and the EPG/
Δ7 angle construction in [Pires et al.](https://www.nature.com/articles/s41586-023-07006-3).

A reproducible upstream diagnostic now lives in `diagnose_stage1.py` with
checks in `test_diagnose_stage1.py`. It independently fits the heading and goal
phase delivered to each PFL3 neuron, partitions recurrent drive by EPG, Δ7,
FC2, and other sources, and scores the original two-pooled-rate steering sign.

| Diagnostic | current identical L/R | mirrored EPG mapping |
|---|---:|---:|
| central EPG+Δ7+FC2 sign-correct | 40/64 (62.5%) | 55/64 (85.9%) |
| full next-step drive sign-correct | 44/64 (68.8%) | 60/64 (93.8%) |
| settled pooled PFL3 sign-correct | 44/64 (68.8%) | 60/64 (93.8%) |
| left goal−heading phase concentration | 0.882 | 0.983 |
| right goal−heading phase concentration | 0.524 | 0.982 |
| mean heading first-harmonic R² | 0.940 | 0.972 |
| mean goal first-harmonic R² | 0.892 | 0.892 |

The literature-consistent diagnostic mapping is `Lᵢ = +(i−1)·45°`,
`Rᵢ = −22.5°−(i−1)·45°`. Its fitted goal-minus-heading offsets are +60.5°
and −80.3° for the two PFL3 output populations, close to the opposing offsets
expected by the published steering circuit, with high within-side coherence.
Persisted-manifest gate artifact:
`runs/stage1/pfl3_diagnostic_corrected_manifest.json`. This is a static
screen, not an episode result and not yet evidence of >20/100 arrivals.

**What was wrong in the Stage 1b design record.** The step-3 registration
screen varied only a *global* goal phase, cue-gain ratio, and recurrent gain
while retaining identical PB handedness. It therefore could not test or repair
the actual side-specific coordinate error; its claim that registration was
ruled out was false. The §10 decoder spike remains a valid result conditional
on the broken input map, but “no PFL3 decode can beat 20/100” was too broad.
The spike showed that an output decoder cannot reconstruct information already
scrambled upstream. Do not build the column-resolved decoder now: the original
pooled readout reaches 60/64 after correcting the input convention.

**Smallest honest route above 20/100:**

1. Patch only the EPG preferred-angle construction in `prepare_connectome.py`,
   update its convention/reference and one mapping test, then regenerate the
   interface manifest without changing graph weights, dynamics, arena, or
   PFL3 output pooling. Preserve the original Stage 1 artifacts and label this
   follow-up separately.
2. Re-run the diagnostic as a gate; require the reproduced settled score to
   remain at least 60/64 and both side concentrations to remain at least 0.95.
3. Recalibrate only pooled adapter gain/bias on the frozen 32 training
   scenarios. Do not carry forward the old gain/bias, which were fitted to the
   broken coordinate map.
4. Freeze a new 100-scenario confirmation set before opening results because
   the old held-out set has already influenced design decisions. Run it once;
   proceed to the expensive matched controls only if arrivals exceed 20 and
   cue-withheld/PFL3-silenced controls collapse.

The older ordered plan below is retained as experiment history, not current
guidance.

**Implementation status (2026-09-14).** Steps 1–2 are complete:
`prepare_connectome.py` now owns the side-aware mapping, the manifest was
regenerated, all 50 tests pass, and the persisted-manifest gate reproduced
60/64 with both phase concentrations above 0.98. A new 100-scenario
confirmation set was frozen at
`runs/stage1/confirmation_corrected_epg_seed20260914.json` before evaluation.
Step 3 calibration uses the unchanged declared default 8×3 grid over the 32
training scenarios and writes `runs/stage1/checkpoint_corrected_epg.json`.
Calibration completed with `gain=32`, `bias=-0.05`. The frozen one-time
confirmation then produced **92/100 arrivals, 1 boundary, 7 timeouts**, mean
arrival time 11.90 s and mean return +0.777. Artifact:
`runs/stage1/results_corrected_epg_confirmation.json`. This legitimately beats
20/100 on unseen scenarios and strongly supports the coordinate-map diagnosis;
the original 20/100 remains the historical Stage 1 result.

**Matched controls complete (2026-09-15).** Corrected neural achieved 92/100;
conventional 100/100; zero and random 4/100 each; goal cue withheld 5/100;
PFL3 silenced 5/100; independently shuffled connectomes 3/100, 6/100, and
3/100 after recalibration. Thus the 92/100 result clears the causal acceptance
gate: performance requires the goal cue, PFL3 pathway, and specific recorded
connectivity. Final artifact: `runs/stage1/results_corrected_epg_controls.json`.

This historical guard has been satisfied: Stage 2 was run only after the
corrected Stage 1 checkpoint existed, and its checkpoint records that exact
artifact and hash. Keep the current 20/100 result as the original Stage 1
experiment; every experiment below is a separately declared follow-up.

**Diagnosis.** The chosen adapter (`gain=32`, `bias=-0.05`) landed on both edges
of the declared calibration grid, so the search did not bracket an optimum.
More importantly, a post-run static diagnostic over eight absolute headings
found the sign of `left-right` correct for only 44/64 nonzero relative goal
errors (68.8%). At zero relative error, the raw offset ranged from about
-0.0040 to +0.0095 depending on heading. A single global adapter bias cannot
remove that heading-dependent error. A leave-one-heading-out check found that
an affine readout of both pooled rates improved sign prediction only slightly
(62.5% to 64.6% on that smaller grid), so adapter complexity is not the first
place to spend compute. These are screening diagnostics, not held-out episode
results or a replacement for the declared evaluation.

Proceed in this order and stop at the first step that beats 20/100:

1. **Finish bracketing the existing adapter.** Predeclare a narrow extension:
   gains `{48, 64, 96}` and biases `{-0.10, -0.05, 0}`. Select on the 32
   training scenarios by arrivals, then mean return, then lower `abs(bias)`,
   then lower gain. Run only the winner on the 100 held-out scenarios. Run the
   full expensive control suite only if that frozen candidate beats 20.
2. **Validate the output interface before changing dynamics.** The current
   PFL3 split uses `somaSide`; the cited circuit defines left/right by the LAL
   projection side. Confirm the MaleCNS mapping from anatomical evidence and
   change it only if that evidence shows misclassified neurons. A quick PB-side
   proxy performed worse than the current split, so do not substitute it.
3. **If still at or below 20, test one declared input-registration slice.** The
   published PFL3 model includes a fixed compass/FC2 phase offset and a relative
   heading-versus-goal input-strength parameter. This implementation uses one
   shared `input_gain` and documents an unresolved phase convention. Screen
   phase, goal/heading gain ratio, and `recurrent_gain` on a static heading/error
   grid; send only the best two candidates through the 32 training episodes.
   Coarse checks so far put the current phase, equal cue gain, and
   `recurrent_gain=1` near the best sign accuracy, so do not launch a broad grid.
4. **Stop if the two pooled rates remain frequently sign-wrong.** More scalar
   gain, bias, smoothing, or a nonlinear adapter cannot repair wrong-direction
   information. The next honest experiment is a separately labeled Stage 1b
   using column-resolved PFL3 activity or anatomically weighted LAL pools. That
   changes the current two-population-mean interface and must not replace or be
   merged into the original 20/100 claim.

Before another multi-hour run, make `run_controls` persist each completed
condition so interruption does not discard the entire run. Do not change arena
physics, target radius, frozen scenarios, graph size, or give the adapter target
geometry merely to raise the score.

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
- **Phase 4: DONE ✓** (see results table above). Reproduce with:
  `evaluate.py --controls --data data/malecns-v1.0
  --checkpoint runs/stage1/checkpoint.json --scenarios runs/stage1/heldout.json
  --calib-scenarios runs/stage1/training_scenarios.json
  --out runs/stage1/results.json --jobs 8`. Harness (neural + baseline +
  zero/random + cue-withheld + pathway-silenced + shuffled×3-recalibrated) is
  unit-tested and now run against the real graph.
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
| `runs/stage2/` | `training.json` (24), `heldout.json` (100) committed; `checkpoint.json` + `results.json` from the real-graph run; `dev_split.json` (100) + `dev_bracket*.json` from the direct-compass screening gate; `training_diagnostic.json`; `README.md` (results + gate write-up). |

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
from `runs/stage1/checkpoint_corrected_epg.json`. Calibration selected
`avoidance_gain=1.0`, `brake_distance=6.0` (7/24 training arrivals). The Stage 2
checkpoint hashes (SHA-256) the Stage 1 checkpoint and both frozen scenario
files; `--controls` refuses a mismatch:

- Stage 1 checkpoint: `2294a104bceb2424c4cd949a8ff42762398f6acc3679bb6962e48c755e9536d1`
- training scenarios: `adf270300f9dcf552550f483848146021ab45c9cc49e7f382a865899d9caac67`
- held-out scenarios: `1329165ad200b216c7ec81a9aeb10839f8ae3fa697e66a68388ea2560ecd83f3`

**Controls (all reuse the selected adapter, no recalibration):** `neural`,
`direct_compass` (labeled non-neural baseline, reads angular error),
`sensor_only`, `neural_no_sensors` (ranges forced to max), `goal_cue_withheld`,
`pfl3_silenced`.

**Held-out results (2026-09-15).** Artifacts:
`runs/stage2/checkpoint.json`, `runs/stage2/results.json`.

| control | trials | arrivals | collisions | timeouts | mean arrival time | mean route length |
|---|---:|---:|---:|---:|---:|---:|
| neural | 100 | **33** | 65 | 2 | 25.30 s | 31.85 |
| direct compass | 100 | 41 | 58 | 1 | 22.88 s | 30.39 |
| sensor only | 100 | 19 | 81 | 0 | 20.81 s | 29.28 |
| neural, no sensors | 100 | 7 | 93 | 0 | 13.83 s | 26.19 |
| goal cue withheld | 100 | 14 | 82 | 4 | 19.55 s | 27.99 |
| PFL3 silenced | 100 | 14 | 86 | 0 | 19.39 s | 27.90 |

Neural arrivals by layout were cross 6/12 (50%), regular 15/44 (34.1%), and
asymmetric 12/44 (27.3%). The control suite took approximately 8 h 30 min
(checkpoint-to-results timestamps, serial). Exact per-control wall time and peak
RSS were not captured: macOS `/usr/bin/time -l` completed the child but failed
its own `sysctl kern.clockrate` query in the sandbox. Do not rerun 600 expensive
episodes solely to recover accounting metadata.

**Stage 2 exit condition: NOT MET.** The neural controller materially beats
`neural_no_sensors` and both neural-pathway ablations, so the connectome signal
still contributes. It does not yet provide reliable road following or
intersection choice: 65% of neural trials collided, every layout was below 51%,
and even the exact direct-compass baseline collided in 58/100 trials. This is a
Stage 2 navigation-adapter failure, not evidence that the project or corrected
Stage 1 steering result is fundamentally unsound.

**Post-run diagnosis.** These are retrospective diagnostics, not new held-out
selection results:

- The declared search did not bracket `avoidance_gain`: arrivals increased
  monotonically from 1/24 at gain 0 to 7/24 at the maximum gain 1.0.
  A cheap direct-compass replay on the training set reached 13/24 at gains
  1.5–2.0, then plateaued; gain alone is insufficient.
- `brake_distance` was effectively unidentifiable: it did not change
  arrival/collision counts within any declared gain row. In direct-compass
  training replays, 13/16 collisions occurred below speed 0.25 (mean terminal
  speed 0.20). Cars creep into obstacles after becoming trapped; excessive
  approach speed is not the dominant failure.
- The current adapter is a stateless potential-field rule. On the held-out
  topology, sensor-only reached 0/58 turn-required cases and 0/64 cases whose
  straight start-to-target segment crossed a building. It can avoid some walls
  but cannot commit to an intersection turn, follow a wall through a local
  minimum, or recover once pointed into a wall.

**2026-09-16 diagnostic review gate.** `diagnose_stage2.py` and
`runs/stage2/training_diagnostic.json` reproduce the selected neural controller
on all 24 training scenarios: 7 arrivals, 17 collisions, 0 timeouts. The useful
confirmed result is narrower than the first interpretation:

- Slow trapping is real: mean collision speed is 0.119 and 16/17 collisions
  occur below speed 0.25. More braking is not the next lever.
- The current `turn_required` flag is not an intersection-route classifier. It
  means line of sight is blocked **or** initial heading error exceeds 45 degrees,
  so it labels 22/24 scenarios. By route topology, the 17 collisions split into
  8 same-axis routes and 9 routes requiring an intersection turn. Do not claim
  that intersection turns are the entire failure mode.
- Steering dominance is correlational, not directional. The existing metrics
  do not establish that the neural/goal term points into the obstacle or that
  avoidance points toward open road. The reported 33%/62% avoidance-dominance
  and 42%/59% opposition values are means of per-episode fractions, not literal
  fractions of all steps. Step-weighted values are 38%/65% and 47%/57% for
  collisions/arrivals, respectively.
- Range rays originate at the vehicle center. The approximately 0.76 value is
  minimum forward sensor range, not nose-to-wall clearance; collision uses a
  separate 0.5-unit footprint.

Before treating commit-and-hold as more than a hypothesis, make the diagnostic
artifact self-auditing and correct the analysis:

1. Separate evaluator-only categories for initial reorientation, blocked line
   of sight, and an actual intersection turn based on route topology.
2. At low forward clearance, record whether each steering contribution points
   toward the more-open or more-blocked side; add step-weighted outcome-group
   summaries and label per-episode means explicitly.
3. Embed the Stage 2 checkpoint hash, source paths/hashes, and data-manifest
   provenance in the output artifact.
4. Add one controller-parity test proving instrumentation returns the same
   controls as `Stage2NeuralController`, plus one aggregation test covering the
   reported group metrics. The existing five tests cover only geometry helpers.

**Next steps, in order.** Do not spend another multi-hour neural run until the
cheap direct-compass gate passes.

1. Correct the existing training diagnostic to clear the review gate above.
   Keep all geometry-derived classifications evaluator-only; do not expose them
   to the controller.
2. Replace the inert braking-only degree of freedom with the smallest
   sensor-only recovery behavior: commit to the more-open turn when forward
   clearance is low and retain that choice with one bit of state/hysteresis
   until the forward lane reopens. Continue to use only bearing, ranges, and
   speed; do not add a map or route planner.
3. Bracket avoidance on the cheap direct-compass controller (start with gains
   1.0, 1.5, 2.0, 3.0). Require a predeclared reliable threshold across all
   three layouts before reconnecting the real graph. Recommended gate: at least
   90/100 overall and at least 80% in each layout on a newly frozen development
   split.
4. After the direct-compass gate passes, freeze a fresh 100-scenario confirmation
   set because the present held-out set has now informed design. Calibrate only
   the minimal adapter parameters on training, run neural confirmation once,
   and run the six expensive controls only if the frozen neural candidate clears
   the predeclared gate.

### 2026-09-16 outcome: recovery mechanisms built, gate NOT passed (74% ceiling)

Steps 1–3 above are done; the direct-compass gate did **not** pass, so step 4
(the expensive neural run) stays **blocked**. Full write-up + provenance in
`runs/stage2/README.md`.

- Step 1: `diagnose_stage2.py` + `test_diagnose_stage2.py` clear the review gate
  (evaluator-only route-topology classes, step-weighted metrics, embedded
  checkpoint/source hashes, controller-parity + aggregation tests).
- Step 2: `Stage2Adapter` gained opt-in **commit-and-hold** recovery
  (`commit_distance`/`release_distance`/`recover_speed`, one bit of state,
  hysteresis; disabled-by-default is byte-identical to the old behavior).
- Step 4 extension: opt-in **goal-aware commit** (`goal_clearance`) — commit
  toward the side the brain/compass already wants (sign of the neural steer
  term), if that side is at least `goal_clearance` open, else the most-open
  side. Reads no geometry; the `__call__` signature and its no-geometry
  guardrail test are unchanged.
- Step 3 gate: a predeclared **64-candidate** bracket
  (`avoidance_gain ∈ {1.0,1.5,2.0,3.0}` × `commit_distance ∈ {1.5,2.5}` ×
  `release_margin ∈ {1.5,3.0}` × `recover_speed ∈ {1.0}` ×
  `goal_clearance ∈ {0.0,0.3,0.5,0.7}`) on the cheap direct-compass controller
  over a fresh 100-scenario dev split
  (`dev_split.json` sha256 `1a3682705…2bb2c3db`), gate `>= 0.90` overall and
  `>= 0.80` per layout. **0/64 pass.** Best is **0.74** (cross 0.83 / regular
  0.75 / asym 0.70) at `goal_clearance = 0.0` (pure openness). Goal-awareness
  did not raise the ceiling (best `gc>0` = 0.73). Artifact:
  `dev_bracket_goalaware.json`.

**Honest framing (do not overstate).** `direct_compass` is a cheap **screening
baseline**, not a mathematical upper bound on the connectome. A failed
64-candidate bracket screens out **these three escape-triggered variants**; it
does **not** prove every map-free reactive controller is exhausted. All three
variants fire only *after* forward blockage, so they test escape behavior, not
proactive intersection navigation (detect a lateral opening and begin the turn
before becoming trapped). Stop extending `Stage2Adapter`.

**Next (separately named Stage 2b experiment; must not merge into the 33/100
Stage 2 result).** Two cheap observation-boundary baselines to localize the
missing component, both scored on the same 90%/80% gate:
1. An **evaluator-only waypoint controller** (privileged map/route/target/pose;
   a solvability **witness** — its success shows the vehicle + layouts admit a
   solution; a failure would not prove a scenario unsolvable).
2. A **conventional observation-only state machine** (FOLLOW / TURN_LEFT /
   TURN_RIGHT) using exactly `heading`, `goal_bearing`, `ranges`, `speed`, that
   detects lateral openings and begins turns *before* becoming trapped.

Interpretation (bounded): the two controllers differ in map, route, and pose,
not only in memory, so a waypoint-vs-SM gap does not by itself isolate one cause.
Waypoint success establishes solvability; a competent SM falling short motivates
route/history memory as the next hypothesis (tested by a same-observation
recurrent baseline, below). Only after that consider biologically grounded
learning, with dopamine as a teaching/modulatory signal — **not** as goal
bearing.

### 2026-09-16 Stage 2b: baselines built, eligible stratum defined

New files (non-neural; do not touch the connectome or frozen Stage 2 artifacts):
`stage2b.py` (`WaypointController`, `ObservationStateMachine`,
`is_outward_road_end`, `plan_route`), `evaluate_stage2b.py`, `test_stage2b.py`
(12 tests). Full write-up: `runs/stage2b/README.md`.

**Full seed-7 dev split (preserved):** waypoint **84/100**, observation-only SM
**32/100**.

**Post-hoc diagnosis (not an impossibility claim).** All 16 waypoint failures
are **outward-facing road-end starts** — a car placed at a road's arena-edge
endpoint (`(x, ±21)` on an x-road or `(±21, y)` on a y-road) facing *outward*.
`stage2b.is_outward_road_end` flags exactly those 16 (precision and recall
1.0). Such a start must reverse to reach any interior target, but the full-lock
turn radius (1.73) sweeps 3.46 laterally in a 6-wide lane, so a naive route
follower cannot U-turn in place there. This is a lane-width vs turn-radius
interaction, **not** proof the starts are physically impossible, and reversed
starts that are *not* outward road-ends succeed (10/10 reversed non-endpoint
starts arrived).

**Eligible stratum (exclude only outward-facing road-end starts):**
- Waypoint: **84/84 = 100%**.
- Observation SM: **32/84 = 38.1%** (per layout 60.0% / 36.8% / 33.3%).

**Fresh frozen splits (forbid outward road-end starts), disjoint by scenario
identity from the inspected seed-7 dev split and from each other:**
- `runs/stage2b/gate_split.json` (100; seed 21) — the one-shot gate.
  sha256 `3f7863a68261f5ee…`. The waypoint witness scores **100% (PASS 90%/80%)**
  here, establishing solvability of the eligible stratum.
- `runs/stage2b/sm_train_split.json` (100; seed 22) — SM tuning only. sha256
  `8df212c4b2700591…`.

**Step 5 result (2026-09-16): SM tuned on training, one-shot gate = 62% (fails).**
`ObservationStateMachine` was reworked (corridor centering on the 90+45deg side
rays; commit a 90deg turn at a goalward opening) and tuned with a 144-point sweep
on `sm_train_split.json` **only** (grid over opening/forward_block/turn_speed/trigger45/turn_delay; best
training config is now the class defaults, training arrival 0.70). The inspected seed-7 dev split
and the gate split were never used for tuning. One-shot on `gate_split.json`
(`runs/stage2b/gate_results.json`, sha256 `b190f38f…`):

| controller | overall | cross / regular / asym |
|---|---:|---|
| waypoint (privileged map/route/pose) | **1.00 PASS** | 1.00 / 1.00 / 1.00 |
| observation-only SM | **0.62 fail** | 0.92 / 0.50 / 0.66 |

The witness passing at 100% establishes the eligible stratum is solvable. The
best of this **144-configuration SM family** — observation-only, no
route/topological memory — reaches 62% and degrades with intersection count
(cross 92% → regular 50%), consistent with greedy per-intersection turns made
without route memory. Because the waypoint controller also has privileged
layout/route/target/pose, the 100%-vs-62% gap does **not** by itself localize the
deficit to memory (the two differ in map, route, and pose too); it establishes
solvability and motivates route/history memory as the next hypothesis.

**Next gate (before any dopamine/connectome learning).** Use a same-observation
**recurrent / history** baseline — the same `heading`, `goal_bearing`, `ranges`,
and `speed`, still with no map or pose — against frozen `gate_split.json`.
Clearing 90%/80% would be evidence that this observation history is sufficient
and memory was the missing piece for this controller family. A failure is
inconclusive between model capacity, training, and observation limits; diagnose
it before proceeding. Only then consider biologically grounded learning, with
dopamine as a teaching/modulatory signal, **never** as goal bearing. Do not bolt
a map/planner onto the SM — that just re-derives the waypoint witness.

## Git

- `master`: Phases 1–2 slice merged.
- `stage1-prepare-connectome`: connectome pipeline + anatomy angle mapping
  (this handoff lives here; merge to master when integrating).
- Working-tree note (2026-09-13): `runs/stage1/results.json` is present but
  untracked; add it before claiming the result artifact is committed.

Reproduce data with SETUP.md — nothing under `data/` is versioned.
