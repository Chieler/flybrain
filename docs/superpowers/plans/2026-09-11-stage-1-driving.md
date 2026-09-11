# Stage 1 Driving Implementation Plan

**Status:** Draft only; the user requested a saved roadmap and plan, not implementation.

**Goal:** Demonstrate a car steering toward a destination in an open arena using heading and goal cues processed by a male-connectome neural model.

**Architecture:** One local Python process loads a prepared sparse connectome, advances neural activity and simple car motion at fixed time steps, and optionally draws the result. A small steering adapter reads neural outputs; evaluation and rendering are separate from controller observations.

**Tech stack:** Python 3.12, NumPy, SciPy, PyArrow for preparation, and Pygame for the viewer. Select and pin compatible package versions during implementation after installation checks; no packages are installed by this plan.

**Spec:** [Project roadmap](../../../ROADMAP.md), Stage 1, together with the detailed decisions below. Future implementation should proceed inline through the checklist; no delegation is required.

## Scope and success

The first demo shows a car reaching a clicked destination in an open driving area. It establishes direction encoding, neural steering, reproducible simulation, and a useful visual demonstration. Street navigation, obstacles, distance sensors, braking, parking, and biological reward learning are outside this stage.

The car cruises at a fixed low speed. Arrival terminates the trial and freezes the display; do not describe this as the brain learning to stop. Clicking a new destination begins a new labeled episode from the current pose, resets neural/controller state, and clears that episode's metrics. A separate restart returns to the selected seed's initial pose.

Default to the complete annotated male brain, including optic lobes, within MaleCNS v1.0. The ventral nerve cord and muscles are outside the Stage 1 embodiment. Record the included/excluded neurons and connections at this boundary. Using the central complex for interfaces does not authorize extracting it as the entire simulated network.

## Stack decisions

| Component | Choice | Purpose |
| --- | --- | --- |
| Runtime and CLI | Python 3.12, standard library | One process; argparse, pathlib, JSON, hashing, timing, seeded experiment orchestration |
| Numerical state | NumPy float32 arrays | Neuron activities, angle encoding, trajectories |
| Connectivity | SciPy CSR sparse array | Directed sparse matrix-vector updates; cached NPZ storage |
| Dataset import | PyArrow | Read selected columns of published Feather tables without an additional dataframe layer |
| Viewer | Pygame, plain 2D drawing | Car, target, trail, compass, activity strip, controls and metrics |
| Checks | Standard-library unittest, one test file | Small deterministic numerical and behavioral checks |
| Artifacts | JSON manifests and NumPy/SciPy files | Reproduce dataset selection, tuning, episodes, and reported results |

Stage 1 is a local desktop demo. A browser deployment is a later packaging decision. Do not add a web backend, frontend framework, database, training framework, physics engine, GPU backend, or neural morphology renderer for this stage.

## Data preparation and biological interfaces

Pin MaleCNS v1.0 and obtain the official neuron annotations, neuron-level transmitter predictions, and aggregated connection weights. The official download page lists the full weight table at about 1.1 GB. Avoid raw microscopy, meshes, and individual synapse-coordinate tables. Record source URLs, locally computed SHA-256 hashes, source license, selected columns, and filtering rules. Locally computed hashes detect later changes; they do not substitute for publisher-supplied authenticity checks.

Keep two representations: unmodified retained anatomical synapse counts and the derived numerical weights used by the rate model. Use postsynaptic rows and presynaptic columns so `W @ activity` propagates activity in the recorded direction. Sum duplicate neuron pairs. Sort stable source IDs before indexing. Preserve neural IDs as IDs; contiguous array indices are a separate field.

Candidate interface roles are heading-associated central-complex populations, FC2-related goal populations, and left/right PFL3 steering output populations. These are candidates grounded in published fly navigation work, not verified male neuron assignments. Before implementation can claim a biologically grounded interface, produce a mapping manifest containing:

- Male neuron IDs, type, side, and anatomical column where relevant.
- Role: heading input, goal input, left steering output, or right steering output.
- Preferred direction for each input population, with its evidence and coordinate convention.
- Source references, confidence, and any cross-dataset anatomical correspondence.

Do not assign preferred angles by sorted neuron ID or assume anatomical columns have interchangeable tuning. Do not substitute female IDs. If published data and male annotations cannot support the mapping, stop that integration step and report the missing evidence. An explicitly artificial mapping is a possible revised experiment, not an unannounced fallback.

The steering signal may be read directly from mapped PFL3 output populations. This bypasses downstream motor mechanics through a declared artificial adapter; it does not simulate the entire natural motor pathway.

## Proposed neural model

Use one bounded rate value per neuron with fixed recurrent weights. The initial numerical hypothesis is:

```text
drive = clip(baseline + recurrent_gain * W @ activity + stimulus, 0, 1)
activity += (neural_dt / tau) * (drive - activity)
```

Start with `neural_dt = 0.01 s`, `tau = 0.05 s`, and baseline `0.01`. Treat these as tunable engineering assumptions. Normalize each nonempty incoming row by its total absolute weight, retaining a global recurrent gain. Test a small declared gain set against spontaneous and stimulated responses before freezing the model. Record normalization and selected values in the checkpoint.

Transmitter sign requires an explicit modeling policy. Proposed baseline: confident acetylcholine predictions contribute positive fast weights; confident GABA and glutamate predictions contribute negative fast weights. The glutamate sign is an approximation, not a universal receptor-level fact. Use confidence at least 0.5 where calibrated probabilities are provided. Uncertain and modulatory outputs remain in the anatomical graph but have zero fast recurrent weight in this first model; report their counts and affected edge weight. Test sensitivity to treating unknown outputs as positive and negative. If the source schema does not support that confidence interpretation, resolve and document the policy before producing W.

The initial model has no synaptic learning, receptor-specific dynamics, or individual spikes. Normalization, omitted fast effects, and input stimulation all limit biological interpretation. If direction-selective behavior cannot be obtained with stable documented dynamics, report the failed model; do not hard-code a turn into the sensory encoder to conceal it.

## Direction cue and steering

Use world coordinates with +x east, +y north, and counterclockwise positive angles. Convert to screen coordinates only when drawing.

The simulator computes current heading and `atan2(target_y - car_y, target_x - car_x)` as goal bearing. It encodes them separately into the documented neural populations, using a circular tuning function such as `max(0, cos(angle - preferred_angle))` with one recorded input gain per role. The signal is an externally supplied compass, not a visual perception system. Handle the exact-target position as arrival before computing or updating its cue.

The adapter sees only pooled left and right output activity. Start with `steering = clip(gain * (left - right) + bias, -max_steering, max_steering)`. Verify output-side interpretation experimentally against the mapping; do not infer motor direction from neuron names alone. Use population means so unequal population sizes do not introduce an unintended bias.

Calibrate only the adapter's gain and small bias using training episodes. Keep neural dynamics fixed after their own stability calibration. Neither the adapter nor any fallback controller in the neural run may read target coordinates, angular error, or simulator geometry. A conventional geometric controller is a separately labeled baseline.

## Car and arena

Use a simple kinematic bicycle model, with fixed speed and a bounded steering angle. Initial settings: wheelbase 1 unit, speed 2 units/s, maximum steering 30 degrees, car collision radius 0.5 units, target radius 1.5 units, and arena bounds [-25, 25] on both axes. Expose these few physical parameters in saved configuration.

```text
x_next = x + speed * cos(heading) * physics_dt
y_next = y + speed * sin(heading) * physics_dt
heading_next = wrap(heading + speed / wheelbase * tan(steering) * physics_dt)
```

Use `physics_dt = 0.02 s` and two neural updates per physics update, holding the current observation during those neural updates. Recompute cues on each physics update. Episode termination occurs on target entry, boundary exit, or 30 simulated seconds. Check swept movement against the target region and arena boundary, with the earliest event determining the outcome, so discrete steps cannot skip arrival or boundary contact. Define arrival as the car center entering the target radius and boundary failure using the car radius.

Draw at 30 FPS independently of simulation updates. Under load, reduce drawing frequency or run slower than wall time; never skip neural/physics updates silently. Pause must freeze simulation time. Headless mode uses the identical update loop.

## Calibration and evaluation

Reward is +1 for arrival, -1 for boundary failure, and a time cost of 0.01 per simulated second. Timeout receives only accumulated time cost. No distance-progress reward is needed for this open-arena calibration.

Use a deterministic small grid of adapter gains and biases, evaluated on 32 fixed training scenarios. Start with gains [0.25, 0.5, 1, 2, 4, 8, 16, 32] radians per unit activity and biases [-0.05, 0, 0.05] radians. Break equal-return ties by lower absolute bias, then smaller gain. This is adapter calibration, not a claim of learning inside the fly brain. Report exhaustion of this budget rather than expanding it silently.

Create and save 100 held-out scenarios before tuning: starts within [-8, 8] on each axis, target offsets of 6–15 units at uniform bearings, and uniformly varied initial headings. All sampled target regions must lie inside the arena; validation rejects invalid scenarios and records the accepted scenario list. Report results over all trials, including failures. Include explicit left, right, behind, near-wraparound, and already-arrived checks outside this random sample.

Comparisons:

- Conventional proportional compass steering using the same car constraints, scored on the same scenarios; clearly label its access to angular error.
- Zero-steering and seeded random-steering controls.
- The calibrated neural controller with the goal cue withheld, and with mapped output pathways silenced, without retuning.
- A shuffled-connectivity control retrained with exactly the same calibration budget and interfaces. Shuffle anatomical destinations while preserving source identity/sign and source weight multiset; describe which graph statistics this does and does not preserve. Use three fixed shuffle seeds.

Measure arrival rate, boundary/timeout counts, arrival time among successes, route length among successes, and elapsed runtime. Report numerator and denominator for every rate and avoid reporting successful-episode averages without failure rates. Neural interventions can support causal claims within the model; shuffled comparisons must not be presented as proof about a biological animal.

## File layout and interfaces

Keep source files at the project root until complexity warrants a package:

| File | Responsibility and boundary |
| --- | --- |
| `prepare_connectome.py` | Download/validate/convert MaleCNS tables; create matrix, ID lookup, and provenance manifest |
| `brain.py` | Load prepared graph, reset activity, encode heading/goal cues, advance neural state, return left/right population means |
| `simulation.py` | Seeded scenarios, car state, numerical loop, adapter, termination, headless CLI; no viewer import in headless execution |
| `evaluate.py` | Scenario files, adapter calibration, baselines/interventions, summary metrics and artifact saving |
| `viewer.py` | Pygame rendering and episode commands; imports simulation but contains no alternate controller or physics implementation |
| `test_stage1.py` | One small unittest module for the numerical and integration contract |
| `requirements.txt`, `.gitignore`, `README.md` | Pinned tested dependencies, generated-data exclusions, setup and truthful model/demo documentation |
| `data/`, `runs/` | Generated connectome cache, interface manifest, scenarios, checkpoints, and evaluation outputs; excluded from source control |

Within this structure, define a small `CarState` containing x, y, and heading; a `Scenario` containing initial state, target, and timeout; a `NeuralObservation` containing heading and goal bearing; and an `EpisodeResult` containing outcome, elapsed simulation time, path length, return, and optional trajectory. Use simple dataclasses where named state improves clarity. The brain consumes only `NeuralObservation`, the adapter consumes only two output rates, and the evaluator owns target geometry and rewards. Do not add controller plugin interfaces or an environment framework.

## Implementation checklist

### 1. Establish the model/data contract

- [ ] Validate the published table schemas and pin artifact URLs and release.
- [ ] Resolve the brain inclusion rule and candidate population/angle mapping using annotation evidence.
- [ ] Save the mapping and import manifest, including unknowns and exclusions; verify that every mapped ID belongs to the retained graph.
- [ ] Add a tiny connection-table check for duplicate aggregation, source/destination orientation, invalid values, and unknown endpoints.
- [ ] Implement cached conversion and validate matrix dimensions, finite weights, and declared sign handling. Never overwrite a completed cache until a replacement is fully written and validated; use a temporary file and atomic rename.

**Deliverable:** Reproducible input data with an evidence-backed interface map, or a specific documented mapping limitation. No neuron IDs or preferred angles are invented in this draft.

### 2. Prove the arena and neural runtime independently

- [ ] Implement and check straight movement, left/right turns, angle wrapping, swept termination, and deterministic scenario reset in the open arena.
- [ ] Run the conventional baseline to verify that the chosen car/target setup is navigable before evaluating the brain.
- [ ] Implement sparse rate updates and check directed activity propagation, finite outputs, and correct reset on a tiny graph.
- [ ] Calibrate spontaneous/stimulated response on the retained male graph. Detect total silence, pervasive saturation, and sign errors.
- [ ] Run a timestep comparison at 10 ms and 5 ms neural steps over the same simulated duration; investigate materially different steering responses.
- [ ] Benchmark at least 60 simulated seconds headlessly and record hardware, versions, selected graph counts, peak resident memory, and simulation-to-wall-time ratio.

**Deliverable:** Stable responsive neural runtime plus a separately verified open-arena simulator. Target at least 1 simulated second per wall-clock second and runtime peak memory below 4 GiB; these are provisional budgets to measure, not promises. Import peak memory is measured separately. A missed budget triggers a documented optimization decision, not silent graph reduction.

### 3. Couple cues, brain, and steering

- [ ] Wire heading and goal encodings to the mapped inputs and output population means to the bounded adapter.
- [ ] Check rotation consistency and left/right symmetry, allowing documented anatomical/model asymmetry.
- [ ] Confirm that changing target geometry with neural outputs held fixed cannot change steering: the adapter must have no geometry shortcut.
- [ ] Run the declared calibration grid, freeze the chosen parameters, and save provenance, seeds, and model settings.
- [ ] Produce untrained and calibrated example trajectories, including unsuccessful ones.

**Deliverable:** A reproducible brain-mediated steering controller or an explicit failed-model result. Do not replace it with the geometric baseline to pass the milestone.

### 4. Evaluate before polishing the demo

- [ ] Execute all held-out scenarios and the stated control/intervention runs under matching conditions.
- [ ] Write machine-readable summaries and trajectory artifacts, including failed episodes.
- [ ] Check deterministic replay within the tested environment. Preserve the scenario list and versioned parameters rather than assuming cross-platform bitwise determinism.
- [ ] Record whether the target of at least 90 arrivals out of 100 held-out episodes was achieved. Report neural contribution separately from task success.

**Deliverable:** Reproducible performance and intervention report. Failure of an acceptance target remains visible and prevents a claim that Stage 1 is complete.

### 5. Add the viewer and handoff

- [ ] Draw the open arena, car, target region, trajectory, heading/goal compass, left/right output activity, outcome, and measured simulation speed.
- [ ] Add pause, restart, destination click, and labeled baseline/neural/intervention selection. Mode changes start a fresh comparable trial.
- [ ] Verify that a scripted scenario produces the same numerical trajectory with and without drawing. Do one manual visual check of coordinate orientation and all controls.
- [ ] Document installation, preparation, calibration, evaluation, display, limitations, and observed performance. Include a short screen recording as the initial shareable demo artifact when implementation is complete.

**Deliverable:** Local interactive Stage 1 demo and reproducible evaluation commands. No hosted service is needed.

## Intended commands after implementation

These are proposed CLI contracts, not commands that exist today:

```sh
python prepare_connectome.py --release v1.0 --output data/malecns-v1.0
python -m unittest test_stage1
python simulation.py --benchmark --data data/malecns-v1.0
python evaluate.py --calibrate --data data/malecns-v1.0 --output runs/stage1
python evaluate.py --checkpoint runs/stage1/checkpoint.json --controls
python viewer.py --checkpoint runs/stage1/checkpoint.json
```

## Completion checklist

- [ ] Actual MaleCNS data and every artificial/assumed interface are documented.
- [ ] Fixed-speed car reaches at least 90/100 held-out targets using neural steering.
- [ ] All failures and controlled comparisons are reported; no unsupported claim that the connectome is superior.
- [ ] Headless and displayed loops agree; small runnable checks pass.
- [ ] Runtime and memory targets are measured and met, or the user has accepted an explicit revised scope/budget.
- [ ] Visitor can place a destination, pause/reset, and inspect the response.
- [ ] No streets, obstacles, route planning, braking/parking claims, or biological-learning claims have leaked into Stage 1.

## Sources

- [MaleCNS v1.0 data and schema descriptions](https://male-cns.janelia.org/download/)
- [Heading/goal comparison and FC2/PFL3 circuit research](https://www.nature.com/articles/s41586-023-07006-3)
- [SciPy sparse arrays and matrix-vector operations](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Pygame rendering and event loop documentation](https://www.pygame.org/docs/)
