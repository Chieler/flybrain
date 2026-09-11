# Flybrain Simulation — Proposed Design and Implementation Plan

**Superseded:** The active direction is now a car navigating a simulated city. See the [three-stage roadmap](../../../ROADMAP.md) and [Stage 1 driving plan](2026-09-11-stage-1-driving.md). This earlier food-target proposal is retained for context.

**Status:** Proposal only. No implementation or dataset download has started.

**Goal:** Let a simplified computational model built from the mapped male fruit fly brain control a virtual fly seeking a reward in a small 2D arena, with low runtime overhead and measurable evidence of the brain model's contribution.

**Architecture:** Local Python simulation, a sparse recurrent neural model, a small motor-output adapter, and an optional lightweight display. Use the real male connectivity while explicitly distinguishing measured anatomy from assumed neural dynamics and trained interface behavior.

**Tech stack:** Python, NumPy, SciPy sparse matrices; PyArrow for importing the published Feather tables; pygame for an optional local viewer. Confirm supported versions when implementing and pin the tested environment. No GPU requirement for the first experiment; actual whole-brain performance must be measured.

**Spec:** The design below is the proposed specification, pending user review. It does not authorize implementation. Execute inline when authorized; no subagents are required.

## Design

### What the user sees

A top-down arena containing a small fly, a visible food beacon, and the fly's recent trail. The fly starts at randomized positions and headings. Entering the food region counts as collection and ends the episode. The display includes selected brain-region activity, episode reward, success rate, and simulated seconds per wall-clock second. Controls: run/pause, reset, relocate food, playback speed, and switch between untrained and trained adapters. Moving food starts a new evaluation episode so it does not corrupt a running trial.

Use the user's self-flying option: a virtual animal moving through an abstract arena. The first version has no weapon model, realistic vehicle dynamics, wing biomechanics, obstacles, or 3D renderer.

### Which brain and which model

Use MaleCNS v1.0, which contains the male brain and ventral nerve cord. Preserve all annotated brain neurons, including optic lobes, rather than silently substituting the female FlyWire dataset or an invented small network. Use descending neurons as the boundary for artificial movement control; the ventral nerve cord and muscles are outside the first embodiment. Record the exact inclusion rules and retained neuron/edge counts. Report connections cut at this boundary.

The MaleCNS project supplies connectivity and annotations; it does not itself establish a validated, reward-seeking executable animal. Shiu et al.'s published FlyWire model is a useful reference for assumptions and checks, but adapting its method to male data is a new model, not a validated drop-in replacement.

Three implementation choices:

| Choice | Benefit | Limitation |
| --- | --- | --- |
| Entire male brain, rate dynamics — recommended default | Keeps individual mapped neurons and wiring; relatively cheap updates | Approximate activity, no individual spikes |
| Identified navigation circuit, rate dynamics | Lowest compute | Represents a subset, not the whole brain; extraction can remove important context |
| Entire male brain, integrate-and-fire dynamics | Closer to the published modeling approach | More temporal detail and likely higher runtime cost; still not a complete biological model |

### Neural computation and interfaces

Represent connections as a directed sparse matrix, aggregating all recorded synapses between the same neuron pair without discarding their total weight. Maintain one float32 activity value per neuron. Apply leaky rate updates with a bounded response and explicitly calibrated global gain and time constant. Synapse counts initialize relative weights, not measured physiological conductances.

Use transmitter predictions only as documented modeling assumptions. A transmitter label alone does not uniquely determine every postsynaptic effect. Track prediction confidence and unknown labels; do not silently classify unknowns as excitatory. Resolve a declared fallback and test its sensitivity during model calibration. Do not include detailed neurotransmission, dendrites, or synaptic plasticity in this initial model.

The arena supplies a small body-relative visual observation, initially 32 angular intensity bins, plus a simple self-motion signal. It never supplies target coordinates, distance-to-target, or the correct turn to the controller. The evaluator may use world geometry for scoring. Connecting these bins to identified male visual pathways is a research task: preserve lateralization and verify annotation evidence before selecting neuron IDs. If this cannot be supported, explicitly label the input mapping as an artificial interface.

Pool activity from a small documented set of descending output populations. A bounded linear adapter maps only those activities to turn and forward movement. The adapter has no direct sensory or world-state shortcut. Use abstract kinematic movement with speed and turn limits; no aerodynamic model.

The closed loop is: arena observation → sensory mapping → recurrent male brain model → output adapter → movement → next observation. Food collection returns an episode reward to the training procedure.

### Learning and interpretation

First characterize the fixed brain model and an untrained adapter. Then optimize only the small adapter using episodic reward and a simple derivative-free search. Freeze and save the brain parameters before adapter training; use seeded, paired evaluations of candidate adapters.

Start with nearby food and varied headings, then expand start distances. Report terminal collection reward and time cost separately. Avoid distance-shaped training rewards in the initial experiment so the controller cannot rely on privileged positional information. If sparse reward defeats this method, report that finding before changing the learning objective.

This is learning at an artificial motor interface. It is not evidence that the fly brain itself learned a new reward association. Biologically motivated plasticity in identified learning circuits is a separate extension, with separate validation. Likewise, successful navigation alone does not demonstrate consciousness, natural flight, or a faithful digital animal.

### Compute budget

Download only neuron annotations, neuron-level transmitter predictions, and aggregated connection weights. The official page lists the full weight table at approximately 1.1 GB; it includes more segments than the final selected neuronal graph. Avoid EM volumes, neuron meshes, and individual synapse-coordinate tables.

Use a one-time conversion to a cached sparse representation, followed by offline execution. Float32 weights plus int32 indices cost approximately 8 bytes per retained directed edge, plus row pointers and state. For illustration, 25 million edges would require roughly 200 MB for those matrix arrays alone; this is not a verified count for the selected v1.0 brain graph. Import and runtime peak memory will be higher.

Each neural update costs O(neurons + retained edges). Start by benchmarking 100 model updates per simulated second, with a separate 30 FPS display and rendering disabled during training. These rates are engineering starting points, not claims of biological timing fidelity. Check sensitivity to the integration step. Aim for at least real-time execution on the user's machine; do not promise it before measurement. If it misses the budget, present measured alternatives: slower playback, an explicitly reduced circuit, or a more efficient numerical kernel.

## Implementation sequence

The workspace is currently empty apart from local tooling configuration. Proposed files: `prepare_connectome.py`, `brain.py`, `simulation.py`, `train.py`, `viewer.py`, `test_simulation.py`, `requirements.txt`, `README.md`, and a dataset/checkpoint manifest under `data/`. Generated data is excluded from source control. Split files further only when the actual code warrants it.

### 1. Resolve the data and anatomy

- [ ] In `prepare_connectome.py`, pin MaleCNS v1.0 download URLs, record checksums and licensing, and import only the three required tables.
- [ ] Verify schema, direction, stable neuron IDs, endpoint membership, finite nonnegative connection counts, transmitter confidence, and annotation coverage. Count excluded fragments and brain-boundary edges.
- [ ] Produce a sparse matrix, ID lookup, annotations, and manifest; preserve original counts separately from any normalized model weights.
- [ ] Identify defensible sensory and descending output populations. Record the evidence, uncertainties, and whether each interface is biological or artificial.
- [ ] Validate a tiny hand-checkable connection table and compare retained counts with the source tables.

**Exit condition:** Reproducible data conversion and documented input/output mappings. If key mappings cannot be established, report the limitation rather than invent biological claims.

### 2. Establish a stable neural runtime

- [ ] In `brain.py`, implement reset, one rate update, sensory stimulation, and selected-population readout using the cached matrix.
- [ ] Calibrate gain and decay against spontaneous and stimulated activity; detect silence, global saturation, nonfinite activity, and excessive sensitivity to the time step.
- [ ] Measure load time, peak memory, update throughput, and a short steady-state runtime on the target machine.
- [ ] Add a small runnable check for directed propagation, assumed inhibitory effects, deterministic reset, and finite activity. Keep all checks in one test file.

**Exit condition:** Stable, responsive model with a measured compute budget. Stability alone does not establish biological validity.

### 3. Close the loop in a headless arena

- [ ] In `simulation.py`, implement seeded episode reset, body-relative sensing, bounded movement, walls, collection, and timeout.
- [ ] Couple observations to the sensory mapping and neural outputs to movement. Keep evaluator geometry inaccessible to the controller.
- [ ] Verify collection boundaries, action limits, repeatability, and that changing the target cannot directly alter the adapter's inputs without passing through sensory and brain updates.

**Exit condition:** An untrained brain-controlled fly completes reproducible episodes and produces an observable trajectory, even if it fails to collect food.

### 4. Train and evaluate the motor adapter

- [ ] In `train.py`, implement a small episodic parameter search, using the same trial seeds for competing candidates and a fixed training budget.
- [ ] Store adapter weights, random seeds, model parameters, dataset checksum, and curriculum stage together.
- [ ] Freeze training and evaluate on 100 held-out episodes per training seed, using three independent training seeds. Report collection rate, median collection time among successes, and episode return with uncertainty.
- [ ] Compare against random movement, a simple direct-sensor controller, and a decoder using shuffled-connectivity activity. Retrain the shuffled control under the same budget; additionally test acute silencing of the trained model's relevant pathways.
- [ ] Keep the same input/output mappings for the graph comparison. Do not infer causality from one silencing result alone.

**Exit condition:** Reproducible evaluation, whether positive or negative. Proposed behavioral target: at least 80% collection on the held-out easy arena and clear improvement over random movement. A connectome-specific contribution requires evidence from the controlled comparisons; meeting the behavioral target alone is insufficient.

### 5. Add the viewer and handoff

- [ ] In `viewer.py`, show the arena, trail, selected activity, reward, and measured simulation speed, with pause/reset and episode-safe food relocation.
- [ ] Decouple drawing from neural updates. Headless and displayed runs with the same seed and actions must agree.
- [ ] In `README.md`, document setup, data provenance, preparation, training, evaluation, display, actual performance, and every biological approximation.

**Exit condition:** One local demo command and one reproducible headless evaluation command, with no network calls after preparation.

## Sources

- [MaleCNS project and release context](https://male-cns.janelia.org/)
- [MaleCNS v1.0 downloads, annotations, and connection weights](https://male-cns.janelia.org/download/)
- [Google Research: male brain and CNS reconstruction](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [Shiu et al. reference simulator: FlyWire connectivity and Brian 2](https://github.com/philshiu/Drosophila_brain_model)
