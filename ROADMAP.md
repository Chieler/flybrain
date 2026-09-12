# Flybrain: a fly brain learns to drive

**Status:** Stage 1 open-arena implementation exists; neural evaluation is in
progress, and its acceptance target has not yet been met.

## Goal

Use a computational model based on the mapped male fruit fly brain to steer a car from A to B in a simulated city. The car receives its current heading, a compass bearing toward the destination, and eventually local road/obstacle observations. Visitors can choose destinations, change the environment, and inspect the modeled brain's response.

Use the male connectome as the recurrent neural network. Keep a clear distinction between measured wiring, assumed neuron dynamics, artificial sensory/motor interfaces, and any learned parameters. The compass does not supply a route or intermediate waypoints.

## Stage 1 — Open driving area

**Question:** Can heading and goal-direction signals passing through the male brain model produce useful car steering?

- Flat, open 2D arena with a visible car, start point, destination, compass, and trajectory.
- Fixed low forward speed; neural output controls steering. Entering the destination region ends the trial. This is not yet a demonstration of braking or parking.
- Inject heading and goal-bearing cues into documented neural populations; no obstacle sensing is required in the open arena.
- Establish reproducible left/right steering, destination reaching, and response to a relocated destination.
- Compare with a conventional compass controller and interventions on neural signals. Report throughput and memory alongside behavior.

**Proposed exit criteria:** At least 90% arrival on 100 fixed held-out open-arena trials, documented neural-interface provenance, a measured runtime budget, and recorded cue/pathway intervention results. Performance targets are engineering acceptance criteria, not existing results or proof of biological fidelity.

**Detailed draft:** [Stage 1 stack and implementation plan](docs/superpowers/plans/2026-09-11-stage-1-driving.md).

## Stage 2 — Small street grid

**Question:** Can the system follow roads and choose useful turns at intersections?

- Add buildings, road boundaries, and a small set of local distance sensors.
- Add acceleration/braking control as needed for turns and obstacles.
- Keep the destination compass and hide the map from the controller.
- Start with a small, fixed collection of reachable street layouts; introduce learning only with an explicit account of which parameters change.
- Evaluate arrival rate, boundary collisions, time, and route length on held-out start/destination pairs.

**Exit condition:** Reliable road following and successful intersection choices on the initial street layouts, with the learned adapter and neural model evaluated separately. Stage 1 results do not establish this capability.

## Stage 3 — Blocked streets and unseen layouts

**Question:** Can the system take detours, recover from wrong turns, and generalize?

- Introduce road closures, dead ends, and layouts withheld from training.
- Include routes that require temporarily moving away from the destination.
- Measure repeated loops, recovery time, collisions, route efficiency, and arrival rate.
- Compare against simple local controllers and use shortest routes as an evaluator-only reference.
- Let visitors close a road or change the destination and replay identical scenarios with neural interventions.

**Exit condition:** Reproducible evidence of detour/recovery behavior on unseen scenarios. Persistent loops are a recorded failure mode; adding a route planner would change the claim and must be identified explicitly.

## Shared constraints

- CPU-first, sparse computation; measure performance before adding acceleration infrastructure.
- Default to the entire annotated male brain, including optic lobes. A reduced circuit must be labeled and cannot silently replace the whole-brain objective.
- No external route planner in the neural controller and no privileged target geometry supplied directly to the motor adapter.
- Separate simulator state and scoring from controller observations.
- Seeded episodes, frozen evaluation scenarios, reproducible checkpoints, and declared model assumptions.
- Success alone does not prove the biological wiring contributes. Use controlled comparisons and report negative findings.

## Earlier proposal

This roadmap supersedes the earlier [virtual-fly food-target proposal](docs/superpowers/plans/2026-09-11-flybrain-simulation.md). That document remains as historical context, not the active implementation specification.
