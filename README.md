# Flybrain — Stage 1 open arena

A kinematic car receives heading and goal-bearing cues and is steered by a
rate-model neural network over the full prepared MaleCNS v1.0 connectome. See
[ROADMAP.md](ROADMAP.md) and the [Stage 1 plan](docs/superpowers/plans/2026-09-11-stage-1-driving.md).

## Current status

The connectome pipeline, arena, neural runtime, adapter calibration, controls
harness, and pygame replay are implemented. The frozen 32-scenario calibration
selected gain 32 and bias -0.05, with 7 arrivals; this is not evidence that the
Stage 1 90/100 held-out target has been met. The 100-trial neural/control run is
in progress. The separately labeled conventional baseline reaches 100/100 on
the frozen held-out scenarios.

## Run it

Follow [SETUP.md](SETUP.md) to download and prepare the data, then run:

```sh
python -m unittest test_stage1 -q
python simulation.py --benchmark --seconds 60 --data data/malecns-v1.0
python viewer.py --controller neural --data data/malecns-v1.0 \
  --checkpoint runs/stage1/checkpoint.json
```

Generated calibration, scenario, trajectory, and control artifacts are under
`runs/stage1/`. The repository records the frozen calibration scenarios,
checkpoint, held-out set, and example trajectories for reproducibility.

## Integrity boundary

Measured wiring, assumed rate dynamics, artificial cue/motor interfaces, and
learned adapter values are kept separate. The neural controller receives only
heading, goal bearing, and pooled PFL3 outputs; it never reads target geometry.
The conventional compass controller is a labeled geometric baseline, not a
neural result. See [HANDOFF.md](HANDOFF.md) for limitations and the current
runtime budget.
