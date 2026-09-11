# Flybrain — Stage 1 (partial: arena + neural runtime)

A car steers around an open 2D arena toward a destination. Long-term goal: drive
the steering with a rate model over the mapped **male** fruit-fly connectome. See
[ROADMAP.md](ROADMAP.md) and the
[Stage 1 plan](docs/superpowers/plans/2026-09-11-stage-1-driving.md).

## What is built (this slice)

Everything that does **not** depend on the connectome download or the biological
interface mapping — the plan's Phase 2 "prove the arena and neural runtime
independently":

- `simulation.py` — kinematic bicycle car, swept target/boundary termination,
  seeded episode loop, headless benchmark. World frame: +x east, +y north,
  counterclockwise-positive angles.
- `brain.py` — bounded rate model over a sparse connectome (row-normalized,
  global recurrent gain), circular cue encoding, pooled left/right output, and a
  geometry-blind steering `Adapter`. Runs on any sparse matrix; validated on a
  tiny synthetic graph.
- `evaluate.py` — seeded held-out scenario generation + validation, the
  **conventional compass baseline** (separately labeled; reads angular error),
  and metrics (arrivals/boundaries/timeouts with denominators, returns).
- `test_stage1.py` — 15 deterministic checks (`python -m unittest test_stage1`).

## What is NOT built (blocked)

These need decisions/inputs beyond code and are deliberately absent — no
fabricated neuron IDs or preferred angles:

- **`prepare_connectome.py`** — download/validate/convert MaleCNS v1.0
  (~1.1 GB). Blocked on authorizing the download + license.
- **Biological interface mapping** — evidence-backed male neuron IDs for
  heading / FC2 goal / PFL3 left–right steering with preferred angles. The plan
  says to stop and report if the data cannot support this; it is unresolved.
- **Neural calibration + controls** (`evaluate.py`) — adapter gain/bias grid,
  cue-withheld / pathway-silenced interventions, shuffled-connectivity control.
  All depend on a real graph + mapping.
- **`viewer.py`** — Pygame demo (Phase 5).

## Commands (working today)

```sh
python -m unittest test_stage1
python simulation.py --benchmark --seconds 60      # physics-only timing
python evaluate.py --gen-scenarios 100 --seed 1 --out scenarios.json
python evaluate.py --baseline --scenarios scenarios.json
```

The conventional baseline reaches 100/100 generated targets, confirming the
car/target setup is navigable before a brain is wired in. This is a baseline
result, **not** a neural-steering result.

## Honesty boundary

Measured wiring, assumed neuron dynamics, artificial sensory/motor interfaces,
and any learned parameters are kept distinct. Nothing here demonstrates that fly
connectivity contributes to steering — that claim requires the blocked pieces
plus the controlled comparisons in the plan.
