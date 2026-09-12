# Current takeover — Stage 1

## State at 2026-09-12

- MaleCNS v1.0 was prepared locally from the three Feather sources in `SETUP.md`.
  The raw `data/` directory is deliberately not versioned; download and run
  `python prepare_connectome.py --data data/malecns-v1.0` on the new machine.
- Frozen calibration used 32 training scenarios.  The best declared-grid
  adapter is **gain 32, bias -0.05**, with **7/32 arrivals**.  This is an
  experimental result, not a passing navigation controller.
- The conventional baseline reaches **100/100** on the frozen held-out set;
  therefore the arena/scenarios are navigable and the shortfall is neural
  control, not the task geometry.
- Versioned artifacts: `runs/stage1/checkpoint.json`,
  `training_scenarios.json`, `heldout.json`, and `example_trajectories.json`.

## Active work

This machine is currently running the real-graph controls evaluation:

```bash
.venv/bin/python evaluate.py --controls \
  --data data/malecns-v1.0 \
  --checkpoint runs/stage1/checkpoint.json \
  --scenarios runs/stage1/heldout.json \
  --calib-scenarios runs/stage1/training_scenarios.json \
  --out runs/stage1/controls.json
```

It runs neural, baseline, zero/random, cue-withheld, pathway-silenced, and
three shuffled-connectivity controls.  It is CPU/memory-bandwidth bound on the
~20M-edge sparse matvec; expect roughly 27–35 CPU-hours total.  A running
process cannot move machines.  Do not run the same command concurrently unless
you intentionally want a separate duplicate result; it writes `controls.json`
only after completion.

## Next decision after controls

Inspect the control report and the steering-versus-goal-bearing relationship.
The current 7/32 calibration does not meet the Stage 1 ≥90/100 held-out target.
Before expanding the controller or tuning dynamics, record the decision and
retain the full connectome—do not shrink the graph merely to fit a time budget.

## GPU note

The current implementation uses SciPy CSR sparse matvec on CPU.  On the local
Apple M2, there is no installed supported GPU sparse backend, so GPU offload is
not a safe drop-in.  A CUDA/NVIDIA machine plus a separately validated sparse
backend is the credible acceleration path; do not interrupt the current run to
attempt a port.
