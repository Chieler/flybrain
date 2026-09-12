# Setup — run Flybrain on another machine

Stage 1: a kinematic car steered by a rate-model neural network built from the
**MaleCNS v1.0** connectome (CC-BY 4.0). This guide gets a fresh machine from
`git clone` to a running neural episode.

## 1. Python + deps

Python 3.12–3.14. Create a venv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installs `numpy`, `scipy`, `pyarrow`, and `pygame` for the Phase 5 viewer.

## 2. Verify the code slice (no data needed)

```bash
python3 -m unittest test_stage1 -q     # deterministic checks, must be OK
python3 simulation.py --benchmark      # physics-only loop timing
```

These run without any download — they cover the arena, kinematics, rate
dynamics on a synthetic graph, scenario generation, and the transmitter-sign
policy.

## 3. Download the connectome (~1.1 GB, NOT in git)

`data/` is gitignored — the raw Feather tables and the generated matrices are
too big to version and are reproducible from the URLs below. Download the three
source tables into `data/malecns-v1.0/`:

```bash
mkdir -p data/malecns-v1.0
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
cd data/malecns-v1.0
curl -O $BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather      # ~14 MB
curl -O $BASE/body-neurotransmitters-male-cns-v1.0.feather            # ~41 MB
curl -O $BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather    # ~1002 MB
cd ../..
```

Release string: `male-cns:v1.0`. License: CC-BY 4.0.

## 4. Build the prepared graph

```bash
python3 prepare_connectome.py --data data/malecns-v1.0
```

Reads the three Feather tables, writes `counts.npz`, `weights_signed.npz`,
`ids.npy`, `manifest.json`. Expected output (recorded in the current
`manifest.json`):

- neurons retained: **191,148** (superclass not starting `vnc`)
- edges: **22,267,078** (32.8% of total synaptic weight; the rest lands on
  unannotated segments and is dropped — see `anatomical_weight_kept_fraction`)
- transmitter sign: pos 84,459 / neg 41,109 / zero 65,580
- interface: heading EPG n=46, goal FC2 n=92, PFL3 left/right 12/12

## 5. Run a neural episode

```bash
python3 -c "
import simulation as sim, brain as br, evaluate as ev
b = br.Brain.load('data/malecns-v1.0')
ctrl = br.NeuralController(b, br.Adapter(gain=1.0, bias=0.0))
sc = ev.generate_scenarios(1, seed=0)[0]
r = sim.run_episode(sc, ctrl, record=True)
print(r.outcome, round(r.elapsed_time,2), 's')
"
```

**Performance note:** the neural step is memory-bandwidth bound on the 22M-edge
matvec — roughly **14.6 ms/step, ~0.68× realtime** on the dev machine. A full
calibration grid (8 gains × 3 biases × 32 scenarios ≈ 768 episodes) is ~9 h at
this speed. This is a known wall, documented in HANDOFF.md; do not shrink the
graph to hit a time budget without a recorded decision (plan constraint).

## 6. Stage 2 — street grid (same data, same frozen Stage 1 checkpoint)

Deterministic checks need no download:

```bash
python3 -m unittest test_stage2 -q
```

Real-graph calibration then six-way held-out controls (reuses the frozen Stage 1
checkpoint; only `avoidance_gain` and `brake_distance` are tuned):

```bash
python3 evaluate_stage2.py --calibrate \
  --data data/malecns-v1.0 --stage1-checkpoint runs/stage1/checkpoint.json \
  --training runs/stage2/training.json --heldout runs/stage2/heldout.json \
  --checkpoint runs/stage2/checkpoint.json

python3 evaluate_stage2.py --controls \
  --data data/malecns-v1.0 --stage1-checkpoint runs/stage1/checkpoint.json \
  --checkpoint runs/stage2/checkpoint.json --heldout runs/stage2/heldout.json \
  --out runs/stage2/results.json
```

One full-graph Stage 2 episode ≈ **47 s**; calibration is 288 episodes and
controls is 600 episodes, so budget ~11–12 h. `--controls` verifies the recorded
SHA-256 of the Stage 1 checkpoint and both frozen scenario files, and refuses a
mismatch with a nonzero exit. Replay a frozen scenario with
`python3 viewer.py --stage 2 --controller baseline --scenarios runs/stage2/heldout.json --index 0`.
