# Stage 2c — same-observation recurrent/history baseline

The gate named at the end of Stage 2b, run *before* any dopamine/connectome
learning: does **temporal memory over the same observation interface** close the
0.62 → 0.90 gap the memoryless state machine could not?

**Non-neural** (in the connectome sense). Does not touch the connectome or the
frozen Stage 2 / Stage 2b artifacts. Files: `stage2c.py`, `evaluate_stage2c.py`,
`test_stage2c.py`.

## Design

- **Model** — a seeded Echo-State Network: a fixed random leaky-tanh reservoir
  (best config: 64 units, spectral radius 0.8, leak 0.5) with a
  **ridge-regression** linear readout to `(steering, acceleration)`. NumPy only,
  fully deterministic (`esn_seed=0`); no BPTT.
- **Interface** — sees exactly `heading, goal_bearing, speed, ranges`, same as
  the Stage 2b SM. `observation_features` uses **faithful** angles (heading and
  goal_bearing each as sin/cos, no engineered compass-difference term), so the
  net must learn the heading↔goal relation from temporal state. No map, no pose;
  `RecurrentController.__call__(self, obs)` (asserted in `test_stage2c`).
- **Labels** — behavior cloning of the Stage 2b `WaypointController` (the
  solvability witness). Its privileged route/pose/segment/carrot state generates
  target actions **only** and never enters the net's inputs. Because the teacher
  acts on that hidden state, its actions are *not* a pure function of the
  student's observation — a caveat that shapes the diagnosis below.
- **Matched ablation** — `recurrent=False` disables temporal state entirely
  (recurrent weights zero **and** unit leak → a static random-feature map of the
  current observation). Same input weights, same readout fitting, same data.
  Recurrent − ablation isolates what memory buys.

## Provenance (same discipline as Stage 2b)

Hyperparameters selected on `runs/stage2b/sm_train_split.json` **only**, via a
deterministic 70/30 train/val split *within* that file (`VAL_SEED=23`). The
frozen `runs/stage2b/gate_split.json` was evaluated **once**, for both models.

| file | sha256 (prefix) |
|---|---|
| `sm_train_split.json` (train) | `8df212c4b2700591` |
| `gate_split.json` (one-shot gate) | `3f7863a68261f5ee` |
| `gate_results.json` (this result) | `f24540770fa58f9c` |

Reproduce: `python evaluate_stage2c.py` (selection sweep + one-shot gate).

## Result — one-shot gate (`gate_results.json`)

| controller | overall | cross / regular / asym | gate |
|---|---:|---|---|
| waypoint witness (privileged, ref) | **1.00** | 1.00 / 1.00 / 1.00 | PASS |
| **recurrent (memory)** | **0.57** | 0.58 / 0.61 / 0.52 | **FAIL** |
| memoryless ablation | 0.56 | 0.67 / 0.59 / 0.50 | FAIL |
| Stage 2b state machine (ref) | 0.62 | 0.92 / 0.50 / 0.66 | fail |

**Negative result.** The imitation-trained recurrent baseline does not clear the
gate, and does **not** meaningfully beat its own memoryless ablation
(0.57 vs 0.56); it also does not beat the hand-tuned Stage 2b SM (0.62).

## Why (diagnosis — bounded)

This result does **not** falsify the route-memory hypothesis. It rules out one
specific setup — behavior cloning this ESN family against the privileged waypoint
teacher — and leaves several explanations open.

- **The teacher is not memoryless w.r.t. the student's observations.** The
  waypoint witness acts on privileged route, pose, segment, and carrot state
  (`stage2b.py`), none of which the student sees, so its actions are *not* a pure
  function of the student's observation. A recurrent student could in principle
  infer some of that hidden state from observation history. The labels are not
  known to be memory-free, so "there is nothing for memory to learn" is **not** a
  claim this run supports.
- **Several failure modes remain live; none is established as dominant.** The
  closed-loop collapse (both models ≈ 0.57 while the teacher reaches 1.00) is
  consistent with (a) imitation covariate shift — small action errors compound
  into states absent from the demonstrations; (b) inadequate capacity in a
  32-config fixed-reservoir family; (c) an action-loss mismatch — a squared-error
  readout on `(steering, acceleration)` is not the arrival objective; and (d)
  partially unobservable teacher state the student cannot recover from history.
  This run does not separate these.
- **No open-loop fit-quality claim.** Per-action open-loop (teacher-forced)
  diagnostics were not persisted in the gate artifact, so this write-up makes no
  "the models fit the teacher well" claim. On the gate itself, recurrent and
  ablation land within one scenario of each other (0.57 vs 0.56); this run shows
  no meaningful recurrence benefit within this imitation setup, and attributes
  nothing beyond that.

**Bounded conclusion.** Under behavior cloning of the privileged waypoint
teacher, this 32-configuration ESN family reached 57% and showed no meaningful
recurrence benefit over its matched static ablation at 56%. The result rules out
this specific imitation setup, not route memory.

## Next step (revised by this result)

Before dopamine/connectome learning, the clean test is a **memory-requiring
training signal** on the same observation interface — an end-to-end reward for
reaching the goal (optimized, not cloned) — as a separate Stage 2d. Two
constraints carry over:

- **The Stage 2c gate is spent.** It informed this design, so it cannot serve as
  the one-shot confirmation for Stage 2d. Stage 2d must freeze a **fresh,
  disjoint** gate split under the same provenance discipline.
- **Only a pass is conclusive.** A reward-trained *failure* would still not
  isolate the interface — capacity, optimization, and observation limits stay
  confounded. Only a pass shows this controller family, on this interface, is
  sufficient.

Dopamine, when it enters, is a teaching/modulatory signal — **never** goal
bearing.
