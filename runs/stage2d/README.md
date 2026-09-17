# Stage 2d — reward-trained same-observation baseline (route-memory gate, take 2)

**Verdict: bounded, confounded FAILURE. Not a clean interface verdict.**
Both the recurrent net and its matched memoryless ablation reached **0% arrival**
on the one-shot gate. Per the pass-only-is-conclusive asymmetry declared in the
design, this does **not** implicate the observation interface — it must be
diagnosed before any interface claim. (The gate has a documented disjointness
defect — see *Provenance defect* below — that cannot fake a pass and so does not
affect this 0% negative.)

## What this stage asked

Stage 2c ruled out *one* setup — behavior-cloning a fixed-reservoir ESN family
against the privileged waypoint teacher reached 57%, no meaningful recurrence
benefit. That is not a falsification of route memory; it says imitating a teacher
that carries hidden state is the wrong instrument.

Stage 2d changed **exactly one variable** versus 2c: the training signal. Instead
of cloning teacher actions, it optimized the linear readout against an **episode
reward** for reaching the goal — a memory-requiring signal — by the Cross-Entropy
Method (pure NumPy, no BPTT). Everything else was held fixed: the observation
interface (10 features: sin/cos heading, sin/cos goal_bearing, speed, 5 ranges),
the seeded recurrent reservoir frozen to the 2c winner (n=64, spectral_radius=0.8,
leak=0.5, input_scale=1.0, seed=0 → 150 readout parameters), and the
recurrent-vs-memoryless-ablation comparison.

Reward was **evaluator-only privileged geometry**, never a controller input:
bounded net progress `clip((d0−d_final)/d0, −1, 1)` + arrival bonus − collision
penalty − small time cost, with frozen coefficients `W_arrive=2.0, W_progress=1.0,
W_collide=1.0, W_time=0.2` (arrival dominates the maximum shaping of 1.0).

## Results (one-shot gate: 12 cross / 44 regular / 44 asymmetric = 100 scenarios)

| Model | Overall arrival | cross | regular | asymmetric | Passes gate (0.90/0.80) |
|---|---|---|---|---|---|
| Recurrent | **0.00** | 0.00 | 0.00 | 0.00 | ✗ |
| Memoryless ablation | **0.00** | 0.00 | 0.00 | 0.00 | ✗ |

References (context, not upper bounds): waypoint witness 1.00, Stage 2b state
machine 0.62, Stage 2c recurrent imitation 0.57.

Recurrent: 0 arrivals, 2 collisions, 98 timeouts. Ablation: 0 arrivals, 1
collision, 99 timeouts.

## Diagnosis — CEM-from-zero failed to bootstrap

Two facts fix the failure on the *optimizer*, not the interface.

**1. The reward function recognizes good policies.** A known-policy check scoring
the exact same reward on the same 40 fitness scenarios:

| Policy | Mean reward | Arrivals |
|---|---|---|
| Zero readout | −0.200 | 0/40 |
| Stage 2d CEM recurrent (trained) | −0.179 | (0 on gate) |
| Stage 2c recurrent readout | **1.106** | 22/40 |
| Stage 2c ablation readout | 1.086 | 22/40 |
| Stage 2b state machine | 1.169 | 22/40 |
| Waypoint witness | 2.837 | 40/40 |

The reward cleanly separates behaving policies (≈1.1, 22/40) from the no-op floor
(−0.20). A policy achieving ≈1.1 reward is reachable *by this exact policy class*
(the 2c readouts are the same 150-parameter linear readout on the same reservoir).

**2. CEM-from-zero landed at −0.179 — essentially the no-op floor.** The CEM best
fitness (`results.json`) plateaus at ≈ **−0.18** for both models, from a `μ=0`
start, versus the ≈1.1 the exact policy class demonstrably supports. So CEM did
not recover behavior that was *within reach of its own search space*.

Why from-scratch reward search stalls here, when 2c's imitation reached 57%:
imitation gave dense per-step teacher targets, whereas the arrival bonus is a
sparse event a `μ=0` population almost never triggers — so the arrival term gets
no gradient and only the weak dense net-progress term drives learning, toward
cautious near-stillness. Black-box CEM over a 150-dim closed-loop policy with this
budget (pop 64 × 25 × 40) is consistent with premature convergence, not a
capacity ceiling — the capacity is the same one that reaches 22/40 under 2c's fit.

**What this does NOT establish.** The recurrent (−0.179) and ablation (−0.175)
final fitnesses are two deterministic point estimates, both stuck at the no-op
floor; their near-equality is uninformative, not evidence about the value of
memory. And the scalar training history records only best/mean fitness per
iteration, not per-episode outcomes — it does **not** support any claim about how
many arrivals (if any) occurred *during* training. The memory question is
**untested**, not answered.

## Bounded conclusion

CEM-from-zero failed to bootstrap goal-reaching behavior that is demonstrably
reachable by this exact policy class, and scored 0% on the fresh gate. **This says
nothing about whether the observation interface can support the 90% gate.** It
rules out *this from-scratch reward-training setup* only. Only a pass would have
been conclusive; the failure remains bounded and confounded (optimization
budget/search, not the interface).

## Provenance defect (documented, not repaired)

The frozen gate is **not fully disjoint** from prior data. The as-run exclusion
list (`_PRIOR_SPLIT_PATHS_AS_RUN` in `evaluate_stage2d.py`) omitted two original
Stage 2 pools, so scenarios leaked in by exact identity:

| Frozen split | ∩ `training.json` (sha `adf27030…`) | ∩ `heldout.json` (sha `1329165a…`) | ∩ prior total |
|---|---|---|---|
| `gate_split.json` (100) | 2 | 6 | 8 |
| `train_split.json` (96) | 1 | 5 | 6 |

Gate and train remain mutually disjoint, and both are disjoint from
`dev_split.json` and the Stage 2b splits. **This does not undermine the negative:**
overlap with prior pools can only make arriving *easier*, and the run still
arrived on nothing (0/100). The spent, one-shot gate is retained exactly as run
(never regenerated); `PRIOR_SPLIT_PATHS` is corrected to the complete list so
Stage 2e's fresh gate excludes these pools **and** the spent 2d splits. The
overlap is codified in `test_stage2d.py` so it cannot be silently erased.

## What would move the needle next (not yet run, not yet approved)

The failure is confounded on **optimization**, so the productive next probes are
about making reward learnable, not about changing the interface or adding a map:

- **Warm-start from the 2c imitation readout** so the search begins inside the
  behaving regime, then let reward fine-tune. (Trades away the "clean reward-only"
  purity — a deliberate, documented choice, not the same one-variable test.)
- **Denser / curriculum shaping or easier scenario curriculum** so arrivals occur
  early enough to get a gradient; re-check that shaping stays unfarmable.
- **A stronger optimizer** (e.g. ES/CMA-ES with more samples, or a differentiable
  surrogate) before concluding anything about capacity.

Each of these is its own predeclared, fresh-gate experiment. **Do not** bolt a
map/planner onto the observation-only controller, and **do not** advance to
connectome/dopamine learning on the strength of this negative — the memory
question is still open.

## Provenance

- Gate split `runs/stage2d/gate_split.json` sha256 `4d52f440892e84f9…` (frozen
  before the run), train split `runs/stage2d/train_split.json` sha256
  `377629e07f01a92f…`. Mutually disjoint and disjoint from `dev_split.json` and
  the Stage 2b splits, but **not** from `training.json`/`heldout.json` — see
  *Provenance defect* above (`test_stage2d.py` codifies the exact overlap).
- As-run excluded priors (sha256 in `gate_results.json`):
  `runs/stage2/dev_split.json`, `runs/stage2b/gate_split.json`,
  `runs/stage2b/sm_train_split.json`. Omitted (the defect):
  `runs/stage2/training.json`, `runs/stage2/heldout.json`.
- Frozen CEM budget: population 64, n_iter 25, elite_frac 0.20, diagonal
  covariance, 40 stratified fitness scenarios, identical seeds for recurrent and
  ablation. 128,000 total rollouts. Wall clock 154.2 min.
- Design: `docs/superpowers/specs/2026-09-16-stage2d-reward-baseline-design.md`.
- Full per-iteration CEM history and breakdowns: `results.json`, `gate_results.json`.
