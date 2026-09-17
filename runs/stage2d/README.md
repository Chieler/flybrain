# Stage 2d — reward-trained same-observation baseline (route-memory gate, take 2)

**Verdict: bounded, confounded FAILURE. Not a clean interface verdict.**
Both the recurrent net and its matched memoryless ablation reached **0% arrival**
on the fresh one-shot gate. Per the pass-only-is-conclusive asymmetry declared in
the design, this does **not** implicate the observation interface — it must be
diagnosed before any interface claim.

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

## Diagnosis — why the optimizer, not the interface

The CEM training histories (`results.json`) are the tell. Both models converge to
a best fitness of ≈ **−0.18**, which is essentially the *no-net-progress floor*:
a car that times out having made ≈0 net progress scores `−W_time·1 ≈ −0.20`. Mean
population fitness falls from ≈ −0.41 at iteration 0 (random readouts drive off /
crash / make negative net progress) to ≈ −0.19 by iteration 24. **Neither run
ever entered the arrival regime.** A policy arriving on even a handful of the 40
fitness scenarios would have shown fitness well above 0 (each arrival contributes
+2.0); nothing close to that appears in 25 iterations of either run.

So the run did **not** discover goal-reaching behavior at all — it found "don't
crash, make near-zero net progress." That is a signature of the *optimization
setup* failing to bootstrap, not of the interface being blind:

- **Reward-from-scratch is far sparser supervision than imitation.** Stage 2c's
  57% came from dense per-step teacher targets. Here, arrival is a rare, sparse
  event; from `μ=0` the population almost never stumbles into an arrival, so the
  arrival term never gets a gradient and only the weak dense net-progress term
  drives learning — toward cautious near-stillness.
- **Black-box CEM over a 150-dim closed-loop policy is a hard search** with this
  budget (pop 64 × 25 iters × 40 scenarios). The plateau at the no-progress floor
  is consistent with premature convergence, not with a capacity ceiling.
- **The interface is not implicated.** The same interface reached 57% under
  imitation (2c) and supports the 1.00 waypoint witness with privileged state.
  0% here is about how the readout was trained, not what the controller can see.

Recurrent (−0.1789) and ablation (−0.1752) final fitnesses are statistically
indistinguishable — but with both stuck at the no-progress floor, that says
nothing about the value of memory. The memory question remains **untested**, not
answered.

## Bounded conclusion

Under from-scratch reward optimization by CEM on this interface and budget, the
fixed-reservoir ESN family (recurrent and ablation alike) failed to bootstrap any
goal-reaching behavior and scored 0% on the fresh gate. This rules out *this
specific reward-training setup*; it does **not** rule out route memory, and it
does **not** implicate the observation interface. Only a pass would have been
conclusive.

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
  `377629e07f01a92f…`. Both verified exact-identity disjoint from every prior
  Stage 2/2b split and from each other (`test_stage2d.py`).
- Excluded priors (sha256 in `gate_results.json`): `runs/stage2/dev_split.json`,
  `runs/stage2b/gate_split.json`, `runs/stage2b/sm_train_split.json`.
- Frozen CEM budget: population 64, n_iter 25, elite_frac 0.20, diagonal
  covariance, 40 stratified fitness scenarios, identical seeds for recurrent and
  ablation. 128,000 total rollouts. Wall clock 154.2 min.
- Design: `docs/superpowers/specs/2026-09-16-stage2d-reward-baseline-design.md`.
- Full per-iteration CEM history and breakdowns: `results.json`, `gate_results.json`.
