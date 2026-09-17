# Stage 2f — aligned-reward warm-start fine-tuning (design)

Date: 2026-09-17 · Status: proposed (awaiting review)

## Why

Stage 2e removed 2d's optimization-bootstrap failure: warm-starting CEM at the
deterministic Stage 2c readout drove the recurrent gate from 0.00 to **0.516**
(zero timeouts — the policy drives purposefully), and the recurrent net beat its
memoryless ablation (0.516 vs 0.400) — the first memory signal in the Stage 2
line. But neither cleared the 0.90/0.80 gate, and the shortfall was diagnosable:

1. **Reward ≠ arrival-rate.** The ablation's CEM *raised reward while arrivals
   fell* (22→18, timeouts 1→16): the net-progress/time terms rewarded a cautious,
   lower-arrival optimum. Fitness and the gate metric were misaligned.
2. **Collisions dominated the misses** (recurrent 46 of 95); the collision penalty
   was too weak to suppress crashes.
3. **Train→gate generalization gap** (65% on the 40 fitness scenarios → 51.6% on
   the fresh gate): optimizing against only 40 fixed scenarios overfit them.

Stage 2f addresses (1)–(3) directly while keeping everything else 2e established
(warm start, frozen 2c-winner reservoir, matched recurrent-vs-ablation, CEM with
best-ever tracking, identical budget/seeds). **This is a bundled push toward a
pass, not a clean one-variable ablation** — justified because a *pass* is
conclusive under the asymmetry regardless of which ingredient carried it, and the
two experimental changes (reward alignment + more fitness scenarios) are both
"make reward learnable and aligned," diagnosed from 2e. A **failure** stays bounded
and is confounded between those two.

**Asymmetry of the verdict (unchanged).** Only a **pass** is conclusive — evidence
that reward, aligned to the gate metric, exploited the available policy state on a
fresh disjoint gate. Recurrent beating its ablation further implicates temporal
memory. A failure does not implicate the observation interface on its own.

## What changes

### 1. Reward aligned to the gate metric (frozen coefficients, by design)
Same bounded-net-progress structure and evaluator-only privileged geometry as
2d/2e (final position, never a controller input), with coefficients re-frozen so
**fitness tracks arrival rate** and **collisions are strictly discouraged**:

```
progress = clip((d0 - d_final)/d0, -1, 1)
R = 4.0·[arrival] + 0.5·progress - 2.0·[collision] - 0.1·(elapsed/MAX_SIM_TIME)
```

Coefficients `W_arrive=4.0, W_progress=0.5, W_collide=2.0, W_time=0.1`, frozen
before training (documented, **not** tuned on the gate). The design intent, all
asserted as tests:

- **Arrival dominates 8:1.** Max non-arrival shaping is `W_progress·1 = 0.5`;
  arrival adds `4.0`. The optimizer cannot trade an arrival for progress — closing
  the 2e reward/metric gap that let the ablation raise reward while arrivals fell.
- **Strict outcome ordering:** collision (≤ `0.5 − 2.0 − 0.1 = −1.6`) < standing
  still (progress 0, timeout ⇒ `−0.1`) < meaningful forward progress without
  arrival (progress 0.5, timeout ⇒ `+0.15`) < arrival (≥ `4.0 − 0.1`). Crashing is
  now strictly worse than making progress and stopping — targeting the 46/95
  collisions.
- **Bounded and unfarmable** (final position, clipped net progress); **standing
  still dominated** by forward progress.

### 2. More fitness scenarios (close the generalization gap)
Optimize against a larger stratified fitness set — **86 scenarios (6 cross / 40
regular / 40 asymmetric)** versus 2e's 40 — drawn from a fresh Stage 2f fitness
split, so the policy is fit to a broader sample and overfits the fixed set less.
(Plain "more," not a moving curriculum — a larger fixed set keeps CEM fitness a
stationary objective and the run deterministic.)

### 3. Everything else held from 2e
- **Warm start** at the deterministic 2c readout (2e proved it essential);
  reproduction asserted before optimizing — **arrivals == 22** for both models on
  the fixed 2d-40 reference set (a coefficient-independent reconstruction
  fingerprint; the reward *value* changes under the new coefficients and is
  informational only).
- **Frozen 2c-winner reservoir** (n=64, sr=0.8, leak=0.5, in=1.0, seed=0), 150
  readout params; matched memoryless ablation; **CEM with best-ever tracking + the
  warm-start guard** (never ships worse than θ₀), `init_std=0.1`, `population=64`,
  `n_iter=25`, identical seeds for both models. Timing smoke first; the 86-scenario
  fitness set roughly doubles per-eval cost — reduce `n_iter` **only** on a timing
  spike, never on training results, never by inspecting the gate.

## Fresh gate — with an expanded cross stratum

**The cross layout is exhausted.** Its total eligible pool under the prior
generation rule (waypoint pairs ≥12 apart × 4 cardinal headings, minus
outward-facing road ends) is only 64, and Stages 2/2b/2d/2e have consumed all 64 —
0 fresh cross remain. Lowering the min-distance yields 0 additional cross (all
cardinal-heading pairs are used at every distance). **The only source of fresh
cross is adding non-cardinal start headings.**

**Expansion (cross only):** add the four diagonal headings (`±π/4, ±3π/4`) to cross
generation, keeping `dist ≥ 12` (so route *lengths* stay in the same regime — only
initial orientation changes) and the outward-road-end filter. To keep the gate
**fair**, filter the expanded cross to **waypoint-solvable** scenarios (the
privileged witness arrives) — necessary because diagonal starts include
witness-unsolvable configurations. This yields **48 fresh, solvable** diagonal-cross
scenarios (verified). Regular and asymmetric generation is **unchanged**
(`generate_stage2b_split`, cardinal headings, outward filter) — they remain
directly comparable to prior stages; only cross changes.

**Documented consequence:** the 2f cross stratum uses diagonal start orientations
and is solvability-filtered, so **cross is not orientation-comparable to prior
stages** and cross trend lines across stages are invalid. It is still a real
one-shot test of solvable cross scenarios. Regular and asymmetric remain
trend-comparable.

**Predeclared counts** (fresh, exact-identity disjoint from the complete corrected
prior set + spent 2d gate/train + spent 2e gate + the fitness split + the 2d-40
reproduction reference):

- `stage2f/gate_split`    = 12 cross / 44 regular / 44 asymmetric = **100**
- `stage2f/fitness_split` = 6 cross / 40 regular / 40 asymmetric = **86**

Supply check (fresh remaining): cross 48 (use 18, buffer 30), regular 101 (use 84,
buffer 17), asymmetric 129 (use 84, buffer 45) — all feasible. With 12 cross the
cross gate (0.80) needs **10/12** (far less coarse than 2e's 6/7). **Assert** (fail
loudly) that the gate shares no `_scenario_key` with any excluded split or the
fitness split. Freeze both to `runs/stage2f/`; record sha256 of them and every
excluded split.

## Selection discipline: none — fixed config, frozen coefficients, one-shot gate

No hyperparameter/config selection, no train/val split, no coefficient tuning on
data. Coefficients are frozen by design; the reservoir is frozen; the warm start is
deterministic. Train `W_out` by warm-start CEM on the 86 fitness scenarios, once
for recurrent and once for ablation (identical seeds/budget), and evaluate the
frozen gate **once** for each. The only permitted pre-run adjustment is a budget
reduction driven solely by the timing smoke.

## Interpretation (bounded, baked into the output)

- **Recurrent passes AND beats ablation (> 0.02)** → aligned reward, fine-tuning
  temporal memory over the observation interface, clears a fresh gate and memory
  carries the advantage. The interface is not the bottleneck; motivates the
  biologically grounded memory/action-selection work (dopamine as
  modulation/teaching, **never** goal bearing).
- **Recurrent passes, ablation also passes** → aligned reward exploited the
  reactive policy state; the interface is sufficient and memory is not required.
- **Both fail** → bounded and confounded between reward alignment and scenario
  coverage (and residually capacity/optimization). Report the recurrent−ablation
  gap and the collision/arrival breakdown; diagnose before any interface claim.
  **Not** an interface verdict.

## Files

- `stage2f.py` — aligned reward coefficients + `episode_reward_aligned(scenario,
  result)` (same bounded structure as `stage2d.episode_reward`, new coefficients).
- Extend `stage2e.{evaluate_policy, train_readout_by_reward_warmstart}` to accept a
  `reward_fn` (default `stage2d.episode_reward`, so 2e semantics are unchanged); 2f
  passes `episode_reward_aligned`.
- `evaluate_stage2f.py` — `generate_expanded_cross(seed, exclude, count)` (diagonal
  headings, dist≥12, outward-filtered, witness-solvable, exclusion-disjoint);
  build/freeze the fresh gate (expanded cross + `generate_stage2b_split` regular/
  asym) and the 86-scenario fitness split, assert disjoint; reproduction assertion
  (arrivals==22); timing smoke; warm-start CEM with the aligned reward for both
  models; one-shot gate; persist best-ever `W_out` + arrival/collision/timeout
  counts; provenance sha256; write `runs/stage2f/{results,gate_results}.json`.
- `test_stage2f.py` (TDD) — aligned reward ordering (collision < standing-still <
  forward-progress < arrival; arrival dominates 8:1; bounded; unfarmable via final
  position); expanded cross are all cross / diagonal-heading / witness-solvable /
  fresh; gate + fitness exact-identity disjoint from all priors + spent 2d/2e + each
  other (100 / 86, all cross diagonal); warm-start reproduces 22/40; the trainer
  uses the aligned reward.
- `runs/stage2f/{gate_split,fitness_split,results,gate_results}.json` + `README.md`.

Does **not** touch the connectome or the frozen Stage 2/2b/2c/2d/2e artifacts; no
spent gate is rerun or replaced.

## Out of scope

No connectome/dopamine learning yet. No map/planner on the observation-only
controller. No stronger optimizer this stage (CEM held from 2e); a CMA-ES / larger
budget probe is a separate later stage if 2f still falls short. Connectome-grounded
learning follows only if a pass settles the memory question.
