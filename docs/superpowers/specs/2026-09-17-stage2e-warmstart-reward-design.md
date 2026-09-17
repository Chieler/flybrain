# Stage 2e — warm-start reward fine-tuning (design)

Date: 2026-09-17 · Status: proposed (awaiting review)

## Why

Stage 2d changed one variable versus 2c (reward instead of imitation) and both the
recurrent net and its ablation scored **0%** on the gate. The known-policy check
settled *why*: the reward function recognizes good policies (the exact same
150-parameter policy class scores ≈1.1 reward / 22-of-40 arrivals under the
deterministic Stage 2c readouts), but **CEM-from-`μ=0` plateaued at the no-op
floor (−0.18)** — it failed to bootstrap behavior *already inside its own search
space*. That is an optimization-bootstrap failure, not an interface verdict.

Stage 2e removes exactly that failure mode: **initialize the search at the Stage
2c readout and let reward fine-tune perturbations around it.** This starts inside
the behaving regime, so the sparse arrival term has gradient signal from the first
iteration. It trades away 2d's "clean reward-only" purity (a warm start is a
deliberate, documented choice), in exchange for actually testing the question 2d
could not reach: *given a policy that already half-works, can reward improve it to
the 90% gate on a fresh, disjoint split?*

**Asymmetry of the verdict (unchanged).** A **pass** is evidence that reward
exploited the available policy state (memory and/or reactive features) to clear
the gate. A **failure stays bounded** — it does not implicate the observation
interface on its own. Only a pass is conclusive.

## What stays identical

- **Observation interface** — `stage2c.observation_features` (10 features). No map,
  no pose, no compass-difference term.
- **Model & policy class** — `stage2c.EchoStateNetwork` frozen to the 2c winner
  (`n=64, spectral_radius=0.8, leak=0.5, input_scale=1.0, seed=0`, `ridge` only
  affects the warm-start readout), `RecurrentController`, and the matched
  memoryless ablation (`recurrent=False`). 150 trainable readout params.
- **Reward** — `stage2d.episode_reward` unchanged: bounded net progress + arrival
  − collision − time, frozen coefficients `2.0 / 1.0 / 1.0 / 0.2`, evaluator-only
  privileged geometry, never a controller input.
- **Discipline** — identical budget/seeds for recurrent and ablation; one-shot
  fresh gate; hyperparameters never tuned against the gate.

## What changes

### 1. Warm start from the deterministic Stage 2c readout
For each model (recurrent, ablation) the CEM mean is initialized to the **exact
Stage 2c readout**, reconstructed deterministically (no grid re-search):

```
best_hp   = json(runs/stage2c/gate_results.json)["hyperparameters"]   # n=64, sr=0.8, leak=0.5, in=1.0, ridge=1e-4
feats,targs = evaluate_stage2c.collect_demos(load_scenarios(sm_train_split), layouts)
esn        = evaluate_stage2c._fit(recurrent, best_hp, feats, targs)   # deterministic (ESN_SEED=0)
theta0     = esn.W_out.flatten()                                       # 150-vector warm start
```

`collect_demos` rolls the deterministic waypoint teacher; `_fit` is ridge
regression with a fixed seed — so `theta0` is reproducible bit-for-bit.

### 2. Reproduction gate (hard assertion before optimizing)
Before any perturbation, evaluate `theta0` on the **40 fixed fitness scenarios**
and assert it reproduces the known-policy check within tolerance:

- recurrent: mean reward ≈ **1.106**, **22/40** arrivals;
- ablation: mean reward ≈ **1.086**, **22/40** arrivals.

Tolerance: arrivals exact (22), mean reward within ±0.02. A mismatch aborts the
run (it would mean the 2c readout was not reconstructed exactly). This both
validates reconstruction and records iteration-zero behavior in the artifact.

### 3. Optimizer: CEM around the warm start, tracking best-ever
Extend `stage2d.cem_maximize` (additively, backward-compatible) so it also records
the **best-ever candidate**, addressing the two defects the review flagged at
`stage2d.py:110`:

- **Returns / persists the best-ever candidate, not just the final elite mean.**
  `info["best"] = {"theta": <150 floats>, "fitness": float, "iter": int}` updated
  whenever a sampled candidate beats the incumbent. Stage 2e uses `info["best"]
  ["theta"]` as the trained readout (the final elite mean is only a fallback).
- **Persists `W_out`.** The best-ever `theta` (reshaped `2×75`) is written to the
  results artifact as a JSON array, so the trained policy is reproducible without
  re-running CEM.

Initialization: `μ = theta0`, per-dim `σ = INIT_STD` (a **small perturbation**,
proposed `INIT_STD = 0.1`, frozen before the run — small enough to stay near the
behaving regime, large enough to explore). Elite refit, diagonal covariance, and
`σ` floor as in 2d. Budget predeclared and **identical** for both models:
`population = 64`, `n_iter = 25`, `elite_frac = 0.20`, over the 40 fixed fitness
scenarios. Same reservoir seed + same CEM seed ⇒ deterministic. **Timing smoke
first**; reduce budget only on the timing spike, never on training results, never
by inspecting the gate.

### 4. Track and persist per-episode outcomes (fixing the 2d blind spot)
2d recorded only scalar fitness, so "arrivals during training" was unknowable.
Stage 2e persists, for the best-ever policy of each model: mean reward, and the
**arrival / collision / timeout counts** over both the 40 fitness scenarios and
the gate. The per-iteration history keeps best/mean fitness as before.

## Fresh, exact-disjoint gate

Build one fresh gate split with `evaluate_stage2b.generate_stage2b_split`
(outward-facing road-end starts excluded), excluding the **complete** prior set —
the corrected `PRIOR_SPLIT_PATHS` (now including `runs/stage2/training.json` and
`runs/stage2/heldout.json`) **plus the spent Stage 2d gate and train splits**:

```
exclude = PRIOR_SPLIT_PATHS + [runs/stage2d/gate_split.json, runs/stage2d/train_split.json]
```

Under this exclusion exactly **7 eligible `cross` scenarios remain** (verified;
regular 145, asymmetric 173), so the gate is **predeclared 7 / 44 / 44 = 95
scenarios** — it consumes all 7 cross, no buffer. **Caveat (documented):** with 7
cross, the per-layout gate (0.80) requires **6 of 7** cross arrivals; the cross
stratum is coarse. Route reuse is acceptable and documented; only **exact scenario
identity** must be disjoint. **Assert** (fail loudly) that the 2e gate shares no
`_scenario_key` with any prior split, the spent 2d splits, or the fitness set.
Freeze to `runs/stage2e/gate_split.json`; record sha256 of it and every excluded
split in `gate_results.json`.

### Fitness scenarios
Reuse the **exact 40 Stage 2d fitness scenarios**
(`evaluate_stage2d.select_fitness_scenarios(stage2d_train)` = 4 cross / 18 regular
/ 18 asymmetric), so the iteration-zero reproduction check matches the known-policy
table exactly. These live inside the spent 2d train split, which the 2e gate fully
excludes — so gate ∩ fitness = ∅. (The 2d train split's own overlap defect is
irrelevant here: it is training data, and the fresh gate is disjoint from it.)

## Selection discipline: none — fixed config, one-shot gate

No hyperparameter/config selection and no train/val split (the reservoir is frozen
and the warm start is deterministic). Train `W_out` by warm-start CEM on the 40
fitness scenarios, once for recurrent and once for ablation with identical
seeds/budget; evaluate the frozen `stage2e/gate_split` **once** for each. The only
permitted pre-run adjustment is a budget reduction driven solely by the timing
smoke.

## Interpretation (bounded, baked into the output)

- **Recurrent passes AND beats ablation meaningfully (> 0.02)** → reward,
  fine-tuning temporal memory over this interface, clears the gate; memory over
  the observation stream is what the interface needed. Motivates the biologically
  grounded memory/action-selection work (dopamine as modulation/teaching, **never**
  goal bearing).
- **Recurrent passes, ablation also passes** → reward exploited the *reactive*
  policy state to clear the gate; memory is not required for it. The interface is
  sufficient; memory is a separate, unproven question.
- **Both fail** → bounded and confounded (optimization budget/search, capacity,
  observation limits are not separated). Warm start alone did not close the gap;
  **not** an interface verdict. Diagnose (e.g. curriculum, denser shaping, stronger
  optimizer) before any interface claim or connectome/dopamine learning.

Any pass is "evidence reward exploited the available policy state," not proof of
general sufficiency; the fresh disjoint gate is what makes a pass meaningful.

## Files

- Extend `stage2d.cem_maximize` to record `info["best"]` (backward-compatible;
  does not change the returned `mu` or any 2d semantics).
- `stage2e.py` — `warm_start_readout(recurrent)` (reconstruct `theta0` via
  `evaluate_stage2c`), `train_readout_by_reward_warmstart(esn, theta0,
  fitness_scenarios, layouts, cfg)` (CEM from `theta0`, returns best-ever readout +
  outcome counts); reuses `stage2d.episode_reward` unchanged.
- `test_stage2e.py` (TDD) — warm start reproduces the 2c readout exactly (22/40,
  reward ≈1.1) for both models; CEM-with-best-ever returns a candidate ≥ the warm
  start and never worse than its own final mean; `info["best"]["theta"]` is
  persisted and reload-equivalent; the 2e gate is exact-identity disjoint from all
  priors + spent 2d splits + the fitness set (7/44/44 = 95, all 7 cross used).
- `evaluate_stage2e.py` — build/freeze the fresh gate (complete exclusion, assert
  disjoint), reproduction-gate assertion, timing smoke, warm-start CEM for both
  models (identical budget/seeds), one-shot gate, persist best-ever `W_out` +
  reward + arrival/collision/timeout counts, provenance sha256, write
  `runs/stage2e/{results,gate_results}.json`.
- `runs/stage2e/{gate_split,results,gate_results}.json` + `README.md`.

Does **not** touch the connectome or the frozen Stage 2 / 2b / 2c / 2d artifacts.
The spent Stage 2d gate is never rerun or replaced.

## Out of scope

No connectome/dopamine learning yet. No map/planner bolted onto the
observation-only controller. Stage 2e only asks whether reward, starting from a
half-working policy, can clear a fresh gate on this interface. Connectome-grounded
learning follows only if a pass settles the question.
