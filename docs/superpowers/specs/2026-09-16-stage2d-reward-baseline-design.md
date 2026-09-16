# Stage 2d — reward-trained same-observation baseline (design)

Date: 2026-09-16 · Status: proposed (awaiting review)

## Why

Stage 2c ruled out *one* setup — behavior-cloning a fixed-reservoir ESN family
against the privileged waypoint teacher reached 57%, with no meaningful
recurrence benefit over its static ablation (56%). That result does **not**
falsify the route-memory hypothesis; it says imitation of a teacher that carries
hidden state is the wrong instrument.

Stage 2d changes exactly one variable versus 2c: **the training signal**. Instead
of cloning teacher actions, we optimize the policy against an **episode reward**
for reaching the goal — a memory-requiring signal. Everything else (observation
interface, the fixed random recurrent reservoir, the recurrent-vs-memoryless
comparison) is held fixed, so a pass/fail is attributable to reward-vs-imitation.

**Asymmetry of the verdict (carried from the 2c review).** Only a **pass** is
conclusive: it shows this controller family, on this interface, is sufficient. A
**failure** stays confounded between optimization budget, capacity, and
observation limits — it does not implicate the interface on its own.

## What stays identical to Stage 2c

- **Observation interface** — `stage2c.observation_features`: faithful sin/cos of
  heading and goal_bearing, speed, 5 ranges (10 features). No map, no pose, no
  compass-difference term.
- **Model** — `stage2c.EchoStateNetwork` (fixed seeded leaky-tanh reservoir +
  linear readout) and `stage2c.RecurrentController`. Reused as-is.
- **Matched ablation** — `recurrent=False` (recurrent weights zero + unit leak):
  a static random-feature map of the current observation. Same reservoir seed,
  same optimizer, same data. Recurrent − ablation isolates what memory buys under
  reward training.

## What changes: reward instead of imitation

### Trainable parameters
Only the linear readout `W_out` (shape `2 × (1 + N_FEATURES + n)`). The reservoir
(`W_in`, `W`, leak, seed) is fixed, exactly as in 2c. Parameter count is small:
`P = 2·(1 + 10 + n)` → 86 for n=32, 130 for n=64.

Warm start is `μ = 0` (no ridge/imitation initialization), so this is a clean
reward-only test — no imitation leaks in through the starting point.

### Reward (privileged geometry, teaching signal only — never an input)
Per episode, from `StreetEpisodeResult` + the scenario's target:

```
d0       = dist(start, target)
d_min    = min over trajectory positions of dist(pos, target)   # closest approach
progress = clip((d0 - d_min) / max(d0, eps), 0, 1)              # in [0, 1]
R = W_arrive·[arrival] + W_progress·progress
      - W_collide·[collision] - W_time·(elapsed / MAX_SIM_TIME)
```

Fixed weights (design choices, documented, **not** tuned on the gate):
`W_arrive=1.0, W_progress=1.0, W_collide=0.5, W_time=0.1`. Privileged distances
enter the *reward*, never the controller's inputs — this is the legitimate
dopamine-style teaching role.

### Optimizer: Cross-Entropy Method (pure NumPy, deterministic)
Black-box optimization over `θ = flatten(W_out)` — the closed-loop rollout is not
differentiable, and CEM is ~30 lines with no framework:

1. Init `μ = 0`, per-dim `σ = init_std`. Seed the CEM rng.
2. Each iteration: sample `pop` candidates `θ_i ~ N(μ, diag(σ²))`; fitness(θ_i) =
   mean reward over the fit scenarios (deterministic — sim + controller are
   deterministic); keep the top `elite_frac`; refit `μ, σ` to the elites (with a
   small `σ` floor to avoid collapse). Repeat for `n_iter`.
3. Return `μ` as the trained readout.

Determinism: fixed reservoir seed + fixed CEM seed ⇒ reproducible.

## Fresh, disjoint gate (the 2c review requirement)

The Stage 2c gate is spent and informed this design, so it cannot confirm 2d.
Build two new splits with the existing, tested `evaluate_stage2.generate_dev_split`
(100 scenarios, 12/44/44, excludes a given set by scenario key, prefers
route-fresh candidates):

1. `stage2d/gate_split`  = `generate_dev_split(seed=D1, exclude = stage2b_train + stage2b_gate)`
2. `stage2d/train_split` = `generate_dev_split(seed=D2, exclude = stage2b_train + stage2b_gate + stage2d_gate)`

Then **assert** (fail loudly otherwise) that the Stage 2d gate shares no
`_scenario_key` with stage2b train, stage2b gate, or the Stage 2d train split.
Freeze both to `runs/stage2d/` and record sha256 of all four splits + the
`gate_results.json`, the same provenance discipline as 2b/2c.

**Risk — route-pool exhaustion (`cross`).** The single-intersection `cross`
layout has a small route pool; `generate_dev_split` raises if it can't fill
12 fresh `cross` scenarios after exclusions. Mitigation, in order: (a) rely on
its existing new-heading reuse (still disjoint by scenario key); (b) if it still
raises, reduce the Stage 2d **train** cross count (keep the gate at 12/44/44 for
comparability). Whichever path is taken is recorded in the run README.

## Selection discipline (train-only; gate untouched until the one-shot)

Identical shape to 2c:
- Split `stage2d/train_split` into fit/val within-file (`VAL_SEED`, 70/30).
- Small config grid — reservoir `{n_reservoir, spectral_radius, leak}` × a couple
  of CEM budgets. Train each with CEM on the fit sub-split; **select** by
  closed-loop arrival on the val sub-split.
- Refit the selected config with CEM on the **full** `stage2d/train_split`.
- Evaluate the frozen `stage2d/gate_split` **once**, for recurrent **and**
  ablation. Never re-tune against it.

**Runtime is the main cost.** CEM fitness = `pop × n_iter × |fit scenarios|`
deterministic rollouts, each up to `MAX_SIM_TIME/PHYSICS_DT` steps. Keep the
budget modest and documented (starting point: `pop≈32`, `n_iter≈20`, fit subset
≈40 scenarios, small config grid), with CEM early-stop on a val plateau. If a
full run is too slow, shrink the grid/budget — recorded in the README — never by
peeking at the gate.

## Interpretation (bounded, baked into the output)

- **Recurrent passes AND beats ablation meaningfully** → temporal memory over the
  observation interface, trained by reward, closes the gap. The interface is not
  the bottleneck; motivates the biologically grounded memory/action-selection
  work (dopamine as modulation/teaching, **never** goal bearing).
- **Recurrent passes but ablation also passes** → reward training (not memory
  specifically) closes the gap; the interface is sufficient and memory is not
  required for it.
- **Both fail** → bounded and confounded: optimization budget, capacity, and
  observation limits are not separated; **not** a clean interface verdict (per
  the pass-only-is-conclusive asymmetry). Diagnose before any interface claim.

## Files

- `stage2d.py` — `episode_reward(...)`, `CEM` optimizer, `train_readout_by_reward(
  esn, scenarios, layouts, cem_cfg)`; reuses `stage2c.{observation_features,
  EchoStateNetwork, RecurrentController}`.
- `test_stage2d.py` (TDD) — reward monotonic in progress/arrival & penalized for
  collision; CEM improves a toy quadratic; controller is observation-only
  (`(self, obs)`); determinism (fixed seeds reproduce); split disjointness assert.
- `evaluate_stage2d.py` — build/freeze disjoint splits, select on train/val,
  refit, one-shot gate (recurrent + ablation), provenance sha256, write
  `runs/stage2d/{results,gate_results}.json`.
- `runs/stage2d/{train_split,gate_split,results,gate_results}.json` + `README.md`.

Does **not** touch the connectome or the frozen Stage 2 / 2b / 2c artifacts.

## Out of scope

No connectome/dopamine learning yet — Stage 2d only asks whether a reward signal
on this interface can close the gap. Connectome-grounded learning follows only if
2d passes and the memory question is settled.
