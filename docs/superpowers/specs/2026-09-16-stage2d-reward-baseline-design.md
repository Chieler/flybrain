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
(`W_in`, `W`, leak, seed) is **fixed to the Stage 2c winner**: `n_reservoir=64`,
`spectral_radius=0.8`, `leak=0.5`, `input_scale=1.0`, `esn_seed=0`. So
`P = 2·(1 + 10 + 64) = 150` trainable parameters.

**No reservoir/config grid.** Running a config sweep would change more than the
training signal and break the one-variable design; the reservoir config is frozen
to 2c's winner and only `W_out` is optimized. There is therefore **no** train/val
selection step in Stage 2d.

Warm start is `μ = 0` (no ridge/imitation initialization), so this is a clean
reward-only test — no imitation leaks in through the starting point.

### Reward — bounded net progress (privileged geometry, evaluator-only)
Per episode, from `StreetEpisodeResult` + the scenario's target. Uses **net**
progress from start to *final* position (not accumulated positive progress, and
not closest approach), so wandering near the goal and leaving cannot be farmed:

```
d0        = dist(start, target)
d_final   = dist(final position, target)
progress  = clip((d0 - d_final) / d0, -1.0, 1.0)     # bounded net progress
R = W_arrive·[arrival] + W_progress·progress
      - W_collide·[collision] - W_time·(elapsed / MAX_SIM_TIME)
```

Coefficients frozen before any training (documented, **not** tuned on the gate):
`W_arrive=2.0, W_progress=1.0, W_collide=1.0, W_time=0.2`. This makes the arrival
bonus (2.0) **dominate the maximum possible shaping contribution** (progress
maxes at 1.0). Sanity property, asserted in a test: standing still (progress 0,
timeout ⇒ R = −0.2) must score **below** meaningful forward progress without a
collision (e.g. progress 0.5, timeout ⇒ R = 0.3). Privileged distances enter the
*reward only*, never the controller's inputs — the legitimate teaching role.

### Optimizer: Cross-Entropy Method (pure NumPy, deterministic)
Black-box optimization over `θ = flatten(W_out)` (150 dims) — the closed-loop
rollout is not differentiable, and CEM is ~30 lines with no framework:

1. Init `μ = 0`, per-dim `σ = init_std`. Seed the CEM rng.
2. Each iteration: sample `pop` candidates `θ_i ~ N(μ, diag(σ²))`; fitness(θ_i) =
   mean reward over the fixed fitness scenarios (deterministic — sim + controller
   are deterministic); keep the top `elite_frac`; refit `μ, σ` (diagonal
   covariance) to the elites (with a small `σ` floor to avoid collapse). Repeat
   for `n_iter`. Return `μ` as the trained readout.

**Frozen budget** (predeclared; identical for recurrent and ablation):
`population=64`, `n_iter=25`, `elite_frac=0.20`, diagonal covariance, over
**40 fixed, stratified fitness scenarios** drawn from the Stage 2d train split
(stratified across layouts, e.g. `cross=4, regular=18, asymmetric=18`). That is
`64 × 25 × 40 = 64,000` rollouts per model, **128,000 total** across recurrent +
ablation. Determinism: fixed reservoir seed + fixed CEM seed ⇒ reproducible.

**Timing-only smoke test first.** Before the real run, time a small slice
(e.g. one CEM iteration, or N rollouts) to estimate wall-clock. The budget is
frozen *before* optimization and reduced **only** on the basis of that timing —
never on training results, and never by looking at the gate.

## Fresh, disjoint gate (the 2c review requirement)

The Stage 2c gate is spent and informed this design, so it cannot confirm 2d.
Build two new splits with `evaluate_stage2b.generate_stage2b_split` (not
`generate_dev_split`) so **outward-facing road-end starts stay excluded**, the
same eligibility rule Stage 2b used. Exclude **every prior Stage 2 / 2b split**:
`runs/stage2/dev_split.json` + `runs/stage2b/gate_split.json` +
`runs/stage2b/sm_train_split.json`.

Only **23** unused eligible `cross` scenarios remain after those exclusions, so
the counts are **predeclared** (gate first, then train excludes the gate):

1. `stage2d/gate_split`  = `generate_stage2b_split(seed=D1, exclude=prior, counts={cross:12, regular:44, asymmetric:44})`
2. `stage2d/train_split` = `generate_stage2b_split(seed=D2, exclude=prior + gate, counts={cross:8, regular:44, asymmetric:44})`

`cross` usage is `12 + 8 = 20`, leaving **3 as buffer** (20 ≤ 23), so generation
cannot exhaust. **Route reuse is acceptable and documented**; only **exact
scenario identity** must be disjoint. There is **no adaptive "new-heading reuse"
fallback** — the predeclared counts guarantee feasibility outright.

Then **assert** (fail loudly otherwise) that the Stage 2d gate shares no
`_scenario_key` with any prior split or with the Stage 2d train split. Freeze
both to `runs/stage2d/` and record sha256 of the two new splits, every excluded
prior split, and `gate_results.json` — the same provenance discipline as 2b/2c.

## Selection discipline: none — fixed config, one-shot gate

Because the reservoir config is frozen to the 2c winner (one-variable design),
there is **no** hyperparameter/config selection and **no** train/val split. The
procedure is:

- Train `W_out` by CEM on the **40 fixed stratified fitness scenarios** (frozen
  budget above), once for recurrent and once for the ablation, identical seeds.
- Evaluate the frozen `stage2d/gate_split` **once** for each. Never re-tune
  against it; the only permitted pre-run adjustment is a budget reduction driven
  solely by the timing smoke test.

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
  esn, fitness_scenarios, layouts, cem_cfg)`; reuses `stage2c.{observation_features,
  EchoStateNetwork, RecurrentController}` unchanged.
- `test_stage2d.py` (TDD) — reward monotonic in net progress & arrival and
  penalized for collision; **standing-still < meaningful forward progress**;
  arrival bonus > max shaping; CEM improves a toy quadratic; controller is
  observation-only (`(self, obs)`); determinism (fixed seeds reproduce); split
  exact-identity disjointness assert.
- `evaluate_stage2d.py` — build/freeze the two disjoint splits (fixed reservoir
  config, no selection), timing smoke test, train `W_out` by CEM on the 40 fixed
  fitness scenarios, one-shot gate (recurrent + ablation, identical budget/seeds),
  provenance sha256, write `runs/stage2d/{results,gate_results}.json`.
- `runs/stage2d/{train_split,gate_split,results,gate_results}.json` + `README.md`.

Does **not** touch the connectome or the frozen Stage 2 / 2b / 2c artifacts.

## Out of scope

No connectome/dopamine learning yet — Stage 2d only asks whether a reward signal
on this interface can close the gap. Connectome-grounded learning follows only if
2d passes and the memory question is settled.
