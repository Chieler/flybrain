# Stage 2g — fully-trainable recurrent baseline (design)

Date: 2026-09-21 · Status: proposed (awaiting review)

## Why

Stage 2f cleared 2d's optimization floor and produced the first memory signal in
the Stage 2 line, but **failed the 0.90/0.80 gate**: recurrent **0.660**, memoryless
ablation **0.540** (reference points: privileged waypoint witness 1.00, Stage 2b SM
0.62, Stage 2c BC 0.57, Stage 2d 0.00, Stage 2e 0.516). The recurrent−ablation gap
(+0.12) says temporal state helps; the absolute shortfall says something else caps
the policy. Across 2d–2f that cap was structurally identical: the reservoir (`W_in`,
`W`) was **frozen** to the 2c winner and only the 150-parameter linear readout
`W_out` was optimized. The recurrent dynamics themselves were never trainable.

Stage 2g changes exactly that lever: **make the recurrent weights trainable** and
optimize the whole network end-to-end against the same non-differentiable aligned
reward. The question: does a trainable recurrent policy over the *same observation
interface* clear a fresh 0.90/0.80 gate that readout-only optimization could not?

**Asymmetry of the verdict (unchanged).** Only a **pass** is conclusive. A
recurrent pass with a feed-forward-ablation fail is strong evidence that learned
temporal state supplies the missing action-selection. A failure stays bounded and
confounded between optimization budget, capacity, and coverage; it never implicates
the observation interface on its own.

**Honest attribution (baked in).** This is **not** literally one variable versus
2f. Making `W` trainable is bundled with dropping reservoir width from n=64 to
**n=8** (190 trainable params, close to 2f's demonstrated 150) — itself an
architectural change. Exact attribution of a pass to "trainable recurrence" alone
would require a matched **n=8 readout-only** control, which this stage does not run.
A pass remains meaningful under the pass-only asymmetry regardless; the language in
every artifact reflects this and claims no more.

## Model

Both arms use the Stage 2c `EchoStateNetwork` at **n=8**, seeded identically (2c
recipe: uniform `W_in` with `input_scale=1.0`, `W` scaled to spectral radius 0.8,
leak 0.5, seed 0). The seeded reservoir sets only the **initialization**; which
blocks CEM may then move defines the arm.

- **Recurrent (fully trainable).** All three blocks are optimized end-to-end:
  `W_in` (8×11) + `W` (8×8) + `W_out` (2×19) = **190 params**. Leak stays 0.5;
  spectral radius 0.8 and `input_scale` seed the initial `W`/`W_in` only — CEM is
  free to move them.
- **Ablation (recurrence-off, hidden-width-matched).** `W ≡ 0` held fixed with leak
  forced to 1 (the 2c–2f `recurrent=False` semantics: `x_t = tanh(W_in·[1;u_t])`, a
  static feature map of the current observation). Trainable set = `W_in` (8×11) +
  `W_out` (2×19) = **126 params**.

The two arms differ by the 64-parameter `W` block (190 vs 126). This gap is
**documented, not equalized** — the ablation is defined by *what recurrence removes*
(the recurrent weights and the temporal state they carry), and matching parameter
count would mean adding non-recurrent capacity that changes nothing about the
memory question. Both arms share hidden width n=8, the same seeded `W_in`
initialization, the same BC initialization procedure, the same block-scaled CEM,
and the same fitness.

## Training method

Black-box evolutionary search (CEM) over the full trainable weight vector against
the **frozen Stage 2f aligned fitness**, no BPTT (the stack is pure NumPy/SciPy with
no autograd, and the reward is non-differentiable through the simulator). Fitness is
`stage2f.aligned_fitness` unchanged:

```
progress   = clip((d0 - d_final)/d0, -1, 1)
secondary  = 0.5·progress - 2.0·[collision] - 0.1·(elapsed/MAX_SIM_TIME)   # per episode
fitness(θ) = arrivals(θ) + mean_over_episodes(secondary) / 4.0
```

Arrival-count-primary; the shaping swing is `< 1`, so one more arrival always wins.
The privileged geometry enters the reward only, never the controller's inputs.

Per arm the training pipeline is:

### 1. BC-initialized θ₀

Build the fixed n=8 reservoir; ridge-fit `W_out` against the already-spent
teacher-demo split `runs/stage2b/sm_train_split.json` (the deterministic 2c
behavior-cloning procedure, `s2c.collect_demos` + `s2c._fit`). Then
`θ₀ = flatten([W_in, W, W_out])` — the 190-vector (recurrent) or, for the ablation,
`flatten([W_in, W_out])` with `W` held at zero (126-vector). BC is **initialization
only, not an objective**: it produces a behaving warm start so the sparse arrival
term has signal from iteration zero (the fix for 2d's stillness), and sidesteps
2c's "cloning a hidden-state teacher is the wrong instrument" critique because the
clone is never scored.

### 2. Block-scaled perturbations

`W_in`, `W`, and `W_out` occupy different numeric scales; one absolute σ over all
190 values would over-perturb the small block and under-perturb the large. CEM
therefore optimizes **normalized deltas** `z` with `init_std = 1`, mapped back
through fixed, seed-derived per-block scales:

```
θ = θ₀ + block_scale ⊙ z
scale_b = 0.1 × max(RMS(θ₀_b), 1e-3)     for each trainable block b ∈ {W_in, W, W_out}
```

`RMS(θ₀_b)` is the root-mean-square of that block's entries in the warm start;
the `1e-3` floor keeps a near-zero block from collapsing its scale. The resulting
per-block scalar scales are **computed once from θ₀ and recorded** in the run
artifact (they are outputs of the frozen warm start, not tuned). The mapping is a
reparameterization **wrapper** around `cem_maximize`; `stage2d.cem_maximize` itself
is untouched — it searches `z`-space with unit `init_std`, and the wrapper's
fitness closure applies `θ = θ₀ + block_scale ⊙ z` before evaluating.

### 3. Go/no-go probe (precondition)

Before committing the full run, a **one-iteration `cem_maximize` probe** runs at the
run seed (population 64, `n_iter=1`, `init_std=1`, same `z`-space wrapper). The
precondition: **θ₀ or the first population must produce ≥1 arrival on the train
split AND beat no-op.** If it does not, **abort this arm and escalate** (see
Escalation) — dense progress alone did not rescue Stage 2d, so a non-behaving start
is not run to 30 iterations. If it clears, the full **30-iteration** run restarts at
the *same seed*, so iteration 0 reproduces the probe bit-for-bit (the probe is a
prefix of the full run, not extra search). This keeps `cem_maximize` unchanged.

### 4. Warm-start guard

The best-ever candidate is shipped only if it beats θ₀'s fitness on the train split;
otherwise θ₀ is kept (the 2e guard — never ship a policy worse than the warm start).

### 5. Frozen budget

`population=64, n_iter=30, elite_frac=0.20, init_std=1.0, std_floor=0.001, seed=0`,
identical for both arms. Two arms cost `2 × 64 × 30 × 66 = 253,440` episode
evaluations — fewer than 2f's `2 × 64 × 25 × 86 = 275,200`. `n_iter=30` is fixed
unconditionally; any timing smoke is reporting-only and never changes the budget.

## Data — three fresh splits

The binding constraint is cross supply: only **30 fresh diagonal witnessed-cross**
scenarios remain after excluding everything spent through 2f (regular/asymmetric are
procedurally abundant). Three fresh, disjoint splits — train, dev, and gate — must
share that pool. Cross uses the 2f diagonal generator (`generate_expanded_cross`:
diagonal headings `±π/4, ±3π/4`, `dist ≥ 12`, outward-road-end filtered,
**waypoint-witnessed**); regular/asym use `generate_stage2b_split` (cardinal, unchanged).

| split | cross | regular | asym | total | seed | role |
|---|---|---|---|---|---|---|
| **train** | 6 | 30 | 30 | 66 | `STAGE2G_TRAIN_SEED = 70` | CEM objective + go/no-go |
| **dev**   | 6 | 20 | 20 | 46 | `STAGE2G_DEV_SEED = 71`   | escalation + selection only |
| **gate**  | 12 | 44 | 44 | 100 | `STAGE2G_GATE_SEED = 72` | one-shot, scored once |

Cross total 6+6+12 = **24 ≤ 30** (6 spare). Gate keeps 2f's 12-cross shape, so the
cross stratum stays directly comparable to 2f (same diagonal generator) and the
`≥0.80` cross bar needs **10/12**. Train is smaller than 2f's 86 to keep the
higher-dimensional search's wall-clock bounded while retaining cross exposure.

**Freeze + hash before training.** All three splits are generated, frozen to
`runs/stage2g/`, and content-hashed (sha256) **before any optimization begins**.
Generation order **train → dev → gate**, each passing the earlier frozen splits into
its exclusion set so disjointness holds *by construction*, then re-asserted on
`_scenario_key`. The **gate is generated last and never inspected** — neither its
scenarios nor its outcomes are read until the single one-shot event.

**Exclusion set (asserted, fail loudly).** Every fresh split is exact-identity
disjoint from: the complete corrected prior set + all spent 2d/2e/2f splits (fitness
and gate) + the **`runs/stage2b/sm_train_split.json` BC-demo split** (per the
amendment: the BC demonstration data is training footprint, so no fresh split — and
in particular the gate — shares a scenario with it) + each other. A shared
`_scenario_key` with any excluded split fails the run.

**Diagonal-cross comparability caveat (carried from 2f).** The cross stratum uses
diagonal start orientations and is witness-filtered, so cross is **not
orientation-comparable to prior cardinal stages**; it *is* comparable to 2f (same
generator). Regular and asymmetric remain trend-comparable across stages. Any pass
conclusion on cross is scoped to the diagonal, waypoint-witnessed eligible stratum.

## Escalation & selection (predeclared)

The gate stays **sealed through all optimization and escalation**. Only the finally
selected configuration touches it, exactly once.

1. **Run n=8** (both arms). The recurrent arm drives the escalation decision.
2. **Escalation trigger (gate-aligned, dev-based).** Escalate iff the trained n=8
   **recurrent** model fails go/no-go, **or** fails to improve **dev arrivals** over
   its own warm start. Train-only improvement does **not** suppress escalation —
   the held-out dev arrivals are the trigger metric.
3. **Escalation target — not implemented on this spec.** If the trigger fires, the
   run **halts before touching the gate** and a short **spec amendment** must be
   written first, freezing the n=16 + CMA-ES escalation's implementation, budget,
   initialization, and block scaling. No unspecified conditional optimizer ships:
   CMA-ES is not built until that amendment exists. (Rationale for the direction:
   n=16 = 486 params, and CMA-ES's full covariance suits the higher-dimensional
   search; but the details are frozen in writing before any code, on the same
   train/dev splits, gate still sealed.)
4. **Selection (if both sizes run).** Choose the configuration with the higher
   gate-aligned dev score `min(overall/0.90, cross/0.80, regular/0.80, asym/0.80)`
   computed on **dev**; tie-break toward **n=8**. Selection is decided on the
   recurrent model.
5. **Gate event.** The selected configuration's **recurrent and recurrence-off
   ablation are scored together** in the single one-shot gate event.

The six spare cross scenarios are held in reserve; the dev cross stratum (6) is
coarse, but the gate-aligned selector and the conservative n=8 tie-break contain
that risk.

## Gate & interpretation

**Fresh one-shot gate.** PASS = **≥0.90 overall** arrival rate **AND ≥0.80 per
layout** (regular, asymmetric, cross); with 12 cross the cross bar needs 10/12.
Scored once for the selected config's recurrent and ablation arms.

- **Recurrent passes AND beats ablation (> 0.02)** → a trainable recurrent policy
  over the observation interface clears a fresh gate, and learned temporal state
  carries the advantage. **Scoped:** on this diagonal, waypoint-witnessed cross
  stratum plus the cardinal regular/asym strata, the interface is not the
  bottleneck. Attribution caveat stands (no matched n=8 readout-only control; the
  n=64→n=8 change is bundled). Motivates the biologically grounded
  memory/action-selection work.
- **Recurrent passes, ablation also passes** → aligned reward exploited the reactive
  policy state; **recurrence is not necessary within this gate and policy family**
  (n=8, this observation interface, this reward) — not a general claim that memory
  is never required.
- **Both fail** → bounded and confounded between optimization budget, capacity
  (n=8), and coverage. Report the recurrent−ablation gap and collision/arrival
  breakdown; **not** an interface verdict.

**Determinism (no cross-stage reproduction fingerprint).** Unlike 2e/2f there is no
prior-stage 22/40 reconstruction to match — the network is new. Instead: assert the
warm-start policy is **deterministic** (same seed → bit-identical θ₀), and **log the
warm-start arrival count on the train split** as a sanity value (the θ₀ baseline the
CEM improves over), persisted per arm.

## Files

- `stage2g.py` —
  - `make_net(recurrent: bool) -> EchoStateNetwork` (n=8, 2c recipe, seed 0).
  - `warm_start_theta(recurrent, layouts) -> (esn, theta0, block_slices)`: ridge-fit
    `W_out` on `sm_train_split.json` demos; return the flattened warm start and the
    per-block index slices for the trainable blocks.
  - `block_scales(theta0, block_slices) -> np.ndarray`: `0.1 × max(RMS(θ₀_b), 1e-3)`
    per block, broadcast to a per-parameter scale vector.
  - `train_by_reward_blockscaled(esn, theta0, block_scales, scenarios, layouts,
    cfg, fitness_fn) -> (esn, info)`: `z`-space CEM wrapper (`θ = θ₀ + scale ⊙ z`),
    warm-start guard, best-ever tracking; records recorded block scales and
    warm-start train outcomes. Reuses `stage2d.cem_maximize` unchanged and
    `stage2f.aligned_fitness`.
  - `go_no_go(esn, theta0, block_scales, train_scenarios, layouts, cfg) -> dict`:
    one-iteration probe at the run seed; reports `arrivals`, `beats_noop`, `passed`.
- `evaluate_stage2g.py` —
  - Frozen seeds `STAGE2G_TRAIN_SEED=70`/`STAGE2G_DEV_SEED=71`/`STAGE2G_GATE_SEED=72`;
    build train → dev → gate (each excluding the earlier + full exclusion set +
    `sm_train_split.json`), freeze to `runs/stage2g/`, assert pairwise-disjoint,
    record sha256 of all splits **before training**.
  - Per arm: warm start → block scales → go/no-go → full 30-iter block-scaled CEM.
  - Escalation check on **dev** (recurrent arm); if triggered, **halt with a clear
    message to write the CMA-ES amendment** — do not proceed to the gate.
  - Selection (dev `min(...)` if both sizes ran) → one-shot gate for recurrent +
    ablation of the selected config; persist best-ever θ, block scales, warm-start
    train outcomes, arrival/collision/timeout counts, provenance sha256; write
    `runs/stage2g/{results,gate_results}.json`.
- `test_stage2g.py` (TDD) — n=8 recurrent has 190 trainable params / ablation 126,
  `W≡0` & leak==1 for the ablation; block scales equal `0.1·max(RMS,1e-3)` per block
  and are recorded; the `z`-space wrapper maps `z=0 → θ₀` and leaves `cem_maximize`
  unmodified; the go/no-go probe is a one-iteration run reproduced by iteration 0 of
  the full run at the same seed; warm start is deterministic (bit-identical θ₀ on
  reseed); train/dev/gate are 66/46/100, correct strata, exact-identity disjoint
  from each other + all priors + spent 2d/2e/2f + `sm_train_split.json`; splits are
  hashed before training; escalation triggers on a dev-arrivals non-improvement and
  halts before the gate.
- `runs/stage2g/{train_split,dev_split,gate_split,results,gate_results}.json` +
  `README.md`.

Does **not** touch the connectome or the frozen Stage 2/2b/2c/2d/2e/2f artifacts; no
spent gate is rerun or replaced.

## Out of scope

- **CMA-ES / n=16 is not implemented** this stage. It is a conditional escalation
  that requires its own frozen spec amendment before any code (see Escalation §3).
- No connectome/dopamine learning; no map/planner on the observation-only controller.
- No matched n=8 readout-only control (the exact-attribution instrument) — the
  attribution caveat is documented instead.
- Connectome-grounded learning follows only if a pass settles the memory question.
