# Stage 2f — aligned-reward warm-start fine-tuning (route-memory gate, take 4)

**Verdict: bounded FAILURE.** Neither model clears the fresh 0.90/0.80 gate, so
per the pass-only-is-conclusive asymmetry this is **not** an interface verdict.
The failure is confounded between the two things 2f changed at once — reward
alignment and scenario coverage — and residually capacity/optimization. **Do
not** advance to connectome/dopamine learning on this result.

The stage's real contribution is that the arrival-primary fitness fix
**worked as designed**: it repaired the reward/arrival misalignment Stage 2e
diagnosed, and reward now moves *with* arrivals instead of against them. It
just was not, by itself, sufficient to clear the gate.

## What this stage asked

Stage 2e escaped the 2d bootstrap failure (warm-starting CEM at the
deterministic Stage 2c readout) and produced actively-driving policies, but
diagnosed three concrete gaps: (1) **reward ≠ arrival-rate** — the memoryless
ablation's CEM *raised* reward while arrivals *fell* (22 → 18); (2) collisions
dominated the misses; (3) a train→gate generalization gap from optimizing
against only 40 fixed scenarios. Stage 2f addresses (1) and (3) directly.

**1. Arrival-primary fitness.** CEM now optimizes

```
progress   = clip((d0 - d_final)/d0, -1, 1)
secondary  = 0.5·progress - 2.0·[collision] - 0.1·(elapsed/MAX_SIM_TIME)
fitness(θ) = arrivals(θ) + mean_over_episodes(secondary) / 4.0
```

with the shaping coefficients frozen from Stage 2e's *results* (never tuned
on the Stage 2f gate or fitness set). Per-episode `secondary` is bounded to
`[-2.6, 0.5]`, so its mean-then-÷4 contribution swings at most `0.775 < 1` —
**one additional arrival always outweighs the shaping term.** This directly
fixes the concrete 2e failure: under the old plain weighted-sum reward, the
actual 2e policies score warm-start 22-arrival = **1.555** < 2e-best
18-arrival = **1.686** — the optimizer preferred *fewer* arrivals. Under
arrival-primary fitness, 22 > 18 unconditionally.

**2. Larger, stratified fitness set.** 86 scenarios (6 cross / 40 regular /
40 asymmetric) versus 2e's 40, drawn from a fresh Stage 2f fitness split, to
fit a broader sample and reduce overfitting to a small fixed set.

**3. Expanded, waypoint-witnessed cross gate.** The cardinal-heading cross
pool is exhausted (all 64 eligible scenarios consumed by Stages 2/2b/2d/2e).
The only source of fresh cross is adding diagonal start headings (`±π/4,
±3π/4`), filtered to **waypoint-witnessed** scenarios (the privileged witness
arrives — a positive filter, never a solvability claim). This yields 12 fresh
diagonal cross scenarios for the gate (48 available). **Documented
consequence: cross is diagonal-heading and not orientation-comparable to
prior stages' cardinal cross** — cross trend lines across stages are invalid.
Regular and asymmetric generation is unchanged and remains trend-comparable.

Everything else is held from 2e: warm start at the deterministic 2c readout,
the frozen 2c-winner reservoir (n=64, sr=0.8, leak=0.5, in=1.0, seed=0),
matched memoryless ablation, CEM with best-ever tracking and a warm-start
guard, identical budget/seeds for both models, no config selection.

**Reproduction gate (asserted before optimizing, on the fixed 2d-40
reference set, coefficient-independent — mean_reward under the *old* 2e
metric):** recurrent 22/40 arrivals @ reward 1.106, ablation 22/40 @ 1.086 —
held bit-exact, confirming the 2c readout was reconstructed exactly.

## Results (fresh one-shot gate: 12 cross / 44 regular / 44 asymmetric = 100)

| Model | Overall | cross | regular | asym | Fitness (train) | Gate pass |
|---|---|---|---|---|---|---|
| Recurrent | **0.660** | 0.75 (9/12) | 0.682 (30/44) | 0.614 (27/44) | warm 54.911 → **66.983** (55→67 arrivals, 26→15 collisions) | ✗ |
| Memoryless ablation | **0.540** | 0.75 (9/12) | 0.523 (23/44) | 0.500 (22/44) | warm 57.902 → **59.933** (58→60 arrivals) | ✗ |

References: waypoint witness 1.00, Stage 2b SM 0.62, Stage 2c recurrent 0.57,
Stage 2d recurrent 0.00, Stage 2e recurrent 0.516. Gate threshold 0.90 overall
/ 0.80 per layout (cross needs 10/12 at n=12; both models land at 9/12).

Recurrent gate outcomes: 66 arrivals, 29 collisions, 5 timeouts. Ablation gate
outcomes: 54 arrivals, 44 collisions, 2 timeouts. Wall clock 145.9 min.

## What this establishes (and what it does not)

**1. The aligned-fitness fix worked as designed.** This is the stage's real
contribution. In 2e, under a plain weighted-sum reward, the ablation's CEM
raised reward while arrivals *fell*. Here reward and arrivals move
*together*: the recurrent model's aligned fitness rose 54.911 → 66.983 on the
86-scenario fitness set **and its arrivals rose 55 → 67 while collisions fell
26 → 15 over the same run.** The ablation shows the same pattern (fitness
57.902 → 59.933, arrivals 58 → 60). The arrival-primary objective now tracks
the gate metric it was designed to track — it just was not sufficient, alone,
to reach 0.90.

**2. Recurrent continues to beat its memoryless ablation.** On the gate,
0.660 vs 0.540 (+0.120 overall), and the recurrent model itself improved on
its 2e gate score (0.516 → 0.660). The gap lives in the well-sampled cardinal
strata — regular 0.682 vs 0.523, asymmetric 0.614 vs 0.500 — both favoring
recurrent, continuing the direction first seen in 2e (regular 0.455 vs 0.364,
asymmetric 0.568 vs 0.409). Cross is tied at 9/12 = 0.75 for both models and
is too coarse (n=12) to read. **This is a hint, not a verdict:** per the
asymmetry only a *pass* is conclusive, and a gap between two *failing*
policies can still reflect capacity or optimization rather than memory per
se. Do not over-read it.

**3. Neither clears the gate — bounded and confounded, not an interface
verdict.** Stage 2f changed two things at once (reward alignment + a larger
fitness set), so a failure cannot be attributed to either individually, and
residual capacity/optimization limits are not ruled out. The result does
**not** implicate the observation interface. The gap to a pass is
diagnosable:

- **Collisions still dominate the recurrent misses.** 29 of the 34
  non-arrival gate trials (85%) are collisions, not timeouts — the policy
  still crashes far more than it stalls.
- **A train→gate generalization gap persists.** The recurrent model reached
  67/86 ≈ 78% on the fitness set it was optimized against, but only 66% on
  the fresh gate — a ~12-point drop, narrower than 2e's ~13-point drop from a
  smaller (40-scenario) fitness set but still present.
- **Cross stratum caveat.** The 2f cross gate is diagonal-heading and
  waypoint-witnessed (a positive filter, never a solvability claim) — it is
  **not** orientation-comparable to prior stages' cardinal cross. Any scoped
  interpretation would be limited to the diagonal witnessed cross plus the
  cardinal regular/asymmetric strata; there is no such scoped pass to report
  here since both models fail overall.

## Bounded conclusion

The arrival-primary fitness fixed the reward/arrival misalignment Stage 2e
diagnosed — reward and arrivals now rise together for both models, and the
recurrent model's fitness-set collisions fell as arrivals rose — but
**neither model clears the fresh 0.90/0.80 gate**, so the route-memory
question is **not** settled. Because 2f bundled two changes (aligned fitness
+ a larger fitness set), this failure is confounded between reward alignment
and scenario coverage, and residually between capacity and optimization; it
says nothing dispositive about whether the observation interface can support
the gate. The recurrent > ablation gap (0.660 vs 0.540, continuing 2e's 0.516
vs 0.400) is a hypothesis-supporting hint that memory helps under aligned
reward, to be confirmed only by an eventual pass.

## What would move the needle next (not yet run, not yet approved)

Aimed at the diagnosed gaps, not at the interface, and each its own
predeclared fresh-gate experiment:

- **Stronger collision shaping** (raise `W_collide` further, or an explicit
  near-miss cost) — collisions, not timeouts, now dominate the recurrent
  misses (29/34).
- **Close the remaining train→gate generalization gap** — the 86-scenario set
  narrowed but did not eliminate the ~12-point fitness→gate drop; a still
  larger or curriculum fitness set is a candidate.
- **More CEM budget or a stronger optimizer (CMA-ES)** now that reward tracks
  arrivals, to test whether the recurrent advantage widens toward a pass.

**Do not** bolt a map/planner onto the observation-only controller, and **do
not** advance to connectome/dopamine learning on the strength of a failure —
the memory question is still open, even though 2f made real progress fixing
the reward signal.

## Provenance

- Fresh gate `runs/stage2f/gate_split.json` sha256
  `6da48c3151d9722fa944c816cf52bab3df86065a08b715265b7297ba3ea0f257`
  (frozen **second**, seed 61, excluding the fitness split). Fitness
  `runs/stage2f/fitness_split.json` sha256
  `cd733968fc18bda4a7109643de330b2f374ba6a9359961eff62836ce0142d38c`
  (frozen **first**, seed 60). Counts: fitness 6/40/40 = 86, gate
  12/44/44 = 100.
- **Fully exact-identity disjoint** from the complete corrected prior set,
  the spent Stage 2d gate + train splits, the spent Stage 2e gate, and each
  other — asserted at build time (`evaluate_stage2f._assert_disjoint`) and in
  `test_stage2f.py`.
- Cross stratum: expanded to diagonal start headings (`±π/4, ±3π/4`),
  `dist ≥ 12`, outward-road-end filtered, and **waypoint-witnessed**
  (privileged-witness positive filter, not a solvability claim). **Not**
  orientation-comparable to prior stages' cardinal cross. Regular and
  asymmetric generation is unchanged (`generate_stage2b_split`) and remain
  trend-comparable across stages.
- Warm start = deterministic Stage 2c readout, reconstructed via
  `evaluate_stage2c.collect_demos` + `_fit`; reproduction asserted (22/40
  arrivals, coefficient-independent) before optimizing.
- Frozen CEM budget (identical to 2e, unconditional `n_iter=25`, no
  timing-driven adjustment): population 64, n_iter 25, elite_frac 0.20,
  `init_std=0.1`, `std_floor=0.001`, seed 0, identical for recurrent and
  ablation. Wall clock 145.9 min.
- Best-ever `W_out`, per-iteration history, warm-start and best fitness-set
  outcome counts persisted in `results.json`; gate outcomes in
  `gate_results.json`. Spec:
  `docs/superpowers/specs/2026-09-17-stage2f-aligned-reward-design.md`. Plan:
  `docs/superpowers/plans/2026-09-17-stage2f-aligned-reward.md`.
