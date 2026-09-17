# Stage 2e — warm-start reward fine-tuning (route-memory gate, take 3)

**Verdict: bounded FAILURE — but a large, informative one.** Neither model clears
the fresh 0.90/0.80 gate, so per the pass-only-is-conclusive asymmetry this is
**not** an interface verdict. Two things did change decisively versus Stage 2d:
warm-start reward fine-tuning **escaped the optimization floor** (recurrent gate
**0.00 → 0.516**, zero timeouts — the car now actively drives), and the recurrent
net **beat its memoryless ablation** (0.516 vs 0.400) — the first memory signal in
the Stage 2 line, though a gap between two *failing* policies is a hint, not proof.

## What this stage asked

Stage 2d showed CEM-from-`μ=0` could not bootstrap goal-reaching behavior that was
demonstrably reachable by this exact 150-parameter policy class. Stage 2e removed
that one failure mode: initialize the CEM search at the deterministic Stage 2c
readout and fine-tune small perturbations (`init_std=0.1`) around it, so the sparse
arrival term has gradient signal from the first iteration. Everything else is held
fixed: the 10-feature observation interface, the frozen 2c-winner reservoir
(n=64, sr=0.8, leak=0.5, in=1.0, seed=0), the same evaluator-only bounded-net-
progress reward, and the recurrent-vs-ablation comparison with identical budget and
seeds.

**Reproduction gate (asserted before optimizing):** the warm start reproduced the
known-policy check bit-exact — recurrent 22/40 arrivals @ reward 1.106, ablation
22/40 @ 1.086 — confirming the 2c readout was reconstructed exactly.

## Results (fresh one-shot gate: 7 cross / 44 regular / 44 asymmetric = 95)

| Model | Overall | cross | regular | asym | Fitness (train) | Gate pass |
|---|---|---|---|---|---|---|
| Recurrent | **0.516** | 0.571 (4/7) | 0.455 | 0.568 | warm 1.106 → **1.481** (26/40) | ✗ |
| Memoryless ablation | **0.400** | 0.571 (4/7) | 0.364 | 0.409 | warm 1.086 → 1.122 (18/40) | ✗ |

References: waypoint witness 1.00, Stage 2b SM 0.62, Stage 2c recurrent 0.57,
**Stage 2d recurrent 0.00**. Gate threshold 0.90 overall / 0.80 per layout.

Recurrent gate outcomes: 49 arrivals, 46 collisions, **0 timeouts**. Ablation:
38 arrivals, 28 collisions, 29 timeouts.

## What this establishes (and what it does not)

**1. Warm start fixed the 2d bootstrap failure — decisively.** The recurrent model
went from 0% (2d, from-zero) to 51.6% on a *fresh, disjoint* gate, with **zero
timeouts**: every episode now ends in an arrival or a collision, i.e. the policy
drives purposefully rather than sitting still. This confirms 2d's 0% was an
optimization-bootstrap artifact, exactly as the known-policy check predicted — not
a property of the interface. CEM improved the recurrent fitness 1.106 → 1.481
(26/40 arrivals on the fitness set).

**2. The recurrent net beat its memoryless ablation.** On the gate, 0.516 vs 0.400
(+0.116 overall). The gap lives in the two well-sampled strata — regular
0.455 vs 0.364, asymmetric 0.568 vs 0.409 (cross is tied at 4/7 and too coarse to
read) — so it is not sampling noise on ~88 scenarios. This is the **first** time in
the Stage 2 line that added temporal memory produced a clear closed-loop advantage.
**It is a hint, not a verdict:** per the asymmetry we committed to, only a *pass* is
conclusive, and a gap between two policies that both fail can still reflect
capacity or optimization rather than memory per se. Do not over-read it.

**3. Neither clears the gate — bounded and confounded.** Both fall well short of
0.90. The result does **not** implicate the observation interface. The gap to a
pass is diagnosable and points at optimization/generalization, not blindness:

- **Collisions dominate the misses.** The recurrent policy collides 46 of 95 times
  — it arrives or crashes, rarely wanders. Reward fine-tuning pushed arrivals up
  but the collision penalty (`W_collide=1.0`) did not suppress crashes enough.
- **Train→gate generalization gap.** Recurrent reached 26/40 = 65% on the fitness
  set but 51.6% on the fresh gate — a ~13-point drop from optimizing against only
  40 fixed scenarios.
- **Reward ≠ arrival rate, especially for the ablation.** The ablation's CEM raised
  *reward* (1.086 → 1.122) while *arrivals fell* (22 → 18, timeouts 1 → 16): the
  net-progress/time terms rewarded a more cautious, lower-arrival optimum. The
  shaping and the gate metric are not perfectly aligned.

## Bounded conclusion

Warm-start reward fine-tuning removed Stage 2d's bootstrap failure and produced
actively-driving policies, with a genuine recurrent-over-ablation advantage (0.516
vs 0.400) — but **neither model clears the fresh 0.90/0.80 gate**, so the route-
memory question is **not** settled. Per the pass-only-is-conclusive asymmetry this
says nothing dispositive about whether the observation interface can support the
gate; the shortfall is confounded by collisions, a train→gate generalization gap,
and reward/metric misalignment — all optimization/generalization issues, not the
interface. The recurrent>ablation gap is a hypothesis-supporting hint that memory
helps under reward, to be confirmed only by an eventual pass.

## What would move the needle next (not yet run, not yet approved)

Aimed at the diagnosed gaps, not at the interface, and each its own predeclared
fresh-gate experiment:

- **More / curriculum fitness scenarios** (beyond 40 fixed) to close the train→gate
  generalization gap.
- **Stronger collision shaping** (raise `W_collide`, or an explicit near-miss cost)
  and/or **arrival-weighted fitness** so reward tracks the gate metric — re-checking
  that shaping stays unfarmable and standing-still stays dominated.
- **More CEM budget / a stronger optimizer** (CMA-ES) now that the regime is
  behaving, to test whether the recurrent advantage widens toward a pass.

**Do not** bolt a map/planner onto the observation-only controller, and **do not**
advance to connectome/dopamine learning on the strength of a failure — the memory
question is still open, even though 2e made real progress on it.

## Provenance

- Fresh gate `runs/stage2e/gate_split.json` sha256 `5b5e24b6cab8368b…` (frozen
  before the run). **Fully exact-identity disjoint** (unlike the spent 2d gate)
  from the complete corrected prior set, the spent Stage 2d gate + train splits,
  and the 40-scenario fitness set — asserted at build time (`evaluate_stage2e.
  build_gate_split`) and in `test_stage2e.py`. 7/44/44 = 95 uses all 7 remaining
  eligible cross; the 0.80 cross gate needs 6/7 (documented coarse stratum).
- Fitness = the exact 40 Stage 2d fitness scenarios (`runs/stage2d/train_split.json`,
  4/18/18), which the fresh gate fully excludes → gate ∩ fitness = ∅.
- Warm start = deterministic Stage 2c readout (winner hp incl. `ridge=1e-4`),
  reconstructed via `evaluate_stage2c.collect_demos` + `_fit`; reproduction
  asserted (22/40, reward ≈1.1) before optimizing.
- Frozen warm-start CEM budget: population 64, n_iter 25, elite_frac 0.20,
  `init_std=0.1`, identical seeds for recurrent and ablation. Wall clock 106.8 min.
- Best-ever `W_out`, per-iteration history, and fitness-set outcome counts persisted
  in `results.json`. Design:
  `docs/superpowers/specs/2026-09-17-stage2e-warmstart-reward-design.md`.
