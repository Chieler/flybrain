# Stage 2h — PPO recurrent-policy test (readiness FAILED; gate unspent)

**Date:** 2026-09-24
**Branch:** `stage2h-ppo`
**Question under test:** Was Stage 2g primarily an *optimizer* failure? Stage 2h keeps
the simulator, the ten-feature observation boundary, the continuous controls, and the
pass-only interpretation unchanged, and swaps the black-box weight search for
gradient-trained PPO — recurrent (`sb3_contrib.RecurrentPPO`, `MlpLstmPolicy`) vs.
feed-forward (`stable_baselines3.PPO`, `MlpPolicy`).

## Outcome (one line)

**Both PPO arms failed the fresh 0.90/0.80 readiness split; the Stage 2h gate was never
opened and remains unspent.** No interface, memory, or recurrence conclusion is licensed.
Do not advance to dopamine/connectome learning.

## Environment / budget (frozen, preregistered)

- Dependencies: torch 2.14.0, gymnasium 1.3.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0
  (Python 3.12, CPU device). Run in the isolated `.venv-rl`.
- Per arm and seed: seeds (0, 1, 2), 1,000,000 timesteps, 8 `DummyVecEnv` envs,
  `n_steps=256`, `batch_size=256`, `n_epochs=5`, `learning_rate=3e-4`, `gamma=0.995`,
  `gae_lambda=0.95`, `ent_coef=0.01`. No hyperparameter sweep; hyperparameters were
  **not** altered after seeing any run.
- Reward (evaluator-only, never in observations):
  `2*(d_prev - d_new)/d_start - 0.001` per step, `+10` on arrival, `-4` on collision.
- Splits (exact-identity disjoint from all 13 prior Stage 2 splits and from each other,
  each scenario positively witnessed by `WaypointController`):
  train seed 80 (450 = 150/150/150), readiness seed 81 (300 = 100/100/100),
  gate seed 82 (300 = 100/100/100). Hashes in `split_provenance.json` and `results.json`.

## Model selection (spent 2f/2g gates only)

All three seeds of **both** arms scored **0/100 arrivals on the Stage 2f gate and
0/100 on the Stage 2g gate** (every episode a timeout; zero collisions). Selection
score (worst normalized rate across the two spent gates) was 0.000 for all six
checkpoints, so selection fell to the tie-breaker (lowest seed): **seed 0 for both arms.**
Fresh readiness/gate data were not used to select.

## Readiness (fresh split, scored once)

| Arm (selected seed 0) | overall | cross | regular | asymmetric | arrivals | collisions | timeouts | passes 0.90/0.80 |
|-----------------------|--------:|------:|--------:|-----------:|---------:|-----------:|---------:|:-----------------:|
| Recurrent PPO         |   0.000 | 0.000 |   0.000 |      0.000 |    0/300 |          0 |      300 | **no** |
| Feed-forward PPO      |   0.000 | 0.000 |   0.000 |      0.000 |    0/300 |          0 |      300 | **no** |

`gate_open = False`. Because the recurrent arm did not reach overall ≥0.90 with every
layout ≥0.80, the one-shot gate was **not** run and `gate_results.json` does not exist.

## What actually happened (diagnosis)

Both policy families converged to a **degenerate "freeze" local optimum**: the
deterministic policy commands full braking, the car never leaves the start, and every
episode ends in a timeout (never a collision, never an arrival) — including on the
policies' *own* training scenarios. With `PHYSICS_DT = 0.02 s`, episodes run up to 2250
steps and the `+10` arrival is hundreds of steps away and rare under exploration, while
the `-4` collision penalty is immediate and common. Standing still (per-step `-0.001`,
bounded total loss) dominates both risky exploration and the unfound arrival, so PPO
settles on not moving. This is an optimization/exploration failure under the frozen
budget, not an observation-interface failure.

**The environment, reward, and scoring were verified correct end-to-end**: a scripted
constant-forward action on the straight cross scenario arrives through
`StreetNavigationEnv` in 713 steps with total reward ≈ 11.2 and the terminal `+10`
bonus present. So the 0.000 rates reflect the learned policies, not a broken harness.

## Bounded interpretation (pass-only asymmetry)

- **PPO did not earn a gate attempt.** Neither arm cleared the fresh readiness bar, so
  under the pass-only asymmetry no positive claim is licensed.
- The failure is bounded to **policy family / optimization / exploration / generalization**
  and does **not** implicate the ten-feature observation interface (which is provably
  sufficient for a scripted controller and for `WaypointController`, the split witness).
- Because the memoryless feed-forward arm also failed, **no** claim about the necessity
  or sufficiency of recurrence is made.
- The original question ("was Stage 2g primarily an optimizer failure?") is **not
  answered**: switching to gradient PPO under this frozen budget produced a different but
  still-failing optimizer, so it neither confirms nor refutes the Stage 2g optimizer
  hypothesis.
- **Do not** start dopamine/connectome learning; that step was gated on a Stage 2h pass,
  which did not occur.

## Artifacts

- `results.json` — dependency versions, budget/hyperparameters, split hashes, per-seed
  spent-gate scores, selected seeds + model hashes, readiness scores, `gate_open`.
- `train_split.json`, `readiness_split.json`, `gate_split.json`, `split_provenance.json`
  — the frozen witnessed splits and their provenance hashes.
- `models/` (git-ignored, local only) — six `*.zip` checkpoints; sha256 recorded in
  `results.json`. The gate split was **not** scored; `gate_results.json` is absent.
