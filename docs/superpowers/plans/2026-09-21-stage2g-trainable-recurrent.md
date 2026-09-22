# Stage 2g — fully-trainable recurrent baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an n=8 echo-state network end-to-end (input + recurrent + readout weights) by block-scaled CEM against the frozen Stage 2f aligned reward, with a matched recurrence-off ablation, on fresh train/dev/gate splits scored once at the 0.90/0.80 gate.

**Architecture:** Two new modules mirroring the Stage 2d–2f pattern. `stage2g.py` holds the model machinery — build the n=8 net, BC-initialize a flat weight vector `θ₀`, derive per-block perturbation scales, and fine-tune all trainable weights by a reparameterized (`θ = θ₀ + block_scale ⊙ z`) CEM search around `stage2d.cem_maximize`, which is reused unmodified. `evaluate_stage2g.py` builds/freezes/hashes the three splits before any training, runs each arm through a go/no-go probe and the full run, applies a dev-arrivals escalation halt (CMA-ES is *not* implemented — a halt demands a written amendment), and scores the sealed gate once. Reward is evaluator-only privileged geometry, never a controller input.

**Tech Stack:** Python 3, NumPy 2.5.1, SciPy 1.18.0 (no autograd, no torch/JAX — the constraint forcing black-box CEM over gradient methods). pytest for TDD.

**Spec:** `docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md`

## Global Constraints

- **No torch/JAX/autograd.** Training is black-box CEM over the non-differentiable episode reward; reuse `stage2d.cem_maximize` unmodified (a reparameterization *wrapper* is allowed; editing `cem_maximize` is not).
- **Model:** n=8 EchoStateNetwork, 2c recipe (`spectral_radius=0.8, leak=0.5, input_scale=1.0, seed=0`). Recurrent arm trains `W_in`(8×11)+`W`(8×8)+`W_out`(2×19) = **190 params**; ablation trains `W_in`+`W_out` = **126 params** with `W≡0` held fixed and leak forced to 1.
- **Block scaling:** `scale_b = 0.1 × max(RMS(θ₀_b), 1e-3)` per trainable block; CEM searches z-space with `init_std=1.0`; resulting scales recorded in the artifact.
- **Frozen CEM budget (both arms identical):** `population=64, n_iter=30, elite_frac=0.20, init_std=1.0, std_floor=0.001, seed=0`. Fixed unconditionally; any timing smoke is report-only.
- **Fitness:** `stage2f.aligned_fitness` unchanged (`arrivals + mean(secondary)/4`). Privileged geometry enters the reward only.
- **Splits:** train(seed 70, 6/30/30=66), dev(seed 71, 6/20/20=46), gate(seed 72, 12/44/44=100). Diagonal witnessed cross via `evaluate_stage2f.generate_expanded_cross`; regular/asym via `generate_stage2b_split`. Generated train→dev→gate, each excluding the earlier; **frozen and sha256-hashed before any training.**
- **Exclusion set — Variant B (2026-09-21 amendment; asserted, fail loudly):** every newly frozen, Stage-2g-policy-unseen split is exact-identity disjoint (`_scenario_key`) from the **enforced-disjoint set** — every prior **scored gate** (`runs/stage2b`, `2d`, `2e`, `2f` gate splits) + **2f fitness** (`runs/stage2f/fitness_split.json`) + `runs/stage2b/sm_train_split.json` (the BC-demo split) + Stage 2g's own earlier splits + each other. The full prior exclusion was **infeasible** (17 regular scenarios). Older prior **training** splits may recur — historical design exposure, **not** Stage 2g policy-training leakage. Overlap counts vs **every** prior split are recorded; enforced overlaps asserted zero.
- **Escalation is a hard halt.** Escalate iff the trained n=8 recurrent model fails go/no-go OR fails to improve **dev arrivals** over its warm start. On escalation the run stops before the gate; n=16 + CMA-ES is **not implemented** — it requires a separate spec amendment first.
- **Gate:** PASS = overall ≥0.90 AND every layout ≥0.80 (cross needs 10/12). The gate is generated last, never inspected until the single scoring event.
- **Determinism:** same seed → bit-identical `θ₀`; log warm-start train arrivals per arm (no cross-stage reproduction fingerprint — the network is new).
- **Honest attribution (in artifacts):** n=64→n=8 is itself architectural; not one-variable-vs-2f; exact attribution would need a matched n=8 readout-only control (not run). "Both pass" is bounded to "recurrence is not necessary *within this gate and policy family*."
- Does **not** touch the connectome or any frozen Stage 2/2b/2c/2d/2e/2f artifact; no spent gate is rerun.

---

## File Structure

- **`stage2g.py`** (new) — model machinery: `make_net`, weight (de)serialization, `warm_start_theta`, `block_scales`, `NoOpController`, `go_no_go`, `train_by_reward_blockscaled`. Pure functions over an `EchoStateNetwork`; no split/gate logic.
- **`evaluate_stage2g.py`** (new) — orchestration: exclusion set, split building/freezing/hashing, per-arm run, dev-arrivals escalation halt, one-shot gate, interpretation, persistence, `main`.
- **`test_stage2g.py`** (new) — TDD for both modules; tests use tiny scenario sets and `population=2, n_iter=1` to stay fast, never the full budget.
- **`runs/stage2g/{train_split,dev_split,gate_split,results,gate_results}.json`** + **`README.md`** — produced by a real run (not committed by the plan tasks unless the run is executed).

---

### Task 1: Model core — build net, (de)serialize weights, block scales

**Files:**
- Create: `stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `stage2c.EchoStateNetwork`, `stage2c.N_FEATURES` (=10).
- Produces:
  - `N8_CONFIG: dict` = `{"n_reservoir":8,"spectral_radius":0.8,"leak":0.5,"input_scale":1.0}`, `ESN_SEED=0`, `RIDGE=1e-3`.
  - `make_net(recurrent: bool) -> EchoStateNetwork`
  - `flatten_theta(esn, recurrent: bool) -> np.ndarray` — order `[W_in, (W if recurrent), W_out]`.
  - `set_theta(esn, theta: np.ndarray, recurrent: bool) -> None` — inverse of `flatten_theta`; leaves `esn.W` untouched (zeros) when `recurrent=False`.
  - `trainable_slices(esn, recurrent: bool) -> list[tuple[int,int]]` — `(start,stop)` per trainable block in flatten order.
  - `block_scales(theta: np.ndarray, slices) -> np.ndarray` — per-parameter scale vector, `0.1·max(RMS(block),1e-3)` broadcast within each block.

- [ ] **Step 1: Write the failing tests**

```python
# test_stage2g.py
import numpy as np
import stage2g


def test_recurrent_net_has_190_trainable_params():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])  # give W_out a shape
    theta = stage2g.flatten_theta(esn, recurrent=True)
    assert theta.size == 8 * 11 + 8 * 8 + 2 * 19  # 88 + 64 + 38 == 190


def test_ablation_net_has_126_trainable_params_and_zero_frozen_W():
    esn = stage2g.make_net(recurrent=False)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = stage2g.flatten_theta(esn, recurrent=False)
    assert theta.size == 8 * 11 + 2 * 19            # 88 + 38 == 126
    assert np.array_equal(esn.W, np.zeros((8, 8)))  # W removed from search
    assert esn.leak == 1.0                          # memoryless


def test_set_theta_is_inverse_of_flatten():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = np.arange(190, dtype=float)
    stage2g.set_theta(esn, theta, recurrent=True)
    assert np.array_equal(stage2g.flatten_theta(esn, recurrent=True), theta)


def test_ablation_set_theta_leaves_W_zero():
    esn = stage2g.make_net(recurrent=False)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    stage2g.set_theta(esn, np.ones(126), recurrent=False)
    assert np.array_equal(esn.W, np.zeros((8, 8)))


def test_block_scales_formula():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = np.concatenate([np.full(88, 2.0), np.full(64, 0.0), np.full(38, 5.0)])
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta, slices)
    assert np.allclose(scales[:88], 0.1 * 2.0)          # RMS 2.0
    assert np.allclose(scales[88:152], 0.1 * 1e-3)      # RMS 0 -> floor
    assert np.allclose(scales[152:], 0.1 * 5.0)         # RMS 5.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k "param or set_theta or block_scales or frozen_W" -v`
Expected: FAIL (module `stage2g` has no such attributes).

- [ ] **Step 3: Write the minimal implementation**

```python
# stage2g.py
"""Stage 2g: a fully-trainable recurrent baseline (route-memory gate, take 5).

Purpose (see docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md).
Stages 2d-2f froze the reservoir (W_in, W) to the 2c winner and trained only the
150-parameter readout; 2f cleared the optimization floor but not the 0.90/0.80 gate.
Stage 2g makes the recurrent weights themselves TRAINABLE: an n=8 echo-state net
whose W_in (8x11) + W (8x8) + W_out (2x19) = 190 params are optimized end-to-end by
block-scaled CEM against the frozen 2f aligned reward (no BPTT -- the stack has no
autograd and the reward is non-differentiable through the simulator). The matched
recurrence-off ablation freezes W=0 with unit leak and trains 126 params.

Honest attribution: n=64 -> n=8 is itself an architectural change, so this is not
literally one variable versus 2f; exact attribution would need a matched n=8
readout-only control (not run). A pass is meaningful under the pass-only asymmetry.
Does not touch the connectome or the frozen Stage 2/2b/2c/2d/2e/2f artifacts.
"""

from __future__ import annotations

import numpy as np

from street import Control, MAX_ACCEL, MAX_BRAKE, run_street_episode
from simulation import MAX_STEERING
from stage2c import EchoStateNetwork, N_FEATURES, RecurrentController, observation_features

N8_CONFIG = {"n_reservoir": 8, "spectral_radius": 0.8, "leak": 0.5, "input_scale": 1.0}
ESN_SEED = 0
RIDGE = 1e-3   # BC init ridge (initialization only; frozen, not tuned)


def make_net(recurrent: bool) -> EchoStateNetwork:
    """The n=8 net (recurrent) or its recurrence-off ablation (W=0, leak=1)."""
    return EchoStateNetwork(
        n_reservoir=N8_CONFIG["n_reservoir"],
        spectral_radius=N8_CONFIG["spectral_radius"], leak=N8_CONFIG["leak"],
        input_scale=N8_CONFIG["input_scale"], seed=ESN_SEED, recurrent=recurrent)


def _dout(esn: EchoStateNetwork) -> int:
    return 1 + N_FEATURES + esn.n            # readout input width (19 for n=8)


def flatten_theta(esn: EchoStateNetwork, recurrent: bool) -> np.ndarray:
    """Flatten the trainable blocks in order [W_in, (W if recurrent), W_out]."""
    parts = [esn.W_in.flatten()]
    if recurrent:
        parts.append(esn.W.flatten())
    parts.append(esn.W_out.flatten())
    return np.concatenate(parts)


def set_theta(esn: EchoStateNetwork, theta: np.ndarray, recurrent: bool) -> None:
    """Inverse of `flatten_theta`. Leaves esn.W at its frozen zeros for the ablation."""
    theta = np.asarray(theta, dtype=float)
    i = 0
    n_in = esn.W_in.size
    esn.W_in = theta[i:i + n_in].reshape(esn.W_in.shape); i += n_in
    if recurrent:
        n_w = esn.n * esn.n
        esn.W = theta[i:i + n_w].reshape(esn.n, esn.n); i += n_w
    d = _dout(esn)
    esn.W_out = theta[i:i + 2 * d].reshape(2, d)


def trainable_slices(esn: EchoStateNetwork, recurrent: bool) -> list[tuple[int, int]]:
    """(start, stop) index ranges of each trainable block in flatten order."""
    slices, i = [], 0
    n_in = esn.W_in.size
    slices.append((i, i + n_in)); i += n_in
    if recurrent:
        n_w = esn.n * esn.n
        slices.append((i, i + n_w)); i += n_w
    d = _dout(esn)
    slices.append((i, i + 2 * d)); i += 2 * d
    return slices


def block_scales(theta: np.ndarray, slices) -> np.ndarray:
    """Per-parameter perturbation scale: 0.1 * max(RMS(block), 1e-3) per block."""
    theta = np.asarray(theta, dtype=float)
    scale = np.empty_like(theta)
    for start, stop in slices:
        rms = float(np.sqrt(np.mean(theta[start:stop] ** 2)))
        scale[start:stop] = 0.1 * max(rms, 1e-3)
    return scale
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k "param or set_theta or block_scales or frozen_W" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add stage2g.py test_stage2g.py
git commit -m "feat(stage2g): n=8 net weight (de)serialization and block scales"
```

---

### Task 2: BC warm start + determinism

**Files:**
- Modify: `stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `evaluate_stage2c.collect_demos`, `evaluate_stage2.load_scenarios`, `street.initial_layouts`, `stage2c.EchoStateNetwork.fit`.
- Produces:
  - `BC_DEMO_SPLIT = "runs/stage2b/sm_train_split.json"`
  - `warm_start_theta(recurrent: bool, layouts=None) -> tuple[EchoStateNetwork, np.ndarray]` — ridge-fit `W_out` on the BC-demo split (`W_in`/`W` at their seeded init), return `(esn, theta0)` with `theta0 = flatten_theta(esn, recurrent)`. Deterministic.

- [ ] **Step 1: Write the failing tests**

```python
def test_warm_start_is_deterministic_bit_identical():
    _, t0a = stage2g.warm_start_theta(recurrent=True)
    _, t0b = stage2g.warm_start_theta(recurrent=True)
    assert np.array_equal(t0a, t0b)              # same seed -> identical theta0


def test_warm_start_sizes_match_arm():
    _, t_rec = stage2g.warm_start_theta(recurrent=True)
    _, t_abl = stage2g.warm_start_theta(recurrent=False)
    assert t_rec.size == 190
    assert t_abl.size == 126
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k warm_start -v`
Expected: FAIL (`warm_start_theta` not defined).

- [ ] **Step 3: Write the minimal implementation**

```python
# add to stage2g.py imports
from street import initial_layouts
from evaluate_stage2 import load_scenarios
import evaluate_stage2c as s2c

BC_DEMO_SPLIT = "runs/stage2b/sm_train_split.json"


def warm_start_theta(recurrent: bool, layouts=None):
    """BC-initialize the net: ridge-fit W_out on the spent teacher-demo split
    (W_in/W at their seeded init), return (esn, theta0). Deterministic: the
    waypoint teacher and ridge fit are deterministic given ESN_SEED. BC is
    initialization only -- the clone is never scored."""
    layouts = layouts or initial_layouts()
    feats, targs = s2c.collect_demos(load_scenarios(BC_DEMO_SPLIT), layouts)
    esn = make_net(recurrent)
    esn.fit(feats, targs, ridge=RIDGE)
    return esn, flatten_theta(esn, recurrent)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k warm_start -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add stage2g.py test_stage2g.py
git commit -m "feat(stage2g): deterministic BC warm start for both arms"
```

---

### Task 3: No-op baseline + go/no-go probe

**Files:**
- Modify: `stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `stage2d.cem_maximize`, `stage2d.CEMConfig`, `stage2f.episode_secondary`, `stage2c.RecurrentController`, `street.{Control, run_street_episode}`.
- Produces:
  - `NoOpController` — `reset()` no-op, `__call__(obs) -> Control(0.0, 0.0)`.
  - `policy_fitness_arrivals(controller, scenarios, layouts) -> tuple[int, float]` — `(arrivals, aligned_fitness_value)` for any controller (the arrival-count-primary fitness computed inline so it works for the no-op controller too).
  - `go_no_go(esn, theta0, scales, scenarios, layouts, cfg, recurrent) -> dict` — replicates CEM iteration-0 sampling (`default_rng(cfg.seed).normal(0, cfg.init_std, (pop, dim))`), evaluates `θ₀` and the population, compares to no-op. Keys: `probe_best_arrivals`, `population_best_fitness`, `noop_fitness`, `beats_noop`, `passed`.

- [ ] **Step 1: Write the failing tests**

```python
from stage2d import CEMConfig, cem_maximize
from street import initial_layouts
from evaluate_stage2b import generate_stage2b_split


def _tiny_scenarios():
    return generate_stage2b_split(
        999, counts={"cross": 0, "regular": 2, "asymmetric": 0})


def test_noop_controller_makes_no_arrivals():
    layouts = initial_layouts()
    arr, _ = stage2g.policy_fitness_arrivals(
        stage2g.NoOpController(), _tiny_scenarios(), layouts)
    assert arr == 0


def test_go_no_go_population_matches_cem_iteration_zero():
    # go/no-go's sampled population is exactly cem_maximize's iteration-0 draw,
    # so the full 30-iter run at the same seed reproduces it bit-for-bit.
    layouts = initial_layouts()
    scenarios = _tiny_scenarios()
    esn, theta0 = stage2g.warm_start_theta(True, layouts)
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta0, slices)
    cfg = CEMConfig(population=4, n_iter=1, init_std=1.0, seed=0)

    probe = stage2g.go_no_go(esn, theta0, scales, scenarios, layouts, cfg, True)

    def fitness_z(z):
        stage2g.set_theta(esn, theta0 + scales * z, True)
        arr, fit = stage2g.policy_fitness_arrivals(
            stage2c_controller := __import__("stage2c").RecurrentController(esn),
            scenarios, layouts)
        return fit

    _, info = cem_maximize(fitness_z, theta0.size, cfg, init_mu=np.zeros(theta0.size))
    assert probe["population_best_fitness"] == info["best"]["fitness"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k "noop or go_no_go" -v`
Expected: FAIL (`NoOpController` / `go_no_go` not defined).

- [ ] **Step 3: Write the minimal implementation**

```python
# add to stage2g.py imports
from stage2f import episode_secondary


class NoOpController:
    """Zero-control baseline: never steers or accelerates."""

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> Control:
        return Control(0.0, 0.0)


def policy_fitness_arrivals(controller, scenarios, layouts) -> tuple[int, float]:
    """Arrival-count-primary aligned fitness AND the arrival count for any
    controller. Mirrors stage2f.aligned_fitness (arrivals + mean(secondary)/4)
    but also returns arrivals, and accepts a bare controller (e.g. no-op)."""
    arrivals, secondary_total = 0, 0.0
    for scenario in scenarios:
        controller.reset()
        result = run_street_episode(scenario, layouts[scenario.layout],
                                    controller, record=True)
        if result.outcome == "arrival":
            arrivals += 1
        secondary_total += episode_secondary(scenario, result)
    fitness = arrivals + (secondary_total / len(scenarios)) / 4.0
    return arrivals, fitness


def go_no_go(esn, theta0, scales, scenarios, layouts, cfg, recurrent) -> dict:
    """One-iteration CEM probe: sample the iteration-0 population exactly as
    cem_maximize does (same seed/std), evaluate theta0 and every candidate, and
    require >=1 arrival AND a fitness above the no-op baseline. The full run at
    the same seed reproduces this population as its iteration 0."""
    theta0 = np.asarray(theta0, dtype=float)
    dim = theta0.size
    rng = np.random.default_rng(cfg.seed)                       # matches cem line
    sigma = np.full(dim, cfg.init_std)
    samples = rng.normal(np.zeros(dim), sigma, size=(cfg.population, dim))

    best_arr, best_fit, pop_best_fit = 0, -np.inf, -np.inf
    controller = RecurrentController(esn)
    for k, z in enumerate(np.vstack([np.zeros(dim), samples])):
        set_theta(esn, theta0 + scales * z, recurrent)
        arr, fit = policy_fitness_arrivals(controller, scenarios, layouts)
        best_arr = max(best_arr, arr)
        best_fit = max(best_fit, fit)
        if k > 0:                                               # population only
            pop_best_fit = max(pop_best_fit, fit)
    noop_arr, noop_fit = policy_fitness_arrivals(NoOpController(), scenarios, layouts)
    return {
        "probe_best_arrivals": int(best_arr),
        "population_best_fitness": float(pop_best_fit),
        "noop_fitness": float(noop_fit),
        "beats_noop": bool(best_fit > noop_fit),
        "passed": bool(best_arr >= 1 and best_fit > noop_fit),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k "noop or go_no_go" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add stage2g.py test_stage2g.py
git commit -m "feat(stage2g): no-op baseline and CEM iteration-0 go/no-go probe"
```

---

### Task 4: Block-scaled trainer with warm-start guard

**Files:**
- Modify: `stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `stage2d.cem_maximize` (unmodified), `stage2e.evaluate_policy`, `stage2f.aligned_fitness`.
- Produces:
  - `train_by_reward_blockscaled(esn, theta0, scales, scenarios, layouts, cfg, fitness_fn, recurrent) -> tuple[EchoStateNetwork, dict]` — z-space CEM (`θ = θ₀ + scales ⊙ z`, `init_mu=0`), best-ever tracking, warm-start guard (never ship worse than `θ₀`). `info` adds: `improved_over_warmstart`, `warmstart_fitness`, `warmstart_outcomes`, `best_fitness`, `best_theta`, `best_outcomes`, `block_scales`.

- [ ] **Step 1: Write the failing tests**

```python
import stage2d


def test_cem_maximize_is_untouched_by_stage2g():
    # Stage 2g must reparameterize AROUND cem_maximize, never edit it.
    import inspect
    src = inspect.getsource(stage2d.cem_maximize)
    assert "sigma = np.full(dim, cfg.init_std)" in src   # signature line intact


def test_trainer_guard_keeps_warm_start_when_nothing_beats_it():
    layouts = initial_layouts()
    scenarios = _tiny_scenarios()
    esn, theta0 = stage2g.warm_start_theta(True, layouts)
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta0, slices)
    from stage2f import aligned_fitness

    # zero scales => every candidate == theta0 => nothing can beat the warm start
    esn, info = stage2g.train_by_reward_blockscaled(
        esn, theta0, np.zeros_like(scales), scenarios, layouts,
        CEMConfig(population=4, n_iter=2, init_std=1.0, seed=0),
        aligned_fitness, recurrent=True)
    assert info["improved_over_warmstart"] is False
    assert np.allclose(np.asarray(info["best_theta"]), theta0)
    assert info["best_fitness"] == info["warmstart_fitness"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k "untouched or guard" -v`
Expected: FAIL (`train_by_reward_blockscaled` not defined).

- [ ] **Step 3: Write the minimal implementation**

```python
# add to stage2g.py imports
from stage2d import cem_maximize
from stage2e import evaluate_policy


def train_by_reward_blockscaled(esn, theta0, scales, scenarios, layouts, cfg,
                                fitness_fn, recurrent):
    """Fine-tune all trainable weights by CEM in normalized z-space
    (theta = theta0 + scales*z, init_mu=0). Best-ever candidate with a warm-start
    guard: never ship a policy worse than theta0. Reuses cem_maximize unmodified."""
    theta0 = np.asarray(theta0, dtype=float)
    scales = np.asarray(scales, dtype=float)
    dim = theta0.size

    def fitness_z(z: np.ndarray) -> float:
        set_theta(esn, theta0 + scales * z, recurrent)
        return fitness_fn(esn, scenarios, layouts)

    f0 = fitness_z(np.zeros(dim))                     # warm-start fitness
    warmstart_outcomes = evaluate_policy(esn, scenarios, layouts, fitness_fn)
    _, info = cem_maximize(fitness_z, dim, cfg, init_mu=np.zeros(dim))
    if info["best"]["fitness"] >= f0:
        best_z = np.asarray(info["best"]["theta"], dtype=float)
        info["improved_over_warmstart"] = True
    else:
        best_z = np.zeros(dim)
        info["improved_over_warmstart"] = False
    best_theta = theta0 + scales * best_z
    set_theta(esn, best_theta, recurrent)
    info["warmstart_fitness"] = float(f0)
    info["warmstart_outcomes"] = warmstart_outcomes
    info["best_fitness"] = float(max(info["best"]["fitness"], f0))
    info["best_theta"] = best_theta.tolist()
    info["best_outcomes"] = evaluate_policy(esn, scenarios, layouts, fitness_fn)
    info["block_scales"] = [float(scales[s]) for s, _ in trainable_slices(esn, recurrent)]
    return esn, info
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k "untouched or guard" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add stage2g.py test_stage2g.py
git commit -m "feat(stage2g): block-scaled CEM trainer with warm-start guard"
```

---

### Task 5: Fresh train/dev/gate splits — build, freeze, hash, assert disjoint

**Files:**
- Create: `evaluate_stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `evaluate_stage2f.{EXCLUDE_PATHS, generate_expanded_cross, _cardinal_strata, _load}`, `evaluate_stage2.{_scenario_key, load_scenarios, save_scenarios, sha256}`, `street.initial_layouts`.
- Produces:
  - **(Amended 2026-09-21 — Variant B.)** `EXCLUDE_PATHS: list[str]` (the enforced-disjoint set) = every prior **scored gate** `["runs/stage2b/gate_split.json","runs/stage2d/gate_split.json","runs/stage2e/gate_split.json","runs/stage2f/gate_split.json"]` + 2f fitness `["runs/stage2f/fitness_split.json"]` + `[stage2g.BC_DEMO_SPLIT]`. NOT 2f's full `EXCLUDE_PATHS` (that pulled in prior *training* splits and left only 17 regular scenarios — infeasible). Older training splits may recur (historical exposure, not leakage).
  - `ALL_PRIOR_SPLITS: list[str]` = every prior split (2f's `EXCLUDE_PATHS` + 2f fitness + 2f gate, deduped) — used for provenance overlap accounting, a superset of `EXCLUDE_PATHS`.
  - Seeds `TRAIN_SEED=70, DEV_SEED=71, GATE_SEED=72`; `TRAIN_COUNTS={"cross":6,"regular":30,"asymmetric":30}`, `DEV_COUNTS={"cross":6,"regular":20,"asymmetric":20}`, `GATE_COUNTS={"cross":12,"regular":44,"asymmetric":44}`.
  - `build_split(seed, counts, extra_exclude) -> list[StreetScenario]` — guards cross gen (`counts["cross"]` of 0 → no cross).
  - `build_all_splits() -> tuple[list, list, list]` — train, then dev (excl train), then gate (excl train+dev); each asserted disjoint.
  - `overlap_provenance(train, dev, gate) -> dict[str, dict[str, int]]` — overlap count of each new split against **every** prior split; asserts enforced-set overlaps are zero, records re-admitted overlaps.
  - `freeze_and_hash(train, dev, gate) -> dict[str, str]` — asserts overlap provenance, writes the three JSONs, returns `{path: sha256}` for them + every prior split.

- [ ] **Step 1: Write the failing tests**

```python
import evaluate_stage2g as e2g
from evaluate_stage2 import _scenario_key


def test_split_sizes_and_strata():
    train, dev, gate = e2g.build_all_splits()
    assert len(train) == 66 and len(dev) == 46 and len(gate) == 100
    for split, n_cross in ((train, 6), (dev, 6), (gate, 12)):
        assert sum(s.layout == "cross" for s in split) == n_cross


def test_splits_pairwise_disjoint_and_clear_of_priors():
    train, dev, gate = e2g.build_all_splits()
    keys = [{_scenario_key(s) for s in sp} for sp in (train, dev, gate)]
    assert keys[0].isdisjoint(keys[1])
    assert keys[0].isdisjoint(keys[2])
    assert keys[1].isdisjoint(keys[2])
    excluded = {_scenario_key(s) for s in e2g._load(e2g.EXCLUDE_PATHS)}
    for k in keys:
        assert k.isdisjoint(excluded)     # incl. sm_train_split.json (BC demo)


def test_bc_demo_split_is_in_exclusion_set():
    import stage2g
    assert stage2g.BC_DEMO_SPLIT in e2g.EXCLUDE_PATHS


def test_every_prior_scored_gate_is_enforced_excluded():
    for g in ("runs/stage2b/gate_split.json", "runs/stage2d/gate_split.json",
              "runs/stage2e/gate_split.json", "runs/stage2f/gate_split.json"):
        assert g in e2g.EXCLUDE_PATHS


def test_overlap_provenance_enforced_zero_covers_all_priors():
    train, dev, gate = e2g.build_all_splits()
    report = e2g.overlap_provenance(train, dev, gate)
    assert set(e2g.EXCLUDE_PATHS).issubset(set(e2g.ALL_PRIOR_SPLITS))
    for name in ("train", "dev", "gate"):
        assert set(report[name]) == set(e2g.ALL_PRIOR_SPLITS)   # every prior split accounted
        for path in e2g.EXCLUDE_PATHS:
            assert report[name][path] == 0                      # enforced overlaps are zero
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k "split or bc_demo or prior or provenance" -v`
Expected: FAIL (`evaluate_stage2g` does not exist).

- [ ] **Step 3: Write the minimal implementation**

```python
# evaluate_stage2g.py
"""Stage 2g evaluation: fully-trainable recurrent baseline, scored once on a newly
frozen 0.90/0.80 gate. Three newly frozen, Stage-2g-policy-unseen splits (train 66 /
dev 46 / gate 100) share the scarce diagonal witnessed-cross pool; they are frozen
and sha256-hashed BEFORE any training. The gate is generated last and never inspected
until the single scoring event. Disjointness follows the Variant B enforced set
(prior scored gates + 2f fitness + BC demo); older prior TRAINING splits may recur as
historical exposure -- overlap counts vs every prior split are recorded.

Escalation is a hard halt: if the trained n=8 recurrent model fails go/no-go or
fails to improve DEV arrivals over its warm start, the run stops before the gate --
n=16 + CMA-ES is NOT implemented here and requires a written spec amendment first.
Does not touch the connectome or any frozen Stage 2..2f artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from street import initial_layouts
from evaluate_stage2 import _scenario_key, load_scenarios, save_scenarios, sha256
from evaluate_stage2f import (
    EXCLUDE_PATHS as _PRIOR_EXCLUDE, _cardinal_strata, _load, generate_expanded_cross,
)
import stage2g

# Variant B enforced-disjoint set (2026-09-21 amendment): every prior SCORED GATE +
# 2f fitness + BC demo. Full exclusion of prior TRAINING splits was infeasible (only
# 17 regular scenarios left). Older training splits may recur -> historical design
# exposure, NOT Stage 2g policy-training leakage.
PRIOR_GATES = [
    "runs/stage2b/gate_split.json",
    "runs/stage2d/gate_split.json",
    "runs/stage2e/gate_split.json",
    "runs/stage2f/gate_split.json",
]
SPENT_2F_FITNESS = ["runs/stage2f/fitness_split.json"]
EXCLUDE_PATHS = PRIOR_GATES + SPENT_2F_FITNESS + [stage2g.BC_DEMO_SPLIT]

# Every prior split (enforced or re-admitted) -- for provenance overlap accounting.
ALL_PRIOR_SPLITS = list(dict.fromkeys(
    list(_PRIOR_EXCLUDE) + SPENT_2F_FITNESS + ["runs/stage2f/gate_split.json"]))

TRAIN_SEED, DEV_SEED, GATE_SEED = 70, 71, 72
TRAIN_COUNTS = {"cross": 6, "regular": 30, "asymmetric": 30}    # 66
DEV_COUNTS = {"cross": 6, "regular": 20, "asymmetric": 20}      # 46
GATE_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}    # 100


def _assert_disjoint(split, exclude, label: str) -> None:
    keys = {_scenario_key(s) for s in split}
    assert len(keys) == len(split), f"duplicate identity within {label}"
    assert keys.isdisjoint({_scenario_key(s) for s in exclude}), \
        f"{label} overlaps an excluded/earlier split"


def build_split(seed: int, counts: dict, extra_exclude) -> list:
    """One fresh stratified split (diagonal witnessed cross + cardinal reg/asym),
    disjoint from EXCLUDE_PATHS plus `extra_exclude` (earlier fresh splits)."""
    exclude = _load(EXCLUDE_PATHS) + list(extra_exclude)
    cross = generate_expanded_cross(seed, exclude, counts["cross"]) if counts["cross"] else []
    rest = _cardinal_strata(seed, exclude, counts)
    split = cross + rest
    _assert_disjoint(split, exclude, f"split(seed={seed})")
    return split


def build_all_splits():
    """train (70) -> dev (71, excl train) -> gate (72, excl train+dev)."""
    train = build_split(TRAIN_SEED, TRAIN_COUNTS, [])
    dev = build_split(DEV_SEED, DEV_COUNTS, train)
    gate = build_split(GATE_SEED, GATE_COUNTS, train + dev)
    return train, dev, gate


def overlap_provenance(train, dev, gate) -> dict:
    """Overlap count of each new split against EVERY prior split. Enforced-set
    overlaps are asserted zero (fail loudly); re-admitted training-split overlaps
    are recorded as historical exposure (not Stage 2g policy-training leakage)."""
    enforced = set(EXCLUDE_PATHS)
    report = {}
    for name, split in (("train", train), ("dev", dev), ("gate", gate)):
        keys = {_scenario_key(s) for s in split}
        per_prior = {}
        for path in ALL_PRIOR_SPLITS:
            n = len(keys & {_scenario_key(s) for s in load_scenarios(path)})
            per_prior[path] = n
            if path in enforced:
                assert n == 0, f"{name} overlaps enforced-excluded {path} ({n})"
        report[name] = per_prior
    return report


def freeze_and_hash(train, dev, gate) -> dict:
    """Assert overlap provenance, write the three splits, and return sha256 for them
    + every prior split (enforced and re-admitted)."""
    overlap_provenance(train, dev, gate)          # fail loudly BEFORE freezing
    paths = {"runs/stage2g/train_split.json": train,
             "runs/stage2g/dev_split.json": dev,
             "runs/stage2g/gate_split.json": gate}
    for path, split in paths.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists():
            save_scenarios(path, split)
    provenance = {p: sha256(p) for p in ALL_PRIOR_SPLITS}
    provenance.update({p: sha256(p) for p in paths})
    return provenance
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k "split or bc_demo or prior or provenance" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluate_stage2g.py test_stage2g.py
git commit -m "feat(stage2g): fresh train/dev/gate splits, frozen and hashed"
```

---

### Task 6: Per-arm run + dev-arrivals escalation halt + selection score

**Files:**
- Modify: `evaluate_stage2g.py`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `stage2g.{warm_start_theta, trainable_slices, block_scales, go_no_go, train_by_reward_blockscaled}`, `stage2e.evaluate_policy`, `stage2f.aligned_fitness`, `stage2d.CEMConfig`.
- Produces:
  - `CEM_BUDGET = CEMConfig(population=64, n_iter=30, elite_frac=0.20, init_std=1.0, std_floor=0.001, seed=0)`.
  - `run_arm(recurrent, train, dev, layouts, cfg) -> dict` — warm start → scales → go/no-go → (if passed) full run; returns `{"recurrent": bool, "gng": dict, "warmstart_train_arrivals": int, "warmstart_dev_arrivals": int, "trained_dev_arrivals": int, "escalate": bool, "esn": EchoStateNetwork, "best_theta": list, "info": dict}`. `escalate=True` if go/no-go failed OR trained dev arrivals ≤ warm dev arrivals.
  - `dev_gate_score(breakdown) -> float` = `min(overall/0.90, cross/0.80, regular/0.80, asym/0.80)` (selection metric; drives a choice only once a future amendment adds n=16).
  - `EscalationHalt(Exception)` — raised to stop before the gate with the amendment instruction.

- [ ] **Step 1: Write the failing tests**

```python
def test_dev_gate_score_is_min_of_normalized_rates():
    bd = {"overall": {"arrival_rate": 0.90},
          "by_layout": {"regular": {"arrival_rate": 0.80},
                        "asymmetric": {"arrival_rate": 0.80},
                        "cross": {"arrival_rate": 0.40}}}  # cross half of bar
    assert abs(e2g.dev_gate_score(bd) - 0.5) < 1e-9        # 0.40/0.80 == 0.5


def test_run_arm_escalates_when_dev_arrivals_do_not_improve(monkeypatch):
    # Force the trainer to return the warm start unchanged (zero scales path):
    import stage2g
    real = stage2g.block_scales
    monkeypatch.setattr(stage2g, "block_scales",
                        lambda theta, slices: real(theta, slices) * 0.0)
    # Force go/no-go to pass so we reach the dev-improvement check:
    monkeypatch.setattr(stage2g, "go_no_go",
                        lambda *a, **k: {"passed": True, "probe_best_arrivals": 1,
                                         "population_best_fitness": 0.0,
                                         "noop_fitness": -1.0, "beats_noop": True})
    layouts = initial_layouts()
    train = e2g.build_split(70, {"cross": 0, "regular": 2, "asymmetric": 0}, [])
    dev = e2g.build_split(71, {"cross": 0, "regular": 2, "asymmetric": 0}, train)
    arm = e2g.run_arm(True, train, dev, layouts,
                      e2g.CEM_BUDGET.__class__(population=2, n_iter=1,
                                               init_std=1.0, seed=0))
    assert arm["escalate"] is True                         # no dev improvement
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k "dev_gate_score or escalates" -v`
Expected: FAIL (`dev_gate_score` / `run_arm` not defined).

- [ ] **Step 3: Write the minimal implementation**

```python
# add to evaluate_stage2g.py imports
from stage2d import CEMConfig
from stage2e import evaluate_policy
from stage2f import aligned_fitness

CEM_BUDGET = CEMConfig(population=64, n_iter=30, elite_frac=0.20,
                       init_std=1.0, std_floor=0.001, seed=0)


class EscalationHalt(Exception):
    """Raised to stop before the gate: write the n=16 + CMA-ES spec amendment
    (implementation/budget/init/scaling) before any further optimization."""


def dev_gate_score(breakdown) -> float:
    """min(overall/0.90, cross/0.80, regular/0.80, asym/0.80) on a breakdown.
    Selection metric; with only n=8 implemented the choice is trivially n=8."""
    o = breakdown["overall"]["arrival_rate"] / 0.90
    per = breakdown["by_layout"]
    layers = [per[name]["arrival_rate"] / 0.80
              for name in ("regular", "asymmetric", "cross")]
    return float(min(o, *layers))


def _arrivals(esn, theta, scenarios, layouts, recurrent) -> int:
    stage2g.set_theta(esn, theta, recurrent)
    return evaluate_policy(esn, scenarios, layouts)["arrivals"]


def run_arm(recurrent: bool, train, dev, layouts, cfg) -> dict:
    """Warm start -> block scales -> go/no-go -> full run. Sets escalate=True on a
    go/no-go failure or when trained DEV arrivals do not exceed the warm start's."""
    esn, theta0 = stage2g.warm_start_theta(recurrent, layouts)
    slices = stage2g.trainable_slices(esn, recurrent)
    scales = stage2g.block_scales(theta0, slices)
    warm_train_arr = evaluate_policy(esn, train, layouts)["arrivals"]  # esn at theta0
    warm_dev_arr = _arrivals(esn, theta0, dev, layouts, recurrent)

    gng = stage2g.go_no_go(esn, theta0, scales, train, layouts, cfg, recurrent)
    if not gng["passed"]:
        return {"recurrent": recurrent, "gng": gng, "escalate": True,
                "warmstart_train_arrivals": warm_train_arr,
                "warmstart_dev_arrivals": warm_dev_arr, "trained_dev_arrivals": None,
                "esn": esn, "best_theta": theta0.tolist(), "info": None}

    esn, info = stage2g.train_by_reward_blockscaled(
        esn, theta0, scales, train, layouts, cfg, aligned_fitness, recurrent)
    trained_dev_arr = evaluate_policy(esn, dev, layouts)["arrivals"]   # esn at best
    return {"recurrent": recurrent, "gng": gng,
            "warmstart_train_arrivals": warm_train_arr,
            "warmstart_dev_arrivals": warm_dev_arr,
            "trained_dev_arrivals": trained_dev_arr,
            "escalate": trained_dev_arr <= warm_dev_arr,
            "esn": esn, "best_theta": info["best_theta"], "info": info}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k "dev_gate_score or escalates" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluate_stage2g.py test_stage2g.py
git commit -m "feat(stage2g): per-arm run with dev-arrivals escalation halt"
```

---

### Task 7: One-shot gate, interpretation, persistence, `main`, README

**Files:**
- Modify: `evaluate_stage2g.py`
- Create: `runs/stage2g/README.md`
- Test: `test_stage2g.py`

**Interfaces:**
- Consumes: `evaluate_stage2b.{_breakdown, _gate, _run_per_scenario}`, `stage2c.RecurrentController`, `evaluate_stage2.{GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE}`, `stage2g.N8_CONFIG`.
- Produces:
  - `gate_breakdown(esn, scenarios, layouts) -> dict` and `score_gate(esn, scenarios, layouts) -> tuple[bool, float, dict, dict]` (`passes, overall, per_layout, breakdown`).
  - `interpret(recurrent_res, ablation_res) -> str` — bounded language; the both-pass branch says "recurrence is not necessary within this gate and policy family".
  - `main(argv=None) -> None` — freeze+hash splits, run recurrent arm, escalate-halt if needed, run ablation arm, score the sealed gate once for both, persist `runs/stage2g/{results,gate_results}.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_interpret_both_pass_is_bounded_to_policy_family():
    passing = {"passes_gate": True, "overall_arrival_rate": 0.95,
               "by_layout_arrival_rate": {"regular": 0.9, "asymmetric": 0.9,
                                          "cross": 0.9}}
    rec = dict(passing)
    abl = dict(passing)
    text = e2g.interpret(rec, abl)
    assert "within this gate and policy family" in text
    assert "not required" not in text or "family" in text


def test_interpret_recurrent_only_pass_beats_ablation():
    rec = {"passes_gate": True, "overall_arrival_rate": 0.95,
           "by_layout_arrival_rate": {"regular": 0.9, "asymmetric": 0.9, "cross": 0.9}}
    abl = {"passes_gate": False, "overall_arrival_rate": 0.60,
           "by_layout_arrival_rate": {"regular": 0.7, "asymmetric": 0.6, "cross": 0.5}}
    text = e2g.interpret(rec, abl)
    assert "temporal state" in text or "learned temporal" in text
    assert "attribution" in text.lower()      # honest-attribution caveat present
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest test_stage2g.py -k interpret -v`
Expected: FAIL (`interpret` not defined).

- [ ] **Step 3: Write the minimal implementation**

```python
# add to evaluate_stage2g.py imports
from evaluate_stage2 import GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE
from evaluate_stage2b import _breakdown, _gate, _run_per_scenario
from stage2c import RecurrentController

REF = {"waypoint": 1.00, "stage2b_sm": 0.62, "stage2c": 0.57,
       "stage2d": 0.00, "stage2e": 0.516, "stage2f_recurrent": 0.660,
       "stage2f_ablation": 0.540}


def gate_breakdown(esn, scenarios, layouts) -> dict:
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def score_gate(esn, scenarios, layouts):
    bd = gate_breakdown(esn, scenarios, layouts)
    passes, overall, per_layout = _gate(bd, layouts)
    return passes, overall, per_layout, bd


_ATTRIBUTION = (" Attribution is bounded: n=64->n=8 is itself an architectural "
                "change, so this is not one-variable-vs-2f, and exact attribution "
                "would need a matched n=8 readout-only control (not run here).")


def interpret(recurrent, ablation) -> str:
    r = recurrent["overall_arrival_rate"]
    a = ablation["overall_arrival_rate"]
    lead = (f"Recurrent {r:.3f} vs recurrence-off ablation {a:.3f} (refs: waypoint "
            f"{REF['waypoint']:.2f}, 2b SM {REF['stage2b_sm']:.2f}, 2c "
            f"{REF['stage2c']:.2f}, 2d {REF['stage2d']:.2f}, 2e "
            f"{REF['stage2e']:.3f}, 2f rec {REF['stage2f_recurrent']:.3f}).")
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " A trainable recurrent policy clears a fresh gate and learned "
                "temporal state carries the advantage over the recurrence-off ablation."
                + _ATTRIBUTION + " Motivates the biologically grounded "
                "memory/action-selection work (dopamine as modulation, never goal "
                "bearing).")
    if recurrent["passes_gate"] and ablation["passes_gate"]:
        return (lead + " Both arms clear the gate, so recurrence is not necessary "
                "within this gate and policy family (n=8, this observation interface, "
                "this reward) -- not a general claim that memory is never required."
                + _ATTRIBUTION)
    if recurrent["passes_gate"]:
        return (lead + " The recurrent policy clears the gate but does not clearly beat "
                "its ablation, so training exploited the reactive policy state; the "
                "interface is sufficient and memory is not required within this policy "
                "family." + _ATTRIBUTION)
    return (lead + " Neither arm cleared the fresh gate. Under the pass-only asymmetry "
            "this stays bounded and confounded between optimization budget, capacity "
            "(n=8) and coverage; it does not implicate the observation interface. "
            "Report the recurrent-ablation gap and collision/arrival breakdown before "
            "any interface claim.")


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("train_history", "best_theta")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2g trainable recurrent baseline.")
    parser.add_argument("--out", default="runs/stage2g/results.json")
    parser.add_argument("--gate-out", default="runs/stage2g/gate_results.json")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    train, dev, gate = build_all_splits()
    provenance = freeze_and_hash(train, dev, gate)     # frozen + hashed BEFORE training
    print(f"Stage 2g: train {len(train)} / dev {len(dev)} / gate {len(gate)} "
          f"(cross 6/6/12). Splits frozen and hashed.")

    t0 = time.perf_counter()
    rec_run = run_arm(True, train, dev, layouts, CEM_BUDGET)
    print(f"Recurrent: warm dev arrivals {rec_run['warmstart_dev_arrivals']} -> "
          f"trained {rec_run['trained_dev_arrivals']}; go/no-go {rec_run['gng']['passed']}.")
    if rec_run["escalate"]:
        raise EscalationHalt(
            "n=8 recurrent did not improve dev arrivals (or failed go/no-go). "
            "STOP before the gate: write the n=16 + CMA-ES spec amendment "
            "(implementation, budget, initialization, block scaling) on the SAME "
            "train/dev splits before any further optimization. Gate is untouched.")

    abl_run = run_arm(False, train, dev, layouts, CEM_BUDGET)
    if abl_run["escalate"]:
        raise EscalationHalt(
            "Ablation failed its go/no-go / dev-improvement precondition; the control "
            "arm cannot be fairly scored. STOP before the gate and diagnose.")
    print(f"Done training in {(time.perf_counter()-t0)/60:.1f} min. Scoring gate once.")

    rec_pass, rec_ov, rec_pl, rec_bd = score_gate(rec_run["esn"], gate, layouts)
    abl_pass, abl_ov, abl_pl, abl_bd = score_gate(abl_run["esn"], gate, layouts)
    recurrent = {"passes_gate": rec_pass, "overall_arrival_rate": rec_ov,
                 "by_layout_arrival_rate": rec_pl, "breakdown": rec_bd,
                 "dev_gate_score": dev_gate_score(gate_breakdown(rec_run["esn"], dev, layouts)),
                 **_strip(rec_run["info"]), "best_theta": rec_run["best_theta"],
                 "train_history": rec_run["info"]["history"],
                 "warmstart_train_arrivals": rec_run["warmstart_train_arrivals"]}
    ablation = {"passes_gate": abl_pass, "overall_arrival_rate": abl_ov,
                "by_layout_arrival_rate": abl_pl, "breakdown": abl_bd,
                **_strip(abl_run["info"]), "best_theta": abl_run["best_theta"],
                "train_history": abl_run["info"]["history"],
                "warmstart_train_arrivals": abl_run["warmstart_train_arrivals"]}
    interpretation = interpret(recurrent, ablation)

    gate_payload = {
        "train_split": "runs/stage2g/train_split.json",
        "dev_split": "runs/stage2g/dev_split.json",
        "gate_split": "runs/stage2g/gate_split.json",
        "provenance_sha256": provenance,
        "gate": {"overall_rate": GATE_OVERALL_RATE, "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "model": stage2g.N8_CONFIG, "esn_seed": stage2g.ESN_SEED,
        "cem_budget": vars(CEM_BUDGET),
        "trainable_params": {"recurrent": 190, "ablation": 126},
        "block_scales": {"recurrent": recurrent.get("block_scales"),
                         "ablation": ablation.get("block_scales")},
        "cross_note": ("diagonal-heading waypoint-witnessed cross (same generator as "
                       "2f, comparable to 2f only); 0.80 cross gate needs 10/12."),
        "recurrent": {k: v for k, v in recurrent.items()
                      if k not in ("train_history", "best_theta")},
        "memoryless_ablation": {k: v for k, v in ablation.items()
                                if k not in ("train_history", "best_theta")},
        "references": REF, "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["recurrent"] = recurrent
    results_payload["memoryless_ablation"] = ablation
    Path(args.out).write_text(json.dumps(results_payload, indent=2) + "\n")

    for name, res in (("Recurrent", recurrent), ("Ablation", ablation)):
        print(f"{name} gate: overall={res['overall_arrival_rate']:.3f} "
              f"{res['by_layout_arrival_rate']} "
              f"{'PASS' if res['passes_gate'] else 'FAIL'}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest test_stage2g.py -k interpret -v`
Expected: PASS (2 tests). Then run the whole file: `pytest test_stage2g.py -v` — all green.

- [ ] **Step 5: Write the README**

```markdown
<!-- runs/stage2g/README.md -->
# Stage 2g — fully-trainable recurrent baseline

Trains an n=8 echo-state net end-to-end (W_in + W + W_out = 190 params) by
block-scaled CEM against the frozen Stage 2f aligned reward, with a matched
recurrence-off ablation (126 params, W=0 / leak=1). Fresh train(70)/dev(71)/
gate(72) splits are frozen and sha256-hashed before training; the gate is scored
once. Escalation (dev arrivals not improved, or go/no-go failed) is a HARD HALT:
n=16 + CMA-ES is not implemented and requires a written spec amendment first.

Spec: `docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md`.
Run: `python evaluate_stage2g.py`. Results (filled in after the one-shot run):
recurrent __, ablation __. Interpretation is bounded (pass-only asymmetry; both-pass
scoped to "recurrence not necessary within this gate and policy family"; no matched
n=8 readout-only control, so exact attribution is not claimed).
```

- [ ] **Step 6: Commit**

```bash
git add evaluate_stage2g.py test_stage2g.py runs/stage2g/README.md
git commit -m "feat(stage2g): one-shot gate, bounded interpretation, run entrypoint"
```

---

## Self-Review

**Spec coverage:**
- Model (190 / 126, W≡0+leak=1 ablation) → Task 1. ✅
- BC warm start + determinism → Task 2. ✅
- Block scaling `0.1·max(RMS,1e-3)`, recorded → Task 1 (formula) + Task 4 (`info["block_scales"]`) + Task 7 (payload). ✅
- Go/no-go = one-iteration probe reproducing CEM iteration 0 → Task 3. ✅
- Block-scaled trainer, warm-start guard, `cem_maximize` untouched → Task 4. ✅
- Fresh train/dev/gate, freeze+hash before training, full exclusion incl. BC-demo split → Task 5. ✅
- Dev-arrivals escalation as a hard halt; CMA-ES not implemented → Task 6 (`run_arm`) + Task 7 (`EscalationHalt` in `main`). ✅
- `dev_gate_score` selection metric → Task 6. ✅
- One-shot gate, bounded interpretation (both-pass scoped, attribution caveat) → Task 7. ✅
- Determinism assertion + logged warm-start train arrivals → Task 2 (test) + Task 6/7 (`warmstart_train_arrivals`). ✅

**Placeholder scan:** No TBD/TODO; every code step is complete. README result blanks are filled by the actual run, not the plan. ✅

**Type consistency:** `flatten_theta`/`set_theta`/`trainable_slices` share the `(esn, recurrent)` signature and flatten order across tasks; `block_scales(theta, slices)` matches its callers in Tasks 4 and 6; `run_arm` returns the keys consumed by `main`; `score_gate` returns `(passes, overall, per_layout, breakdown)` as unpacked in `main`. ✅

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-21-stage2g-trainable-recurrent.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
