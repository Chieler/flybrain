"""Stage 2d evaluation: train the same-observation readout by REWARD (CEM) on a
fixed 2c-winner reservoir, and score it once on a fresh, disjoint gate.

Provenance discipline (see the spec). The reservoir config is frozen to the 2c
winner (one-variable design), so there is NO config/hyperparameter selection and
no train/val split. Two fresh splits are built with `generate_stage2b_split`
(outward-facing road-end starts excluded), intended disjoint by exact scenario
identity from every prior Stage 2/2b split and from each other. NOTE (documented
defect): the spent run's exclusion list omitted `runs/stage2/training.json` and
`runs/stage2/heldout.json`, so the frozen gate overlaps them by 8 scenarios and
the frozen train split by 6; see `_PRIOR_SPLIT_PATHS_AS_RUN`. The result was 0%,
so the leak cannot fake a pass and the negative stands. The readout is trained by
CEM on 40 fixed stratified fitness scenarios with a frozen budget, identically
for the recurrent model and its matched memoryless ablation, and the frozen gate
is evaluated exactly once for each.

Interpretation (bounded). Only a PASS is conclusive. If recurrent clears the gate
and beats its ablation, temporal memory over the observation interface -- trained
by reward -- closes the gap Stage 2c's imitation could not. If both fail, the
result stays confounded between optimization budget, capacity, and observation
limits; it does not implicate the interface on its own.

Not neural in the connectome sense; does not touch the connectome or the frozen
Stage 2 / 2b / 2c artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from street import initial_layouts, run_street_episode
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, load_scenarios, save_scenarios, sha256,
)
from evaluate_stage2b import (
    _breakdown, _gate, _run_per_scenario, _scenario_key, generate_stage2b_split,
)
from stage2c import RecurrentController
from stage2d import (
    CEMConfig, ESN_SEED, RESERVOIR_CONFIG, episode_reward, make_reservoir,
    train_readout_by_reward,
)

# Every prior Stage 2 / 2b scenario-list split -- excluded wholesale from any
# freshly generated Stage 2d/2e split. This is the CORRECTED, complete list.
PRIOR_SPLIT_PATHS = [
    "runs/stage2/dev_split.json",
    "runs/stage2/training.json",   # original Stage 2 training pool
    "runs/stage2/heldout.json",    # original Stage 2 held-out pool
    "runs/stage2b/gate_split.json",
    "runs/stage2b/sm_train_split.json",
]

# DEFECT (documented, not silently repaired): the spent, one-shot Stage 2d run
# excluded only this incomplete subset, so `training.json` and `heldout.json`
# leaked in. The frozen gate overlaps them by 8 scenarios (2 training + 6
# heldout) and the frozen train split by 6 (1 training + 5 heldout); gate and
# train remain mutually disjoint. Because the Stage 2d result was a 0% FAILURE,
# the leak cannot manufacture a false pass -- overlap with prior pools can only
# make arriving *easier*, and the run still arrived on nothing -- so the negative
# verdict stands. The frozen splits are retained AS RUN (never regenerated); the
# corrected list above is what Stage 2e must exclude (plus the spent 2d splits).
_PRIOR_SPLIT_PATHS_AS_RUN = [
    "runs/stage2/dev_split.json",
    "runs/stage2b/gate_split.json",
    "runs/stage2b/sm_train_split.json",
]

# Fresh seeds (distinct from stage2b's GATE_SPLIT_SEED=21 / SM_TRAIN_SEED=22).
STAGE2D_GATE_SEED = 41
STAGE2D_TRAIN_SEED = 42

# Predeclared counts, as the spent run declared them under the (incomplete)
# as-run exclusion; 12+8=20 `cross` used. Under the corrected PRIOR_SPLIT_PATHS
# fewer eligible `cross` remain -- Stage 2e re-derives its own budget (7/44/44).
GATE_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}
TRAIN_COUNTS = {"cross": 8, "regular": 44, "asymmetric": 44}

# 40 fixed, stratified fitness scenarios drawn from the Stage 2d train split.
FITNESS_STRATA = {"cross": 4, "regular": 18, "asymmetric": 18}

# Frozen CEM budget (identical for recurrent and ablation). See the timing smoke.
CEM_BUDGET = CEMConfig(population=64, n_iter=25, elite_frac=0.20, seed=0)

REF_WAYPOINT_GATE = 1.00
REF_STATE_MACHINE_GATE = 0.62
REF_STAGE2C_RECURRENT = 0.57


def _load_prior():
    prior = []
    for path in PRIOR_SPLIT_PATHS:
        prior += load_scenarios(path)
    return prior


def build_disjoint_splits():
    """Build the Stage 2d gate and train splits, disjoint by exact scenario
    identity from every prior split and from each other. Route reuse is allowed;
    only identities must differ. Raises if any overlap slips through."""
    prior = _load_prior()
    gate = generate_stage2b_split(STAGE2D_GATE_SEED, exclude=prior, counts=GATE_COUNTS)
    train = generate_stage2b_split(STAGE2D_TRAIN_SEED, exclude=prior + gate,
                                   counts=TRAIN_COUNTS)
    prior_keys = {_scenario_key(s) for s in prior}
    gate_keys = {_scenario_key(s) for s in gate}
    train_keys = {_scenario_key(s) for s in train}
    assert len(gate_keys) == len(gate), "duplicate identity within gate"
    assert len(train_keys) == len(train), "duplicate identity within train"
    assert gate_keys.isdisjoint(prior_keys), "gate overlaps a prior split"
    assert train_keys.isdisjoint(prior_keys), "train overlaps a prior split"
    assert gate_keys.isdisjoint(train_keys), "gate overlaps train"
    return gate, train


def _freeze_splits(gate_path: str, train_path: str):
    """Freeze the splits to disk once; reuse the frozen files thereafter so the
    one-shot gate is stable across runs."""
    if Path(gate_path).exists() and Path(train_path).exists():
        return load_scenarios(gate_path), load_scenarios(train_path)
    gate, train = build_disjoint_splits()
    Path(gate_path).parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(gate_path, gate)
    save_scenarios(train_path, train)
    return gate, train


def select_fitness_scenarios(train):
    """40 fixed, stratified fitness scenarios: the first k per layout (the split
    is already shuffled by its generation seed, so this is deterministic)."""
    chosen = []
    for layout, k in FITNESS_STRATA.items():
        chosen += [s for s in train if s.layout == layout][:k]
    return chosen


def _closed_loop_breakdown(esn, scenarios, layouts):
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def _one_shot_gate(recurrent: bool, fitness_scenarios, gate, layouts):
    esn = make_reservoir(recurrent)
    esn, info = train_readout_by_reward(esn, fitness_scenarios, layouts, CEM_BUDGET)
    breakdown = _closed_loop_breakdown(esn, gate, layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    return {"passes_gate": passes, "overall_arrival_rate": overall,
            "by_layout_arrival_rate": per_layout, "breakdown": breakdown,
            "train_final_mu_fitness": info["final_mu_fitness"],
            "train_history": info["history"]}


def _interpretation(recurrent: dict, ablation: dict) -> str:
    r = recurrent["overall_arrival_rate"]
    a = ablation["overall_arrival_rate"]
    lead = (f"Recurrent {r:.3f} vs memoryless ablation {a:.3f} (references: "
            f"waypoint witness {REF_WAYPOINT_GATE:.2f}, Stage 2b SM "
            f"{REF_STATE_MACHINE_GATE:.2f}, Stage 2c recurrent "
            f"{REF_STAGE2C_RECURRENT:.2f}).")
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " The reward-trained recurrent baseline clears the gate and "
                "beats its matched ablation: temporal memory over the same "
                "observation interface, given a memory-requiring (reward) signal, "
                "closes the gap imitation could not. The interface is not the "
                "bottleneck; this motivates the biologically grounded "
                "memory/action-selection work (dopamine as modulation/teaching, "
                "never goal bearing).")
    if recurrent["passes_gate"]:
        return (lead + " The reward-trained recurrent baseline clears the gate but "
                "does not clearly beat its ablation, so reward training -- not "
                "temporal memory specifically -- closes the gap; the interface is "
                "sufficient and memory is not required for it.")
    return (lead + " The reward-trained recurrent baseline does not clear the gate. "
            "Per the pass-only-is-conclusive asymmetry this stays confounded "
            "between optimization budget, capacity, and observation limits -- it "
            "does not implicate the interface on its own; diagnose before any "
            "interface claim or dopamine/connectome learning.")


def _timing_smoke(fitness_scenarios, layouts, n_evals: int = 3):
    """Time a few full fitness evaluations and project the frozen-budget cost."""
    import numpy as np
    from stage2c import N_FEATURES
    esn = make_reservoir(True)
    esn.W_out = np.zeros((2, 1 + N_FEATURES + esn.n))
    ctrl = RecurrentController(esn)

    def one_eval():
        for s in fitness_scenarios:
            ctrl.reset()
            run_street_episode(s, layouts[s.layout], ctrl, record=True)

    one_eval()  # warm up
    t0 = time.perf_counter()
    for _ in range(n_evals):
        one_eval()
    per_eval = (time.perf_counter() - t0) / n_evals
    per_iter = per_eval * CEM_BUDGET.population
    total = per_iter * CEM_BUDGET.n_iter * 2  # recurrent + ablation
    print(f"[smoke] {len(fitness_scenarios)} scenarios/eval; "
          f"{per_eval:.3f}s per fitness eval (population member)")
    print(f"[smoke] projected per CEM iteration: {per_iter:.1f}s "
          f"({CEM_BUDGET.population} members)")
    print(f"[smoke] projected TOTAL (pop {CEM_BUDGET.population} x iters "
          f"{CEM_BUDGET.n_iter} x 2 models): {total/60:.1f} min "
          f"({int(CEM_BUDGET.population*CEM_BUDGET.n_iter*2)} rollouts x "
          f"{len(fitness_scenarios)})")
    return per_eval, total


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2d reward-trained baseline.")
    parser.add_argument("--gate-split", default="runs/stage2d/gate_split.json")
    parser.add_argument("--train-split", default="runs/stage2d/train_split.json")
    parser.add_argument("--out", default="runs/stage2d/results.json")
    parser.add_argument("--gate-out", default="runs/stage2d/gate_results.json")
    parser.add_argument("--smoke", action="store_true",
                        help="timing-only smoke test; no optimization, no gate")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    gate, train = _freeze_splits(args.gate_split, args.train_split)
    fitness = select_fitness_scenarios(train)
    print(f"Stage 2d: gate {len(gate)} (frozen, one-shot), train {len(train)}, "
          f"fitness {len(fitness)} stratified {FITNESS_STRATA}.")

    if args.smoke:
        _timing_smoke(fitness, layouts)
        return

    print(f"Training readout by CEM {CEM_BUDGET} (recurrent, then ablation)...")
    t0 = time.perf_counter()
    recurrent = _one_shot_gate(True, fitness, gate, layouts)
    ablation = _one_shot_gate(False, fitness, gate, layouts)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min.")

    interpretation = _interpretation(recurrent, ablation)
    provenance = {p: sha256(p) for p in PRIOR_SPLIT_PATHS}
    provenance[args.gate_split] = sha256(args.gate_split)
    provenance[args.train_split] = sha256(args.train_split)
    gate_payload = {
        "gate_split": args.gate_split, "train_split": args.train_split,
        "provenance_sha256": provenance,
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "reservoir_config": RESERVOIR_CONFIG, "esn_seed": ESN_SEED,
        "cem_budget": vars(CEM_BUDGET), "fitness_strata": FITNESS_STRATA,
        "recurrent": {k: v for k, v in recurrent.items() if k != "train_history"},
        "memoryless_ablation": {k: v for k, v in ablation.items()
                                if k != "train_history"},
        "references": {"waypoint_witness": REF_WAYPOINT_GATE,
                       "state_machine": REF_STATE_MACHINE_GATE,
                       "stage2c_recurrent": REF_STAGE2C_RECURRENT},
        "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["recurrent_train_history"] = recurrent["train_history"]
    results_payload["ablation_train_history"] = ablation["train_history"]
    Path(args.out).write_text(json.dumps(results_payload, indent=2) + "\n")

    print(f"\nRecurrent gate:  overall={recurrent['overall_arrival_rate']:.3f} "
          f"{recurrent['by_layout_arrival_rate']} "
          f"{'PASS' if recurrent['passes_gate'] else 'FAIL'}")
    print(f"Ablation gate:   overall={ablation['overall_arrival_rate']:.3f} "
          f"{ablation['by_layout_arrival_rate']} "
          f"{'PASS' if ablation['passes_gate'] else 'FAIL'}")
    print(f"Wrote {args.gate_out} and {args.out}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
