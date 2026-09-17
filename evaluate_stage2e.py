"""Stage 2e evaluation: warm-start reward fine-tuning of the same-observation
readout, scored once on a fresh, exact-disjoint gate.

Provenance discipline (see the spec). One fresh gate split is built with
`generate_stage2b_split` (outward-facing road-end starts excluded), disjoint by
exact scenario identity from the CORRECTED complete prior set AND the spent Stage
2d gate + train splits. Under that exclusion exactly 7 eligible `cross` scenarios
remain, so the gate is predeclared 7/44/44 = 95 (all 7 cross used; the cross gate
of 0.80 then requires 6/7). The readout is warm-started at the deterministic Stage
2c readout and fine-tuned by CEM on the exact 40 Stage 2d fitness scenarios (which
the fresh gate fully excludes), identically for the recurrent model and its
ablation. The frozen gate is evaluated exactly once for each.

Interpretation (bounded, asymmetric). Only a PASS is conclusive -- evidence reward
exploited the available policy state to clear a fresh gate. Recurrent beating its
ablation further implicates temporal memory. Both failing stays confounded between
optimization budget/search, capacity, and observation limits; it does not
implicate the interface on its own.

The spent Stage 2d gate is never rerun or replaced. Does not touch the connectome.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from street import initial_layouts
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, load_scenarios, save_scenarios, sha256,
)
from evaluate_stage2b import (
    _breakdown, _gate, _run_per_scenario, _scenario_key, generate_stage2b_split,
)
from evaluate_stage2d import PRIOR_SPLIT_PATHS, select_fitness_scenarios
from stage2c import RecurrentController
from stage2d import CEMConfig, ESN_SEED, RESERVOIR_CONFIG, episode_reward
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
)

# Complete exclusion: the corrected prior set PLUS the spent Stage 2d splits.
SPENT_STAGE2D_SPLITS = [
    "runs/stage2d/gate_split.json",
    "runs/stage2d/train_split.json",
]
EXCLUDE_PATHS = list(PRIOR_SPLIT_PATHS) + SPENT_STAGE2D_SPLITS

# Fitness = the exact 40 Stage 2d fitness scenarios (4/18/18), drawn from the 2d
# train split -- which the fresh gate fully excludes, so gate INTERSECT fitness = 0.
FITNESS_SPLIT = "runs/stage2d/train_split.json"

# Fresh seed (distinct from 2b's 21/22 and 2d's 41/42).
STAGE2E_GATE_SEED = 51
# Predeclared: only 7 eligible `cross` remain under the full exclusion.
GATE_COUNTS = {"cross": 7, "regular": 44, "asymmetric": 44}

# Frozen warm-start CEM budget (identical for recurrent and ablation). init_std is
# a SMALL perturbation around the warm start (design choice; see the spec).
CEM_BUDGET = CEMConfig(population=64, n_iter=25, elite_frac=0.20,
                       init_std=0.1, seed=0)

# Reproduction gate: the warm start must reproduce the known-policy check exactly.
REPRO_ARRIVALS = 22
REPRO_REWARD = {True: 1.106, False: 1.086}
REPRO_REWARD_TOL = 0.02

REF_WAYPOINT_GATE = 1.00
REF_STATE_MACHINE_GATE = 0.62
REF_STAGE2C_RECURRENT = 0.57
REF_STAGE2D_RECURRENT = 0.00


def _load(paths):
    out = []
    for p in paths:
        out += load_scenarios(p)
    return out


def build_gate_split():
    """Fresh gate, exact-identity disjoint from the complete prior set, the spent
    Stage 2d splits, and the fitness set. Raises if any overlap slips through."""
    exclude = _load(EXCLUDE_PATHS)
    fitness = select_fitness_scenarios(load_scenarios(FITNESS_SPLIT))
    gate = generate_stage2b_split(STAGE2E_GATE_SEED, exclude=exclude + fitness,
                                  counts=GATE_COUNTS)
    gate_keys = {_scenario_key(s) for s in gate}
    assert len(gate_keys) == len(gate), "duplicate identity within gate"
    assert gate_keys.isdisjoint({_scenario_key(s) for s in exclude}), \
        "gate overlaps a prior/2d split"
    assert gate_keys.isdisjoint({_scenario_key(s) for s in fitness}), \
        "gate overlaps the fitness set"
    return gate


def _freeze_gate(gate_path: str):
    if Path(gate_path).exists():
        return load_scenarios(gate_path)
    gate = build_gate_split()
    Path(gate_path).parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(gate_path, gate)
    return gate


def _closed_loop_breakdown(esn, scenarios, layouts):
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def _reproduction_gate(recurrent: bool, esn, fitness, layouts) -> dict:
    """Assert the warm start reproduces the known-policy check before optimizing."""
    out = evaluate_policy(esn, fitness, layouts)
    exp = REPRO_REWARD[recurrent]
    ok = (out["arrivals"] == REPRO_ARRIVALS
          and abs(out["mean_reward"] - exp) <= REPRO_REWARD_TOL)
    if not ok:
        raise SystemExit(
            f"reproduction gate FAILED (recurrent={recurrent}): got "
            f"{out['arrivals']}/{out['n']} arrivals, reward {out['mean_reward']:.4f}; "
            f"expected {REPRO_ARRIVALS} arrivals, reward ~{exp}. The Stage 2c "
            f"readout was not reconstructed exactly.")
    return out


def _one_shot(recurrent: bool, fitness, gate, layouts) -> dict:
    esn, theta0 = warm_start_readout(recurrent, layouts)
    repro = _reproduction_gate(recurrent, esn, fitness, layouts)
    esn, info = train_readout_by_reward_warmstart(esn, theta0, fitness, layouts,
                                                  CEM_BUDGET)
    breakdown = _closed_loop_breakdown(esn, gate, layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    return {
        "passes_gate": passes, "overall_arrival_rate": overall,
        "by_layout_arrival_rate": per_layout, "breakdown": breakdown,
        "reproduction": repro,
        "warmstart_fitness": info["warmstart_fitness"],
        "best_fitness": info["best_fitness"],
        "improved_over_warmstart": info["improved_over_warmstart"],
        "best_outcomes_on_fitness": info["best_outcomes"],
        "best_theta": info["best_theta"],
        "train_history": info["history"],
    }


def _interpretation(recurrent: dict, ablation: dict) -> str:
    r = recurrent["overall_arrival_rate"]
    a = ablation["overall_arrival_rate"]
    lead = (f"Recurrent {r:.3f} vs memoryless ablation {a:.3f} (references: "
            f"waypoint witness {REF_WAYPOINT_GATE:.2f}, Stage 2b SM "
            f"{REF_STATE_MACHINE_GATE:.2f}, Stage 2c recurrent "
            f"{REF_STAGE2C_RECURRENT:.2f}, Stage 2d recurrent "
            f"{REF_STAGE2D_RECURRENT:.2f}).")
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " Warm-start reward fine-tuning of temporal memory over the "
                "observation interface clears the fresh gate and beats its ablation: "
                "reward exploited memory over the observation stream to close the "
                "gap. Motivates the biologically grounded memory/action-selection "
                "work (dopamine as modulation/teaching, never goal bearing).")
    if recurrent["passes_gate"]:
        return (lead + " The warm-started recurrent policy clears the fresh gate but "
                "does not clearly beat its ablation, so reward exploited the reactive "
                "policy state (not memory specifically); the interface is sufficient "
                "and memory is not required for it.")
    return (lead + " Warm-start reward fine-tuning did not clear the fresh gate. Per "
            "the pass-only-is-conclusive asymmetry this stays bounded and confounded "
            "between optimization budget/search, capacity, and observation limits -- "
            "it does not implicate the interface on its own; diagnose (curriculum, "
            "denser shaping, stronger optimizer) before any interface claim or "
            "dopamine/connectome learning.")


def _timing_smoke(fitness, layouts, n_evals: int = 3):
    import numpy as np
    esn, theta0 = warm_start_readout(True, layouts)
    d = theta0.size // 2
    ctrl = RecurrentController(esn)
    from street import run_street_episode

    def one_eval():
        esn.W_out = np.asarray(theta0).reshape(2, d)
        for s in fitness:
            ctrl.reset()
            run_street_episode(s, layouts[s.layout], ctrl, record=True)

    one_eval()
    t0 = time.perf_counter()
    for _ in range(n_evals):
        one_eval()
    per_eval = (time.perf_counter() - t0) / n_evals
    total = per_eval * CEM_BUDGET.population * CEM_BUDGET.n_iter * 2
    print(f"[smoke] {len(fitness)} scenarios/eval; {per_eval:.3f}s per fitness eval")
    print(f"[smoke] projected TOTAL (pop {CEM_BUDGET.population} x iters "
          f"{CEM_BUDGET.n_iter} x 2 models): {total/60:.1f} min")
    return per_eval, total


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2e warm-start reward.")
    parser.add_argument("--gate-split", default="runs/stage2e/gate_split.json")
    parser.add_argument("--out", default="runs/stage2e/results.json")
    parser.add_argument("--gate-out", default="runs/stage2e/gate_results.json")
    parser.add_argument("--smoke", action="store_true",
                        help="timing-only smoke test; no optimization, no gate")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    gate = _freeze_gate(args.gate_split)
    fitness = select_fitness_scenarios(load_scenarios(FITNESS_SPLIT))
    print(f"Stage 2e: gate {len(gate)} (frozen, one-shot, 7/44/44), fitness "
          f"{len(fitness)} (the 40 Stage 2d fitness scenarios).")

    if args.smoke:
        _timing_smoke(fitness, layouts)
        return

    print(f"Warm-start CEM {CEM_BUDGET} (recurrent, then ablation)...")
    t0 = time.perf_counter()
    recurrent = _one_shot(True, fitness, gate, layouts)
    ablation = _one_shot(False, fitness, gate, layouts)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min.")

    interpretation = _interpretation(recurrent, ablation)
    provenance = {p: sha256(p) for p in EXCLUDE_PATHS}
    provenance[args.gate_split] = sha256(args.gate_split)
    strip = lambda d: {k: v for k, v in d.items()
                       if k not in ("train_history", "best_theta")}
    gate_payload = {
        "gate_split": args.gate_split, "fitness_split": FITNESS_SPLIT,
        "provenance_sha256": provenance,
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "reservoir_config": RESERVOIR_CONFIG, "esn_seed": ESN_SEED,
        "cem_budget": vars(CEM_BUDGET), "warm_start": "stage2c_readout",
        "gate_counts": GATE_COUNTS,
        "cross_gate_note": "7 cross scenarios; the 0.80 cross gate requires 6/7.",
        "recurrent": strip(recurrent),
        "memoryless_ablation": strip(ablation),
        "references": {"waypoint_witness": REF_WAYPOINT_GATE,
                       "state_machine": REF_STATE_MACHINE_GATE,
                       "stage2c_recurrent": REF_STAGE2C_RECURRENT,
                       "stage2d_recurrent": REF_STAGE2D_RECURRENT},
        "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["recurrent"] = recurrent          # full, incl. best_theta + history
    results_payload["memoryless_ablation"] = ablation
    Path(args.out).write_text(json.dumps(results_payload, indent=2) + "\n")

    print(f"\nRecurrent gate:  overall={recurrent['overall_arrival_rate']:.3f} "
          f"{recurrent['by_layout_arrival_rate']} "
          f"{'PASS' if recurrent['passes_gate'] else 'FAIL'} "
          f"(warm {recurrent['warmstart_fitness']:.3f} -> best "
          f"{recurrent['best_fitness']:.3f})")
    print(f"Ablation gate:   overall={ablation['overall_arrival_rate']:.3f} "
          f"{ablation['by_layout_arrival_rate']} "
          f"{'PASS' if ablation['passes_gate'] else 'FAIL'} "
          f"(warm {ablation['warmstart_fitness']:.3f} -> best "
          f"{ablation['best_fitness']:.3f})")
    print(f"Wrote {args.gate_out} and {args.out}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
