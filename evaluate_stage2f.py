"""Stage 2f evaluation: aligned-reward warm-start fine-tuning, scored once on a
fresh disjoint gate with a diagonal, waypoint-witnessed cross stratum.

Provenance discipline (see the spec). The cross layout's cardinal-heading pool is
exhausted (all 64 eligible pairs consumed by Stages 2/2b/2d/2e). The ONLY source of
fresh cross is non-cardinal start headings, so this stage adds the four diagonal
headings and filters the result to WAYPOINT-WITNESSED scenarios (the privileged
witness arrives) -- a positive filter, never a solvability claim. Regular/asym are
generated unchanged with `generate_stage2b_split` (cardinal). The fitness split is
frozen FIRST (seed 60); the gate is frozen SECOND (seed 61) excluding it, so
gate INTERSECT fitness = 0 by construction and by assertion.

Interpretation (bounded, asymmetric). Only a PASS is conclusive, scoped to this
diagonal witnessed cross stratum plus the cardinal regular/asym strata. Both
failing stays confounded between reward alignment and scenario coverage; it does
not implicate the interface. No spent gate is rerun or replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from street import (
    MAX_SIM_TIME, StreetCarState, StreetScenario, initial_layouts,
    run_street_episode,
)
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, _scenario_key, _waypoints,
    load_scenarios, save_scenarios, sha256,
)
from evaluate_stage2b import (
    _breakdown, _gate, _run_per_scenario, generate_stage2b_split,
)
from evaluate_stage2d import PRIOR_SPLIT_PATHS, select_fitness_scenarios
from stage2b import WaypointController, is_outward_road_end
from stage2c import RecurrentController
from stage2d import CEMConfig, ESN_SEED, RESERVOIR_CONFIG
from stage2e import (
    evaluate_policy, train_readout_by_reward_warmstart, warm_start_readout,
)
from stage2f import aligned_fitness

DIAGONAL_HEADINGS = (math.pi / 4, 3 * math.pi / 4, -math.pi / 4, -3 * math.pi / 4)

# Complete exclusion: the corrected prior set PLUS every spent one-shot split.
SPENT_SPLITS = [
    "runs/stage2d/gate_split.json",
    "runs/stage2d/train_split.json",
    "runs/stage2e/gate_split.json",
]
EXCLUDE_PATHS = list(PRIOR_SPLIT_PATHS) + SPENT_SPLITS

# Preregistered seeds (distinct from 2b 21/22, 2d 41/42, 2e 51). Fitness FIRST.
STAGE2F_FITNESS_SEED = 60
STAGE2F_GATE_SEED = 61
FITNESS_COUNTS = {"cross": 6, "regular": 40, "asymmetric": 40}    # 86
GATE_COUNTS = {"cross": 12, "regular": 44, "asymmetric": 44}      # 100

# The 40 Stage 2d fitness scenarios -- used ONLY as the reproduction fingerprint
# (arrivals == 22), never optimized on. Excluded from both fresh splits via
# runs/stage2d/train_split.json in EXCLUDE_PATHS.
REPRO_REFERENCE_SPLIT = "runs/stage2d/train_split.json"

# Frozen warm-start CEM budget (identical for recurrent and ablation).
CEM_BUDGET = CEMConfig(population=64, n_iter=25, elite_frac=0.20,
                       init_std=0.1, seed=0)

REPRO_ARRIVALS = 22
REPRO_REWARD = {True: 1.106, False: 1.086}
REPRO_REWARD_TOL = 0.02

REF_WAYPOINT_GATE = 1.00
REF_STATE_MACHINE_GATE = 0.62
REF_STAGE2C_RECURRENT = 0.57
REF_STAGE2D_RECURRENT = 0.00
REF_STAGE2E_RECURRENT = 0.516


def _load(paths):
    out = []
    for p in paths:
        out += load_scenarios(p)
    return out


def _cardinal_strata(seed: int, exclude, counts: dict) -> list[StreetScenario]:
    """Regular + asymmetric from the unchanged cardinal generator (cross count 0)."""
    strata = dict(counts)
    strata["cross"] = 0
    return [s for s in generate_stage2b_split(seed, exclude=exclude, counts=strata)
            if s.layout != "cross"]


def build_fitness_split() -> list[StreetScenario]:
    """Fresh 86-scenario fitness split (6 diagonal cross / 40 regular / 40 asym),
    disjoint from the complete exclusion. Generated FIRST (seed 60)."""
    exclude = _load(EXCLUDE_PATHS)
    cross = generate_expanded_cross(STAGE2F_FITNESS_SEED, exclude, FITNESS_COUNTS["cross"])
    rest = _cardinal_strata(STAGE2F_FITNESS_SEED, exclude, FITNESS_COUNTS)
    split = cross + rest
    _assert_disjoint(split, exclude, "fitness")
    return split


def build_gate_split(fitness) -> list[StreetScenario]:
    """Fresh 100-scenario gate (12 diagonal cross / 44 regular / 44 asym), disjoint
    from the complete exclusion AND the frozen fitness split. Generated SECOND
    (seed 61) excluding `fitness`, so gate INTERSECT fitness = 0."""
    exclude = _load(EXCLUDE_PATHS) + list(fitness)
    cross = generate_expanded_cross(STAGE2F_GATE_SEED, exclude, GATE_COUNTS["cross"])
    rest = _cardinal_strata(STAGE2F_GATE_SEED, exclude, GATE_COUNTS)
    split = cross + rest
    _assert_disjoint(split, exclude, "gate")
    return split


def _assert_disjoint(split, exclude, label: str) -> None:
    keys = {_scenario_key(s) for s in split}
    assert len(keys) == len(split), f"duplicate identity within {label}"
    assert keys.isdisjoint({_scenario_key(s) for s in exclude}), \
        f"{label} overlaps an excluded/fitness split"


def _freeze(path: str, builder) -> list[StreetScenario]:
    if Path(path).exists():
        return load_scenarios(path)
    split = builder()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(path, split)
    return split


def generate_expanded_cross(seed: int, exclude, count: int) -> list[StreetScenario]:
    """Fresh diagonal-heading cross scenarios, waypoint-witnessed and identity-
    disjoint from `exclude`. Shuffles candidates deterministically by `seed`, then
    takes the first `count` whose waypoint witness arrives."""
    rng = random.Random(seed)
    layout = initial_layouts()["cross"]
    points = _waypoints(layout)
    pairs = [(a, b) for a in points for b in points
             if a != b and math.dist(a, b) >= 12.0]
    excluded = {_scenario_key(s) for s in (exclude or [])}
    candidates = [(a, b, h) for (a, b) in pairs for h in DIAGONAL_HEADINGS
                  if ("cross", a[0], a[1], h, b[0], b[1]) not in excluded
                  and not is_outward_road_end(a[0], a[1], h, layout)]
    rng.shuffle(candidates)
    chosen: list[StreetScenario] = []
    for (a, b, h) in candidates:
        scenario = StreetScenario(
            "cross", StreetCarState(a[0], a[1], h, 0.0), b[0], b[1],
            timeout=MAX_SIM_TIME, label=f"stage2f-cross-{len(chosen):03d}")
        wc = WaypointController(scenario, layout)
        wc.reset()
        if run_street_episode(scenario, layout, wc).outcome == "arrival":
            chosen.append(scenario)
            if len(chosen) == count:
                return chosen
    raise SystemExit(f"expanded cross: only {len(chosen)} witnessed scenarios "
                     f"(need {count}) under the given exclusion")


def _closed_loop_breakdown(esn, scenarios, layouts):
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def _reproduction_gate(recurrent: bool, esn, repro_ref, layouts) -> dict:
    """Assert the warm start reproduces the known-policy fingerprint (arrivals==22
    on the 40 Stage 2d fitness scenarios) before optimizing. Coefficient-
    independent: uses the 2e episode-reward `mean_reward`, not the aligned fitness."""
    out = evaluate_policy(esn, repro_ref, layouts)
    exp = REPRO_REWARD[recurrent]
    ok = (out["arrivals"] == REPRO_ARRIVALS
          and abs(out["mean_reward"] - exp) <= REPRO_REWARD_TOL)
    if not ok:
        raise SystemExit(
            f"reproduction gate FAILED (recurrent={recurrent}): got "
            f"{out['arrivals']}/{out['n']} arrivals, reward {out['mean_reward']:.4f}; "
            f"expected {REPRO_ARRIVALS}, reward ~{exp}. Stage 2c readout not "
            f"reconstructed exactly.")
    return out


def _one_shot(recurrent: bool, repro_ref, fitness, gate, layouts) -> dict:
    esn, theta0 = warm_start_readout(recurrent, layouts)
    repro = _reproduction_gate(recurrent, esn, repro_ref, layouts)
    esn, info = train_readout_by_reward_warmstart(
        esn, theta0, fitness, layouts, CEM_BUDGET, fitness_fn=aligned_fitness)
    breakdown = _closed_loop_breakdown(esn, gate, layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    return {
        "passes_gate": passes, "overall_arrival_rate": overall,
        "by_layout_arrival_rate": per_layout, "breakdown": breakdown,
        "reproduction": repro,
        "warmstart_fitness": info["warmstart_fitness"],
        "warmstart_outcomes_on_fitness": info["warmstart_outcomes"],
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
            f"{REF_STAGE2D_RECURRENT:.2f}, Stage 2e recurrent "
            f"{REF_STAGE2E_RECURRENT:.2f}).")
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " Arrival-primary aligned fitness, fine-tuning temporal "
                "memory over the observation interface, clears a fresh gate and "
                "beats its ablation. Scoped to this diagonal waypoint-witnessed "
                "cross stratum plus the cardinal regular/asym strata, the interface "
                "is not the bottleneck -- not a claim about cross scenarios outside "
                "the witnessed eligible set. Motivates the biologically grounded "
                "memory/action-selection work (dopamine as modulation/teaching, "
                "never goal bearing).")
    if recurrent["passes_gate"]:
        return (lead + " The warm-started recurrent policy clears the fresh gate but "
                "does not clearly beat its ablation, so aligned reward exploited the "
                "reactive policy state (not memory specifically); the interface is "
                "sufficient and memory is not required for it, over this stratum.")
    return (lead + " Aligned-reward fine-tuning did not clear the fresh gate. Per "
            "the pass-only-is-conclusive asymmetry this stays bounded and confounded "
            "between reward alignment and scenario coverage (and residually "
            "capacity/optimization); it does not implicate the interface. Report the "
            "recurrent-ablation gap and the collision/arrival breakdown; diagnose "
            "before any interface claim or dopamine/connectome learning.")


def _timing_smoke(fitness, layouts, n_evals: int = 3):
    import numpy as np
    esn, theta0 = warm_start_readout(True, layouts)
    d = theta0.size // 2
    esn.W_out = np.asarray(theta0).reshape(2, d)

    def one_eval():
        aligned_fitness(esn, fitness, layouts)

    one_eval()  # warm up
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
    parser = argparse.ArgumentParser(description="Stage 2f aligned-reward warm-start.")
    parser.add_argument("--fitness-split", default="runs/stage2f/fitness_split.json")
    parser.add_argument("--gate-split", default="runs/stage2f/gate_split.json")
    parser.add_argument("--out", default="runs/stage2f/results.json")
    parser.add_argument("--gate-out", default="runs/stage2f/gate_results.json")
    parser.add_argument("--smoke", action="store_true",
                        help="timing-only smoke test; no optimization, no gate")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    fitness = _freeze(args.fitness_split, build_fitness_split)
    gate = _freeze(args.gate_split, lambda: build_gate_split(fitness))
    repro_ref = select_fitness_scenarios(load_scenarios(REPRO_REFERENCE_SPLIT))
    print(f"Stage 2f: fitness {len(fitness)} (6/40/40, diagonal cross), gate "
          f"{len(gate)} (12/44/44, diagonal cross). Repro ref {len(repro_ref)}.")

    if args.smoke:
        _timing_smoke(fitness, layouts)
        return

    print(f"Warm-start CEM {CEM_BUDGET} (recurrent, then ablation)...")
    t0 = time.perf_counter()
    recurrent = _one_shot(True, repro_ref, fitness, gate, layouts)
    ablation = _one_shot(False, repro_ref, fitness, gate, layouts)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min.")

    interpretation = _interpretation(recurrent, ablation)
    provenance = {p: sha256(p) for p in EXCLUDE_PATHS}
    provenance[args.fitness_split] = sha256(args.fitness_split)
    provenance[args.gate_split] = sha256(args.gate_split)
    strip = lambda d: {k: v for k, v in d.items()
                       if k not in ("train_history", "best_theta")}
    gate_payload = {
        "fitness_split": args.fitness_split, "gate_split": args.gate_split,
        "repro_reference_split": REPRO_REFERENCE_SPLIT,
        "provenance_sha256": provenance,
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "reservoir_config": RESERVOIR_CONFIG, "esn_seed": ESN_SEED,
        "cem_budget": vars(CEM_BUDGET), "warm_start": "stage2c_readout",
        "fitness_counts": FITNESS_COUNTS, "gate_counts": GATE_COUNTS,
        "cross_note": ("diagonal-heading waypoint-witnessed cross; NOT "
                       "orientation-comparable to prior stages; 0.80 cross gate "
                       "needs 10/12."),
        "recurrent": strip(recurrent),
        "memoryless_ablation": strip(ablation),
        "references": {"waypoint_witness": REF_WAYPOINT_GATE,
                       "state_machine": REF_STATE_MACHINE_GATE,
                       "stage2c_recurrent": REF_STAGE2C_RECURRENT,
                       "stage2d_recurrent": REF_STAGE2D_RECURRENT,
                       "stage2e_recurrent": REF_STAGE2E_RECURRENT},
        "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["recurrent"] = recurrent          # full, incl. best_theta + history
    results_payload["memoryless_ablation"] = ablation
    Path(args.out).write_text(json.dumps(results_payload, indent=2) + "\n")

    for name, res in (("Recurrent", recurrent), ("Ablation", ablation)):
        print(f"{name} gate:  overall={res['overall_arrival_rate']:.3f} "
              f"{res['by_layout_arrival_rate']} "
              f"{'PASS' if res['passes_gate'] else 'FAIL'} "
              f"(warm {res['warmstart_fitness']:.3f} -> best {res['best_fitness']:.3f})")
    print(f"Wrote {args.gate_out} and {args.out}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
