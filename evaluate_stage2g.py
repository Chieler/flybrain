"""Stage 2g evaluation: fully-trainable recurrent baseline, scored once on a newly
frozen 0.90/0.80 gate. Three newly frozen, Stage-2g-policy-unseen splits (train 66 /
dev 46 / gate 100) share the scarce diagonal witnessed-cross pool; they are frozen
and sha256-hashed BEFORE any training. The gate is generated last and never inspected
until the single scoring event.

Disjointness follows the Variant B enforced set (2026-09-21 amendment): every prior
SCORED GATE + 2f fitness + BC demo + Stage 2g's own earlier splits. Full exclusion of
all prior TRAINING splits was infeasible (only 17 regular scenarios remained). Older
prior training splits may recur -- historical design exposure, NOT Stage 2g
policy-training leakage; overlap counts vs every prior split are recorded.

Escalation is a hard halt: if the trained n=8 recurrent model fails go/no-go or fails
to improve DEV arrivals over its warm start, the run stops before the gate -- n=16 +
CMA-ES is NOT implemented here and requires a written spec amendment first. Does not
touch the connectome or any frozen Stage 2..2f artifact.
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
from stage2d import CEMConfig
from stage2e import evaluate_policy
from stage2f import aligned_fitness
from evaluate_stage2 import GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE
from evaluate_stage2b import _breakdown, _gate, _run_per_scenario
from stage2c import RecurrentController

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
        f"{label} overlaps an enforced-excluded/earlier split"


def build_split(seed: int, counts: dict, extra_exclude) -> list:
    """One newly frozen stratified split (diagonal witnessed cross + cardinal
    reg/asym), disjoint from EXCLUDE_PATHS plus `extra_exclude` (earlier splits).
    Guards cross generation so a cross count of 0 yields no cross scenarios."""
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
    overlaps are asserted zero (fail loudly); re-admitted training-split overlaps are
    recorded as historical exposure (not Stage 2g policy-training leakage)."""
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
    warm_train_arr = _arrivals(esn, theta0, train, layouts, recurrent)  # esn->theta0
    warm_dev_arr = _arrivals(esn, theta0, dev, layouts, recurrent)

    gng = stage2g.go_no_go(esn, theta0, scales, train, layouts, cfg, recurrent)
    if not gng["passed"]:
        stage2g.set_theta(esn, theta0, recurrent)  # go_no_go leaves esn at a probe cand.
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
        # rec passed, ablation did NOT pass, and the overall margin is within 0.02.
        # A failed memoryless arm is weak evidence FOR memory, not against it -- do
        # not claim memory is unnecessary here (confined to the both-pass branch).
        return (lead + " The recurrent policy clears the gate while the recurrence-off "
                "ablation does not, yet the overall-rate margin is within 0.02, so the "
                "edge sits in the stratum the ablation missed rather than a broad gap. "
                "Weak, mixed evidence: report the per-layout breakdown; a failed "
                "memoryless arm does not license any 'memory unnecessary' claim."
                + _ATTRIBUTION)
    return (lead + " Neither arm cleared the fresh gate. Under the pass-only asymmetry "
            "this stays bounded and confounded between optimization budget, capacity "
            "(n=8) and coverage; it does not implicate the observation interface. "
            "Report the recurrent-ablation gap and collision/arrival breakdown before "
            "any interface claim.")


def _strip(d: dict) -> dict:
    # drop "history" (re-added as "train_history" in the full results.json) and "best"
    # (a z-space weight sample, redundant with best_theta/best_fitness) so no weight
    # vector or per-iteration trace lingers in the sealed gate_results.json.
    return {k: v for k, v in d.items()
            if k not in ("history", "train_history", "best_theta", "best")}


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
