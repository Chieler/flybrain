"""Stage 2c evaluation: train the same-observation recurrent baseline by behavior
cloning the waypoint witness, and score it once on the frozen Stage 2b gate.

Provenance discipline (identical to Stage 2b). Hyperparameters are selected on
`runs/stage2b/sm_train_split.json` ONLY -- via a deterministic train/val split
*within* that file -- and the frozen `runs/stage2b/gate_split.json` is evaluated
exactly once, for both the recurrent model and its matched memoryless ablation.

Interpretation (bounded; see HANDOFF Stage 2c). Reference points from Stage 2b:
the privileged waypoint witness scores 1.00 on the gate and the memoryless
observation state machine scores 0.62. If the recurrent baseline clears the
90%/80% gate AND beats its own memoryless ablation, temporal memory over the
observation interface -- not the interface itself -- is the missing piece. If it
does not, the interface (or imitation/capacity) is implicated, to be diagnosed
before any dopamine/connectome learning. This is imitation of a privileged
teacher; a pass is evidence within that frame, not proof of general sufficiency.

Not neural in the connectome sense; does not touch the connectome or the frozen
Stage 2 / Stage 2b artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from street import StreetLayout, initial_layouts, run_street_episode
from evaluate_stage2 import (
    GATE_OVERALL_RATE, GATE_PER_LAYOUT_RATE, load_scenarios, sha256,
)
from evaluate_stage2b import _breakdown, _gate, _run_per_scenario
from stage2b import WaypointController
from stage2c import (
    EchoStateNetwork, RecurrentController, TrajectoryRecorder,
)

# Fixed reservoir seed (determinism) and the within-train validation split.
ESN_SEED = 0
VAL_SEED = 23
VAL_FRACTION = 0.3

# Reference gate scores from Stage 2b (runs/stage2b/gate_results.json).
REF_WAYPOINT_GATE = 1.00
REF_STATE_MACHINE_GATE = 0.62

# Hyperparameter grid searched on the training split only.
HP_GRID = {
    "n_reservoir": [32, 64],
    "spectral_radius": [0.8, 0.95],
    "leak": [0.2, 0.5],
    "input_scale": [0.5, 1.0],
    "ridge": [1e-4, 1e-2],
}


def collect_demos(scenarios, layouts: dict[str, StreetLayout]):
    """Behavior-cloning demos: roll the waypoint teacher and keep the (features,
    targets) sequence for every scenario it solves (clean demonstrations)."""
    feats, targs = [], []
    for scenario in scenarios:
        layout = layouts[scenario.layout]
        rec = TrajectoryRecorder(WaypointController(scenario, layout))
        rec.reset()
        if run_street_episode(scenario, layout, rec).outcome == "arrival":
            feats.append(np.asarray(rec.features))
            targs.append(np.asarray(rec.targets))
    return feats, targs


def _train_val_split(scenarios, seed: int, val_fraction: float):
    """Deterministic scenario-level split of the training file."""
    order = list(scenarios)
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    return order[n_val:], order[:n_val]


def _fit(recurrent: bool, hp: dict, feats, targs) -> EchoStateNetwork:
    esn = EchoStateNetwork(
        n_reservoir=hp["n_reservoir"], spectral_radius=hp["spectral_radius"],
        leak=hp["leak"], input_scale=hp["input_scale"], seed=ESN_SEED,
        recurrent=recurrent)
    return esn.fit(feats, targs, ridge=hp["ridge"])


def _closed_loop_breakdown(esn: EchoStateNetwork, scenarios,
                           layouts: dict[str, StreetLayout]) -> dict:
    make = lambda s, l: (ctrl := RecurrentController(esn), ctrl.reset)
    return _breakdown(_run_per_scenario(scenarios, layouts, make), layouts)


def select_hyperparameters(train_scenarios, layouts) -> tuple[dict, list]:
    """Grid search on the training split only (fit on a sub-split, select by
    closed-loop arrival on the held-out validation sub-split)."""
    fit_scenarios, val_scenarios = _train_val_split(
        train_scenarios, VAL_SEED, VAL_FRACTION)
    fit_feats, fit_targs = collect_demos(fit_scenarios, layouts)
    keys = list(HP_GRID)
    trials = []
    best = None
    for combo in itertools.product(*(HP_GRID[k] for k in keys)):
        hp = dict(zip(keys, combo))
        esn = _fit(True, hp, fit_feats, fit_targs)
        breakdown = _closed_loop_breakdown(esn, val_scenarios, layouts)
        _, overall, per_layout = _gate(breakdown, layouts)
        min_layer = min(per_layout.values())
        trials.append({"hp": hp, "val_overall": overall,
                       "val_by_layout": per_layout})
        key = (overall, min_layer)
        if best is None or key > best[0]:
            best = (key, hp)
        print(f"  val overall={overall:.3f} min_layer={min_layer:.3f} {hp}",
              flush=True)
    trials.sort(key=lambda t: (t["val_overall"], min(t["val_by_layout"].values())),
               reverse=True)
    return best[1], trials


def _one_shot_gate(recurrent: bool, hp: dict, train_feats, train_targs,
                   gate_scenarios, layouts) -> dict:
    esn = _fit(recurrent, hp, train_feats, train_targs)
    breakdown = _closed_loop_breakdown(esn, gate_scenarios, layouts)
    passes, overall, per_layout = _gate(breakdown, layouts)
    return {"passes_gate": passes, "overall_arrival_rate": overall,
            "by_layout_arrival_rate": per_layout, "breakdown": breakdown}


def _interpretation(recurrent: dict, ablation: dict) -> str:
    r = recurrent["overall_arrival_rate"]
    a = ablation["overall_arrival_rate"]
    lead = f"Recurrent {r:.3f} vs memoryless ablation {a:.3f} (references: " \
           f"waypoint witness {REF_WAYPOINT_GATE:.2f}, Stage 2b state machine " \
           f"{REF_STATE_MACHINE_GATE:.2f})."
    if recurrent["passes_gate"] and r > a + 0.02:
        return (lead + " The recurrent baseline clears the gate and beats its "
                "matched memoryless ablation: temporal memory over the same "
                "observation interface -- not the interface itself -- accounts "
                "for the gap. This motivates the biologically grounded "
                "memory/action-selection experiment (dopamine as "
                "modulation/teaching, never goal bearing).")
    if recurrent["passes_gate"]:
        return (lead + " The recurrent baseline clears the gate but does not "
                "clearly beat its memoryless ablation, so the gain is not "
                "attributable to temporal state alone; diagnose before "
                "attributing to memory.")
    return (lead + " The recurrent baseline does not clear the gate: temporal "
            "memory over this interface, trained by imitation, is not "
            "sufficient here. Diagnose capacity, imitation drift, and "
            "observation ambiguity before any dopamine/connectome learning.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2c recurrent baseline.")
    parser.add_argument("--train-split", default="runs/stage2b/sm_train_split.json")
    parser.add_argument("--gate-split", default="runs/stage2b/gate_split.json")
    parser.add_argument("--out", default="runs/stage2c/results.json")
    parser.add_argument("--gate-out", default="runs/stage2c/gate_results.json")
    args = parser.parse_args(argv)

    layouts = initial_layouts()
    train = load_scenarios(args.train_split)
    gate = load_scenarios(args.gate_split)
    print(f"Stage 2c: selecting hyperparameters on {len(train)} training "
          f"scenarios (gate {len(gate)} untouched until the one-shot).")

    best_hp, trials = select_hyperparameters(train, layouts)
    print(f"Best training config: {best_hp}")

    # Refit on ALL training demos, then evaluate the gate ONCE (recurrent + ablation).
    train_feats, train_targs = collect_demos(train, layouts)
    print(f"Refit on {len(train_feats)} teacher demos; one-shot gate evaluation.")
    recurrent = _one_shot_gate(True, best_hp, train_feats, train_targs, gate, layouts)
    ablation = _one_shot_gate(False, best_hp, train_feats, train_targs, gate, layouts)

    interpretation = _interpretation(recurrent, ablation)
    gate_payload = {
        "train_split": args.train_split,
        "train_split_sha256": sha256(args.train_split),
        "gate_split": args.gate_split,
        "gate_split_sha256": sha256(args.gate_split),
        "gate": {"overall_rate": GATE_OVERALL_RATE,
                 "per_layout_rate": GATE_PER_LAYOUT_RATE},
        "hyperparameters": best_hp,
        "esn_seed": ESN_SEED,
        "recurrent": recurrent,
        "memoryless_ablation": ablation,
        "references": {"waypoint_witness": REF_WAYPOINT_GATE,
                       "state_machine": REF_STATE_MACHINE_GATE},
        "interpretation": interpretation,
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate_payload, indent=2) + "\n")
    results_payload = dict(gate_payload)
    results_payload["hyperparameter_trials"] = trials
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
