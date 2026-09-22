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
