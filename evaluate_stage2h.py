# evaluate_stage2h.py
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from evaluate_stage2 import (_scenario_key, _waypoints, load_scenarios,
                             sha256, verify_or_save_scenarios)
from evaluate_stage2g import ALL_PRIOR_SPLITS
from stage2b import WaypointController
from street import (MAX_SIM_TIME, StreetCarState, StreetScenario,
                    initial_layouts, run_street_episode)
from stage2h import (PPO_KWARGS, PPO_SEEDS, TOTAL_TIMESTEPS, gate_score,
                     score_model, train_model)
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

HEADINGS = tuple(i * math.pi / 16 for i in range(-15, 17))
PRIOR_SPLITS = list(dict.fromkeys(ALL_PRIOR_SPLITS + [
    "runs/stage2g/train_split.json",
    "runs/stage2g/dev_split.json",
    "runs/stage2g/gate_split.json",
]))
TRAIN_COUNTS = {name: 150 for name in ("cross", "regular", "asymmetric")}
READINESS_COUNTS = {name: 100 for name in TRAIN_COUNTS}
GATE_COUNTS = dict(READINESS_COUNTS)


def _load(paths):
    return [scenario for path in paths for scenario in load_scenarios(path)]


def build_split(seed, counts, extra_exclude):
    rng = random.Random(seed)
    layouts = initial_layouts()
    excluded = {_scenario_key(s) for s in _load(PRIOR_SPLITS) + list(extra_exclude)}
    split = []
    for name, layout in layouts.items():
        points = _waypoints(layout)
        candidates = [(start, target, heading)
                      for start in points for target in points
                      if start != target and math.dist(start, target) >= 12.0
                      for heading in HEADINGS
                      if (name, *start, heading, *target) not in excluded]
        rng.shuffle(candidates)
        chosen = []
        for start, target, heading in candidates:
            scenario = StreetScenario(
                name, StreetCarState(*start, heading, 0.0), *target,
                timeout=MAX_SIM_TIME,
                label=f"stage2h-{seed}-{name}-{len(chosen):03d}")
            controller = WaypointController(scenario, layout)
            if run_street_episode(scenario, layout, controller).outcome == "arrival":
                chosen.append(scenario)
                if len(chosen) == counts[name]:
                    break
        if len(chosen) != counts[name]:
            raise SystemExit(
                f"{name}: only {len(chosen)} fresh witnessed scenarios; need {counts[name]}")
        split.extend(chosen)
    return split


def build_all_splits():
    train = build_split(80, TRAIN_COUNTS, [])
    readiness = build_split(81, READINESS_COUNTS, train)
    gate = build_split(82, GATE_COUNTS, train + readiness)
    return train, readiness, gate


def freeze_splits():
    train, readiness, gate = build_all_splits()
    paths = {
        "runs/stage2h/train_split.json": train,
        "runs/stage2h/readiness_split.json": readiness,
        "runs/stage2h/gate_split.json": gate,
    }
    for path, split in paths.items():
        verify_or_save_scenarios(path, split)
    provenance = {"splits": {path: sha256(path) for path in paths},
                  "excluded": {path: sha256(path) for path in PRIOR_SPLITS}}
    Path("runs/stage2h/split_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return provenance


SPENT_SELECTION_SPLITS = [
    "runs/stage2f/gate_split.json",
    "runs/stage2g/gate_split.json",
]
RESULTS_PATH = Path("runs/stage2h/results.json")
GATE_RESULTS_PATH = Path("runs/stage2h/gate_results.json")


def select_seed(rows):
    return max(rows, key=lambda row: (row["selection_score"], -row["seed"]))


def require_gate_open(results_path=RESULTS_PATH, gate_path=GATE_RESULTS_PATH):
    payload = json.loads(Path(results_path).read_text())
    if not payload.get("readiness", {}).get("recurrent", {}).get("passes_gate"):
        raise SystemExit("Stage 2h readiness did not pass; gate remains sealed")
    if Path(gate_path).exists():
        raise SystemExit(f"{gate_path} already exists; Stage 2h gate is spent")
    return payload


def train_all():
    if RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} already exists; refuse to replace a completed run")
    Path("runs/stage2h/models").mkdir(parents=True, exist_ok=True)
    train = load_scenarios("runs/stage2h/train_split.json")
    selection = [load_scenarios(path) for path in SPENT_SELECTION_SPLITS]
    from importlib.metadata import version
    payload = {
        "dependencies": {name: version(name) for name in (
            "torch", "gymnasium", "stable-baselines3", "sb3-contrib")},
        "budget": {"timesteps": TOTAL_TIMESTEPS, "seeds": list(PPO_SEEDS),
                   "ppo_kwargs": PPO_KWARGS},
        "split_sha256": {path: sha256(path) for path in (
            "runs/stage2h/train_split.json",
            "runs/stage2h/readiness_split.json",
            "runs/stage2h/gate_split.json")},
        "arms": {},
    }
    for recurrent, name in ((True, "recurrent"), (False, "feedforward")):
        rows = []
        for seed in PPO_SEEDS:
            path = f"runs/stage2h/models/{name}-seed{seed}"
            model = train_model(recurrent, train, seed, path)
            scores = [score_model(model, split) for split in selection]
            model_path = path + ".zip"
            rows.append({"seed": seed, "model_path": model_path,
                         "model_sha256": sha256(model_path),
                         "spent_gate_scores": scores,
                         "selection_score": min(gate_score(s) for s in scores)})
        payload["arms"][name] = {"runs": rows, "selected": select_seed(rows)}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def run_readiness():
    payload = json.loads(RESULTS_PATH.read_text())
    if "readiness" in payload:
        raise SystemExit("Stage 2h readiness already spent")
    scenarios = load_scenarios("runs/stage2h/readiness_split.json")
    payload["readiness"] = {}
    for recurrent, name, cls in (
        (True, "recurrent", RecurrentPPO), (False, "feedforward", PPO)):
        model = cls.load(payload["arms"][name]["selected"]["model_path"])
        payload["readiness"][name] = score_model(model, scenarios)
    payload["gate_open"] = payload["readiness"]["recurrent"]["passes_gate"]
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def run_gate():
    payload = require_gate_open()
    scenarios = load_scenarios("runs/stage2h/gate_split.json")
    scores = {}
    for name, cls in (("recurrent", RecurrentPPO), ("feedforward", PPO)):
        selected = payload["arms"][name]["selected"]
        model = cls.load(selected["model_path"])
        scores[name] = score_model(model, scenarios)
    GATE_RESULTS_PATH.write_text(json.dumps({
        "gate_split": "runs/stage2h/gate_split.json",
        "gate_split_sha256": sha256("runs/stage2h/gate_split.json"),
        "selected_seeds": {name: payload["arms"][name]["selected"]["seed"]
                           for name in scores},
        "selected_model_sha256": {
            name: payload["arms"][name]["selected"]["model_sha256"]
            for name in scores},
        "results": scores,
        "interpretation": interpret(scores),
    }, indent=2) + "\n")


def interpret(scores):
    recurrent = scores["recurrent"]
    feedforward = scores["feedforward"]
    if recurrent["passes_gate"] and not feedforward["passes_gate"]:
        return ("Recurrent PPO passed while feed-forward PPO failed on the same "
                "observations and gate. Within this PPO family, learned temporal "
                "state supplies missing action-selection; proceed to the "
                "biologically grounded memory/learning experiment.")
    if recurrent["passes_gate"] and feedforward["passes_gate"]:
        return ("Both PPO arms passed. The observation interface supports the task, "
                "but recurrence is not necessary within this PPO family.")
    return ("The selected recurrent PPO did not pass the fresh gate. This remains "
            "confounded by policy, optimization, and generalization and does not "
            "implicate the observation interface; do not advance to dopamine or "
            "connectome learning.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2h PPO experiment")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-splits", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--readiness", action="store_true")
    modes.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_splits:
        freeze_splits()
    elif args.train:
        train_all()
    elif args.readiness:
        run_readiness()
    else:
        run_gate()


if __name__ == "__main__":
    main()
