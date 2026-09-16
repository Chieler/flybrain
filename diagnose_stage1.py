"""Localize Stage 1's PFL3 steering failure without running car episodes.

For a fixed heading/error grid, partition each PFL3 neuron's recurrent drive
into heading (EPG/Delta7), goal (FC2), and all-other presynaptic sources. Compare
the central-complex-only drive, full next-step drive, and settled activity.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from brain import Brain
from prepare_connectome import SOURCES, epg_preferred_angles, sha256


HEADINGS = np.linspace(-math.pi, math.pi, 8, endpoint=False)
ERRORS = (-2.4, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.4)
PHASE_ANGLES = np.linspace(-math.pi, math.pi, 16, endpoint=False)


def _apply_heading_convention(
    brain: Brain, data: Path, ids: np.ndarray, convention: str
) -> None:
    if convention == "current":
        return
    if convention != "mirrored_epg":
        raise ValueError(f"unknown heading convention {convention!r}")
    table = feather.read_table(
        data / SOURCES["annotations"][0], columns=["bodyId", "type", "instance"]
    )
    instances = {
        row["bodyId"]: row["instance"]
        for row in table.to_pylist()
        if row["type"] == "EPG"
    }
    indices = brain.input_map["heading"][0]
    brain.input_map["heading"] = (
        indices,
        np.asarray(
            epg_preferred_angles([instances[int(ids[index])] for index in indices]),
            dtype=np.float32,
        ),
    )


def fit_first_harmonic(angles: np.ndarray, responses: np.ndarray) -> dict[str, np.ndarray]:
    """Fit offset + amplitude*cos(angle - phase) independently per column."""
    angles = np.asarray(angles, dtype=float)
    responses = np.asarray(responses, dtype=float)
    design = np.column_stack([np.cos(angles), np.sin(angles), np.ones(len(angles))])
    coefficients = np.linalg.lstsq(design, responses, rcond=None)[0]
    predicted = design @ coefficients
    residual = np.sum((responses - predicted) ** 2, axis=0)
    total = np.sum((responses - responses.mean(axis=0)) ** 2, axis=0)
    r2 = np.ones_like(total)
    np.divide(residual, total, out=r2, where=total > 0.0)
    r2 = np.where(total > 0.0, 1.0 - r2, 1.0)
    return {
        "offset": coefficients[2],
        "amplitude": np.hypot(coefficients[0], coefficients[1]),
        "phase": np.arctan2(coefficients[1], coefficients[0]),
        "r2": r2,
    }


def _circular_summary(angles: np.ndarray, weights: np.ndarray | None = None) -> dict:
    angles = np.asarray(angles, dtype=float)
    weights = np.ones_like(angles) if weights is None else np.asarray(weights, dtype=float)
    vector = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
    return {"offset": float(np.angle(vector)), "concentration": float(abs(vector))}


def best_circular_alignment(
    measured: np.ndarray, anatomical: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """Fit global phase and handedness between anatomical and measured phases."""
    candidates = []
    for handedness in (1, -1):
        summary = _circular_summary(measured - handedness * anatomical, weights)
        candidates.append({"handedness": handedness, **summary})
    return max(candidates, key=lambda item: item["concentration"])


def recurrent_components(
    brain: Brain, targets: np.ndarray, source_groups: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Partition normalized recurrent drive onto `targets` by source group."""
    assigned = np.zeros(brain.n, dtype=bool)
    rows = brain.W[np.asarray(targets, dtype=int)]
    parts: dict[str, np.ndarray] = {}
    for name, raw_indices in source_groups.items():
        indices = np.asarray(raw_indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError(f"{name} contains an out-of-range neuron index")
        if np.any(assigned[indices]):
            raise ValueError(f"{name} overlaps an earlier source group")
        assigned[indices] = True
        parts[name] = brain.params.recurrent_gain * np.asarray(
            rows[:, indices] @ brain.activity[indices]
        ).ravel()
    other = np.flatnonzero(~assigned)
    parts["other"] = brain.params.recurrent_gain * np.asarray(
        rows[:, other] @ brain.activity[other]
    ).ravel()
    return parts


def _indices_by_type(data: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    table = feather.read_table(
        data / SOURCES["annotations"][0], columns=["bodyId", "type"]
    )
    positions: dict[str, list[int]] = {}
    for body_id, neuron_type in zip(table["bodyId"], table["type"]):
        if neuron_type.as_py() is None:
            continue
        pos = int(np.searchsorted(ids, body_id.as_py()))
        if pos < len(ids) and ids[pos] == body_id.as_py():
            positions.setdefault(neuron_type.as_py(), []).append(pos)
    return {name: np.asarray(value, dtype=int) for name, value in positions.items()}


def _pfl3_anatomy(data: Path, ids: np.ndarray, indices: np.ndarray) -> list[dict]:
    table = feather.read_table(
        data / SOURCES["annotations"][0],
        columns=["bodyId", "type", "instance", "somaSide"],
    )
    rows = {
        row["bodyId"]: row
        for row in table.to_pylist()
        if row["type"] == "PFL3"
    }
    result = []
    for index in indices:
        body_id = int(ids[index])
        row = rows[body_id]
        match = re.search(r"_([LR])(\d+)_C(\d+)", row["instance"])
        if match is None:
            raise ValueError(f"cannot parse PFL3 instance {row['instance']!r}")
        result.append(
            {
                "index": int(index),
                "body_id": body_id,
                "instance": row["instance"],
                "soma_side": row["somaSide"],
                "pb_side": match.group(1),
                "pb_index": int(match.group(2)),
                "fb_column": int(match.group(3)),
            }
        )
    return result


def _pooled(values: np.ndarray, n_left: int) -> float:
    return float(values[:n_left].mean() - values[n_left:].mean())


def _sign_correct(signal: float, error: float) -> bool:
    return (signal > 0.0) == (error > 0.0)


def phase_audit(
    brain: Brain,
    pfl3: np.ndarray,
    groups: dict[str, np.ndarray],
    anatomy: list[dict],
    settle_steps: int,
) -> dict:
    """Measure effective heading/goal phases delivered directly onto PFL3."""
    heading_responses = []
    goal_responses = []
    heading_inputs = brain.input_map["heading"][0]
    goal_inputs = brain.input_map["goal"][0]
    for angle in PHASE_ANGLES:
        brain.reset()
        stimulus = brain.encode(float(angle), 0.0)
        stimulus[goal_inputs] = 0.0
        for _ in range(settle_steps):
            brain.step(stimulus)
        parts = recurrent_components(brain, pfl3, groups)
        heading_responses.append(parts["epg"] + parts["delta7"])

        brain.reset()
        stimulus = brain.encode(0.0, float(angle))
        stimulus[heading_inputs] = 0.0
        for _ in range(settle_steps):
            brain.step(stimulus)
        parts = recurrent_components(brain, pfl3, groups)
        goal_responses.append(parts["goal"])

    heading_fit = fit_first_harmonic(PHASE_ANGLES, np.stack(heading_responses))
    goal_fit = fit_first_harmonic(PHASE_ANGLES, np.stack(goal_responses))
    fb_angles = 2.0 * math.pi * (np.array([x["fb_column"] for x in anatomy]) - 1) / 9.0
    pb_angles = 2.0 * math.pi * (np.array([x["pb_index"] for x in anatomy]) - 1) / 8.0
    heading_weights = heading_fit["amplitude"] * np.clip(heading_fit["r2"], 0.0, 1.0)
    goal_weights = goal_fit["amplitude"] * np.clip(goal_fit["r2"], 0.0, 1.0)
    phase_delta = np.angle(np.exp(1j * (goal_fit["phase"] - heading_fit["phase"])))
    side_offsets = {}
    heading_alignment = {}
    for side in ("L", "R"):
        mask = np.array([x["soma_side"] == side for x in anatomy])
        side_offsets[side] = _circular_summary(
            phase_delta[mask], np.minimum(heading_weights[mask], goal_weights[mask])
        )
        heading_alignment[side] = best_circular_alignment(
            heading_fit["phase"][mask], pb_angles[mask], heading_weights[mask]
        )

    per_neuron = []
    for i, entry in enumerate(anatomy):
        per_neuron.append(
            {
                **entry,
                "heading": {name: float(values[i]) for name, values in heading_fit.items()},
                "goal": {name: float(values[i]) for name, values in goal_fit.items()},
                "goal_minus_heading_phase": float(phase_delta[i]),
            }
        )
    return {
        "angles": PHASE_ANGLES.tolist(),
        "heading_fit": {
            "mean_amplitude": float(heading_fit["amplitude"].mean()),
            "mean_r2": float(heading_fit["r2"].mean()),
            "median_r2": float(np.median(heading_fit["r2"])),
            "pb_alignment_by_soma_side": heading_alignment,
        },
        "goal_fit": {
            "mean_amplitude": float(goal_fit["amplitude"].mean()),
            "mean_r2": float(goal_fit["r2"].mean()),
            "median_r2": float(np.median(goal_fit["r2"])),
            "fb_column_alignment": best_circular_alignment(
                goal_fit["phase"], fb_angles, goal_weights
            ),
        },
        "goal_minus_heading_by_soma_side": side_offsets,
        "neurons": per_neuron,
    }


def run_diagnostic(
    data_dir: str, settle_steps: int = 30, heading_convention: str = "current"
) -> dict:
    data = Path(data_dir)
    brain = Brain.load(data)
    ids = np.load(data / "ids.npy")
    _apply_heading_convention(brain, data, ids, heading_convention)
    by_type = _indices_by_type(data, ids)
    left = np.asarray(brain.output_map["left"], dtype=int)
    right = np.asarray(brain.output_map["right"], dtype=int)
    pfl3 = np.concatenate([left, right])
    groups = {
        "epg": by_type.get("EPG", np.array([], dtype=int)),
        "delta7": by_type.get("Delta7", np.array([], dtype=int)),
        "goal": np.concatenate(
            [by_type.get(name, np.array([], dtype=int))
             for name in ("FC2A", "FC2B", "FC2C")]
        ),
    }
    anatomy = _pfl3_anatomy(data, ids, pfl3)
    stages = {name: 0 for name in ("central_drive", "full_drive", "settled")}
    magnitude = {name: [] for name in ("epg", "delta7", "goal", "other")}
    regimes = {
        name: {"low": 0, "linear": 0, "high": 0}
        for name in ("central_drive", "full_drive")
    }
    interactions: list[float] = []
    cells = []
    baseline = brain.params.baseline

    for heading in HEADINGS:
        for error in ERRORS:
            goal = (heading + error + math.pi) % (2.0 * math.pi) - math.pi
            brain.reset()
            stimulus = brain.encode(float(heading), float(goal))
            for _ in range(settle_steps):
                brain.step(stimulus)

            parts = recurrent_components(brain, pfl3, groups)
            heading_drive = parts["epg"] + parts["delta7"]
            central_pre = baseline + heading_drive + parts["goal"]
            full_pre = central_pre + parts["other"]
            central_rate = np.clip(central_pre, 0.0, 1.0)
            full_rate = np.clip(full_pre, 0.0, 1.0)
            settled = brain.activity[pfl3].astype(float)
            signals = {
                "central_drive": _pooled(central_rate, len(left)),
                "full_drive": _pooled(full_rate, len(left)),
                "settled": _pooled(settled, len(left)),
            }
            for name, signal in signals.items():
                stages[name] += _sign_correct(signal, error)
            for name, values in parts.items():
                magnitude[name].append(float(np.mean(np.abs(values))))
            for name, values in (
                ("central_drive", central_pre), ("full_drive", full_pre)
            ):
                regimes[name]["low"] += int(np.count_nonzero(values <= 0.0))
                regimes[name]["linear"] += int(
                    np.count_nonzero((values > 0.0) & (values < 1.0))
                )
                regimes[name]["high"] += int(np.count_nonzero(values >= 1.0))

            base_rate = np.clip(baseline, 0.0, 1.0)
            interaction = (
                central_rate
                - np.clip(baseline + heading_drive, 0.0, 1.0)
                - np.clip(baseline + parts["goal"], 0.0, 1.0)
                + base_rate
            )
            interactions.append(float(np.mean(np.abs(interaction))))
            cells.append(
                {
                    "heading": float(heading),
                    "error": error,
                    "signals": signals,
                    "mean_abs_component": {
                        name: float(np.mean(np.abs(values)))
                        for name, values in parts.items()
                    },
                }
            )

    total_conditions = len(HEADINGS) * len(ERRORS)
    total_rates = total_conditions * len(pfl3)
    return {
        "experiment": "stage1_pfl3_upstream_diagnostic",
        "heading_convention": heading_convention,
        "data_dir": data_dir,
        "annotation_sha256": sha256(data / SOURCES["annotations"][0]),
        "settle_steps": settle_steps,
        "headings": HEADINGS.tolist(),
        "errors": list(ERRORS),
        "source_counts": {name: len(indices) for name, indices in groups.items()},
        "pfl3_counts": {"left": len(left), "right": len(right)},
        "sign_accuracy": {
            name: {"correct": correct, "total": total_conditions,
                   "fraction": correct / total_conditions}
            for name, correct in stages.items()
        },
        "mean_abs_recurrent_component": {
            name: float(np.mean(values)) for name, values in magnitude.items()
        },
        "activation_regime": {
            name: {key: {"count": count, "fraction": count / total_rates}
                   for key, count in counts.items()}
            for name, counts in regimes.items()
        },
        "mean_abs_heading_goal_nonlinear_interaction": float(np.mean(interactions)),
        "phase_audit": phase_audit(brain, pfl3, groups, anatomy, settle_steps),
        "conditions": cells,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument(
        "--heading-convention",
        choices=("current", "mirrored_epg"),
        default="current",
    )
    parser.add_argument("--out", default="runs/stage1/pfl3_diagnostic.json")
    args = parser.parse_args()
    result = run_diagnostic(args.data, args.settle_steps, args.heading_convention)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "sign_accuracy", "mean_abs_recurrent_component", "activation_regime",
        "mean_abs_heading_goal_nonlinear_interaction", "phase_audit")}, indent=2))
