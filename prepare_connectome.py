"""Convert downloaded MaleCNS v1.0 tables into a prepared sparse connectome.

Inputs (already downloaded into --data, CC-BY 4.0, release male-cns:v1.0):
  body-annotations.feather      bodyId, type, somaSide, superclass, ...
  body-neurotransmitters.feather body, predicted_nt, predicted_nt_confidence, ...
  connectome-weights.feather     body_pre, body_post, weight   (~151.8M edges)

Outputs (into --data), written atomically (temp file + rename):
  counts.npz          CSR anatomical synapse counts (postsyn rows, presyn cols)
  weights_signed.npz  CSR signed fast weights (acetylcholine +, gaba/glut -, else 0)
  ids.npy             int64 bodyId per contiguous matrix index
  manifest.json       provenance: URLs, SHA-256, license, columns, filters, sign
                      policy counts, candidate interface map (populations only)

Inclusion rule: annotated bodies whose superclass does NOT start with "vnc"
(central brain + optic lobe kept; ventral nerve cord excluded). Unlabeled
(None) superclass bodies are kept and counted.

Interface mapping: this script records the SUPPORTED parts only — candidate
output populations (PFL3 split L/R by somaSide) and candidate input populations
(EPG heading, FC2 goal) by type. It does NOT assign preferred angles: the male
annotations expose no clean central-complex column/phase field, and the plan
forbids assigning angles by sorted neuron ID. That gap is recorded, not faked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

RELEASE = "male-cns:v1.0"
LICENSE = "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
BASE_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome")
SOURCES = {
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        f"{BASE_URL}/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
    "transmitters": (
        "body-neurotransmitters-male-cns-v1.0.feather",
        f"{BASE_URL}/body-neurotransmitters-male-cns-v1.0.feather"),
    "weights": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        f"{BASE_URL}/connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
}

NT_CONF_MIN = 0.5
POSITIVE_NT = {"acetylcholine"}
NEGATIVE_NT = {"gaba", "glutamate"}  # glutamate sign is an approximation

# Candidate central-complex navigation populations (types), grounded in fly
# navigation literature. Roles are candidates pending Phase 3 verification.
INPUT_TYPES = {"heading": ["EPG"], "goal": ["FC2A", "FC2B", "FC2C"]}
OUTPUT_TYPE = "PFL3"  # split L/R by somaSide

# Anatomical position parsed from the `instance` string:
#   EPG(PB08)_R2  -> protocerebral-bridge glomerulus index 2  (heading)
#   FC2B_C2_L     -> fan-shaped-body column index 2           (goal)
_PB_GLOM = re.compile(r"_[LR](\d+)")     # PB glomerulus index
_FB_COL = re.compile(r"_C(\d+)")          # FB column index

# Corrected coordinate convention (published PB/EB geometry): PB halves map in
# opposite directions around the EB, with medial EPG wedges separated by 22.5
# degrees. FB goals retain the uniform nine-column assumption; the absolute
# world origin remains free. This is not a per-neuron measured preference.
ANGLE_CONVENTION = (
    "heading: L_i=(i-1)*pi/4, R_i=-pi/8-(i-1)*pi/4; goal: "
    "2*pi*(FB_column-1)/9; ccw-positive, absolute world origin arbitrary."
)
ANGLE_CONFIDENCE = "moderate (published PB/EB mapping; nine-column FB assumption)"
ANGLE_REFERENCES = [
    "MaleCNS v1.0 annotations (PB glomerulus / FB column in instance names)",
    "Heading/goal FC2/PFL3 circuit: https://www.nature.com/articles/s41586-023-07006-3",
    "PB-to-EB projection geometry: https://pmc.ncbi.nlm.nih.gov/articles/PMC4407839/",
]


def epg_preferred_angles(instances: list[str]) -> list[float]:
    """Return side-aware EPG angles from PB glomerulus instance names."""
    angles = []
    for instance in instances:
        match = re.search(r"_([LR])(\d+)$", instance)
        if match is None:
            raise ValueError(f"cannot parse EPG instance {instance!r}")
        side, raw_index = match.groups()
        step = (int(raw_index) - 1) * math.pi / 4.0
        angles.append(step if side == "L" else -math.pi / 8.0 - step)
    return angles


def _indices_from_instances(instances: list[str | None], pattern: re.Pattern):
    """Parse one integer anatomical index per neuron; None if absent."""
    out = []
    for s in instances:
        m = pattern.search(s) if s else None
        out.append(int(m.group(1)) if m else None)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transmitter_sign(nt: str | None, conf: float | None) -> int:
    if nt is None or conf is None or conf < NT_CONF_MIN:
        return 0
    if nt in POSITIVE_NT:
        return 1
    if nt in NEGATIVE_NT:
        return -1
    return 0  # dopamine/serotonin/histamine/octopamine/unclear -> modulatory/uncertain


def _save_npz_atomic(path: Path, matrix) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:      # file handle -> scipy does not append .npz
        sp.save_npz(f, matrix)
    os.replace(tmp, path)


def _save_npy_atomic(path: Path, arr) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:      # file handle -> numpy does not append .npy
        np.save(f, arr)
    os.replace(tmp, path)


def main(data_dir: str) -> None:
    data = Path(data_dir)
    ann_path = data / SOURCES["annotations"][0]
    nt_path = data / SOURCES["transmitters"][0]
    w_path = data / SOURCES["weights"][0]
    for p in (ann_path, nt_path, w_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; download it first")

    # --- Retained neuron set + contiguous index ---
    ann = feather.read_table(ann_path,
                             columns=["bodyId", "type", "instance", "somaSide", "superclass"])
    body = np.asarray(ann.column("bodyId").to_pylist(), dtype=np.int64)
    superclass = ann.column("superclass").to_pylist()
    types = ann.column("type").to_pylist()
    instances = ann.column("instance").to_pylist()
    sides = ann.column("somaSide").to_pylist()

    keep = np.array([sc is None or not sc.startswith("vnc") for sc in superclass])
    keep_pos = np.where(keep)[0]
    ids_unsorted = body[keep]
    order = np.argsort(ids_unsorted, kind="stable")  # sort stable IDs before indexing
    ids = ids_unsorted[order]
    kept_types = [types[keep_pos[i]] for i in order]
    kept_instances = [instances[keep_pos[i]] for i in order]
    kept_sides = [sides[keep_pos[i]] for i in order]
    n = len(ids)
    id_to_idx = {int(b): i for i, b in enumerate(ids)}
    excluded = int((~keep).sum())

    # --- Per-neuron transmitter sign (indexed by matrix position) ---
    nt = feather.read_table(nt_path, columns=["body", "predicted_nt", "predicted_nt_confidence"])
    nt_body = nt.column("body").to_pylist()
    nt_pred = nt.column("predicted_nt").to_pylist()
    nt_conf = nt.column("predicted_nt_confidence").to_pylist()
    sign = np.zeros(n, dtype=np.int8)
    have_nt = np.zeros(n, dtype=bool)
    for b, pred, conf in zip(nt_body, nt_pred, nt_conf):
        i = id_to_idx.get(int(b))
        if i is not None:
            sign[i] = transmitter_sign(pred, conf)
            have_nt[i] = True
    sign_pos = int((sign == 1).sum())
    sign_neg = int((sign == -1).sum())
    sign_zero = int((sign == 0).sum())

    # --- Edges: filter to retained endpoints, map to indices ---
    w = feather.read_table(w_path, columns=["body_pre", "body_post", "weight"])
    pre = w.column("body_pre").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    post = w.column("body_post").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    wt = w.column("weight").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32)
    del w

    # Vectorized bodyId -> index via searchsorted on the sorted retained ids.
    def to_idx(arr):
        pos = np.searchsorted(ids, arr)
        pos_clip = np.clip(pos, 0, n - 1)
        valid = ids[pos_clip] == arr
        return pos_clip, valid

    total_edges = int(pre.size)
    total_weight = float(wt.sum())
    pre_i, pre_ok = to_idx(pre)
    post_i, post_ok = to_idx(post)
    edge_ok = pre_ok & post_ok
    dropped_edges = int((~edge_ok).sum())
    pre_i, post_i, wt = pre_i[edge_ok], post_i[edge_ok], wt[edge_ok]
    del pre, post, pre_ok, post_ok, edge_ok
    kept_weight_fraction = float(wt.sum()) / total_weight if total_weight else 0.0

    # Anatomical counts: postsyn rows, presyn cols. COO->CSR sums duplicate pairs.
    counts = sp.coo_array((wt, (post_i, pre_i)), shape=(n, n)).tocsr()

    # Signed fast weights: edge sign follows the presynaptic transmitter (Dale).
    edge_sign = sign[pre_i].astype(np.float32)
    weights_signed = sp.coo_array((wt * edge_sign, (post_i, pre_i)), shape=(n, n)).tocsr()
    weights_signed.eliminate_zeros()
    zeroed_edges = int((edge_sign == 0).sum())
    zeroed_weight = float(wt[edge_sign == 0].sum())

    # --- Candidate interface map with anatomy-derived preferred angles ---
    type_arr = np.array(kept_types, dtype=object)
    side_arr = np.array(kept_sides, dtype=object)

    def build_input(types_list, pattern):
        """Return (indices, preferred_angles) for neurons whose instance carries
        an anatomical index, angles by the documented uniform-tiling convention."""
        idx, raw = [], []
        for t in types_list:
            for i in np.where(type_arr == t)[0]:
                anat = _indices_from_instances([kept_instances[i]], pattern)[0]
                if anat is not None:
                    idx.append(int(i))
                    raw.append(anat)
        if not idx:
            return [], []
        n_idx = max(raw)  # observed index range sets the tiling resolution
        angles = [float(2.0 * math.pi * (r - 1) / n_idx) for r in raw]
        # Sort by index for stable output; keep angle alignment.
        order2 = sorted(range(len(idx)), key=lambda k: idx[k])
        return [idx[k] for k in order2], [angles[k] for k in order2]

    heading_idx, heading_ang = build_input(INPUT_TYPES["heading"], _PB_GLOM)
    heading_ang = epg_preferred_angles([kept_instances[i] for i in heading_idx])
    goal_idx, goal_ang = build_input(INPUT_TYPES["goal"], _FB_COL)
    pfl3 = np.where(type_arr == OUTPUT_TYPE)[0]
    left_idx = sorted(int(i) for i in pfl3 if side_arr[i] == "L")
    right_idx = sorted(int(i) for i in pfl3 if side_arr[i] == "R")

    interface = {
        "status": "anatomy-derived preferred angles (documented convention)",
        "angle_convention": ANGLE_CONVENTION,
        "angle_confidence": ANGLE_CONFIDENCE,
        "angle_references": ANGLE_REFERENCES,
        "heading_input": {"types": INPUT_TYPES["heading"], "source": "PB glomerulus",
                          "indices": heading_idx, "preferred_angles": heading_ang},
        "goal_input": {"types": INPUT_TYPES["goal"], "source": "FB column",
                      "indices": goal_idx, "preferred_angles": goal_ang},
        "left_output": {"type": OUTPUT_TYPE, "side": "L", "indices": left_idx},
        "right_output": {"type": OUTPUT_TYPE, "side": "R", "indices": right_idx},
        "output_direction_note": (
            "L/R by somaSide is a candidate; motor direction must be verified "
            "experimentally in Phase 3, not inferred from names."),
    }
    for grp in (heading_idx, goal_idx, left_idx, right_idx):
        assert all(0 <= i < n for i in grp)  # mapped indices belong to retained graph
    assert len(heading_idx) == len(heading_ang) and len(goal_idx) == len(goal_ang)

    # --- Save atomically ---
    _save_npz_atomic(data / "counts.npz", counts)
    _save_npz_atomic(data / "weights_signed.npz", weights_signed)
    _save_npy_atomic(data / "ids.npy", ids)

    manifest = {
        "release": RELEASE,
        "license": LICENSE,
        "sources": {k: {"url": v[1], "sha256": sha256(data / v[0])}
                    for k, v in SOURCES.items()},
        "selected_columns": {
            "annotations": ["bodyId", "type", "somaSide", "superclass"],
            "transmitters": ["body", "predicted_nt", "predicted_nt_confidence"],
            "weights": ["body_pre", "body_post", "weight"],
        },
        "inclusion_rule": "superclass does not start with 'vnc' (brain + optic lobe)",
        "counts": {
            "neurons_retained": n,
            "neurons_excluded_vnc": excluded,
            "neurons_with_nt_prediction": int(have_nt.sum()),
            "edges_retained": int(counts.nnz),
            "edges_total_in_source": total_edges,
            "edges_dropped_unknown_endpoint": dropped_edges,
            "anatomical_weight_kept_fraction": kept_weight_fraction,
        },
        "sign_policy": {
            "confidence_min": NT_CONF_MIN,
            "positive_nt": sorted(POSITIVE_NT),
            "negative_nt": sorted(NEGATIVE_NT),
            "note": "glutamate sign is an approximation; unclear/modulatory -> 0",
            "neurons_positive": sign_pos,
            "neurons_negative": sign_neg,
            "neurons_zero_fast_weight": sign_zero,
            "edges_zeroed": zeroed_edges,
            "anatomical_weight_zeroed": zeroed_weight,
        },
        "interface_map": interface,
        "matrix_orientation": "postsynaptic rows, presynaptic columns (W @ activity)",
    }
    tmp = data / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, data / "manifest.json")

    print(json.dumps({"neurons": n, "edges": int(counts.nnz),
                      "dropped_edges": dropped_edges,
                      "sign_pos": sign_pos, "sign_neg": sign_neg,
                      "sign_zero": sign_zero,
                      "heading_n": len(heading_idx), "goal_n": len(goal_idx),
                      "left_n": len(left_idx), "right_n": len(right_idx)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare MaleCNS connectome.")
    p.add_argument("--data", default="data/malecns-v1.0")
    main(p.parse_args().data)
