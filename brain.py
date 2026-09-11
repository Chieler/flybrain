"""Bounded rate neural model over a sparse connectome.

Dynamics (per plan):
    drive    = clip(baseline + recurrent_gain * (W @ activity) + stimulus, 0, 1)
    activity += (neural_dt / tau) * (drive - activity)

W is postsynaptic-rows / presynaptic-columns so `W @ activity` propagates in the
recorded direction. Rows are normalized by total absolute incoming weight, with a
single global recurrent_gain retained.

This slice runs the mechanics on any provided sparse matrix (e.g. a tiny
synthetic graph in tests). Loading a real MaleCNS graph plus its biologically
grounded interface mapping is BLOCKED pending the dataset and mapping evidence
(see README). Preferred angles are NOT invented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class RateParams:
    neural_dt: float = 0.01   # s
    tau: float = 0.05         # s
    baseline: float = 0.01
    recurrent_gain: float = 1.0
    input_gain: float = 1.0


def normalize_rows(W) -> sp.csr_array:
    """Divide each nonempty row by its total absolute weight."""
    W = sp.csr_array(W, dtype=np.float32)
    abs_sums = np.asarray(np.abs(W).sum(axis=1)).ravel()
    scale = np.ones_like(abs_sums, dtype=np.float32)
    nz = abs_sums > 0
    scale[nz] = 1.0 / abs_sums[nz]
    return (sp.diags_array(scale) @ W).tocsr()


class Brain:
    """Fixed-weight recurrent rate network with role-tagged input/output pops.

    input_map: {role: (indices, preferred_angles)} for "heading" and "goal".
    output_map: {"left": indices, "right": indices}.
    Indices are contiguous array positions, not neuron IDs.
    """

    def __init__(self, W, input_map, output_map, params: RateParams | None = None):
        self.W = normalize_rows(W)
        self.n = self.W.shape[0]
        self.input_map = input_map
        self.output_map = output_map
        self.params = params or RateParams()
        self.activity = np.zeros(self.n, dtype=np.float32)

    def reset(self) -> None:
        self.activity = np.zeros(self.n, dtype=np.float32)

    def encode(self, heading: float, goal_bearing: float) -> np.ndarray:
        """Circular tuning: max(0, cos(angle - preferred)) * input_gain."""
        stim = np.zeros(self.n, dtype=np.float32)
        g = self.params.input_gain
        for role, angle in (("heading", heading), ("goal", goal_bearing)):
            entry = self.input_map.get(role)
            if entry is None:
                continue
            idx, preferred = entry
            stim[idx] = g * np.maximum(0.0, np.cos(angle - preferred)).astype(np.float32)
        return stim

    def step(self, stimulus: np.ndarray) -> None:
        p = self.params
        drive = np.clip(
            p.baseline + p.recurrent_gain * (self.W @ self.activity) + stimulus,
            0.0, 1.0,
        ).astype(np.float32)
        self.activity += np.float32(p.neural_dt / p.tau) * (drive - self.activity)

    def outputs(self) -> tuple[float, float]:
        """Pooled left/right output rates (population means)."""
        left = self.output_map["left"]
        right = self.output_map["right"]
        lm = float(self.activity[left].mean()) if len(left) else 0.0
        rm = float(self.activity[right].mean()) if len(right) else 0.0
        return lm, rm


@dataclass
class Adapter:
    """Reads ONLY the two pooled output rates. No geometry access."""
    gain: float = 1.0
    bias: float = 0.0
    max_steering: float = math.radians(30.0)

    def __call__(self, left: float, right: float) -> float:
        s = self.gain * (left - right) + self.bias
        return max(-self.max_steering, min(self.max_steering, s))


class NeuralController:
    """Bridges Brain + Adapter into a simulation Controller (obs -> steering).

    Holds the observation across `neural_updates` brain steps per physics step.
    """

    def __init__(self, brain: Brain, adapter: Adapter, neural_updates: int = 2):
        self.brain = brain
        self.adapter = adapter
        self.neural_updates = neural_updates

    def reset(self) -> None:
        self.brain.reset()

    def __call__(self, obs) -> float:
        stim = self.brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(self.neural_updates):
            self.brain.step(stim)
        left, right = self.brain.outputs()
        return self.adapter(left, right)
