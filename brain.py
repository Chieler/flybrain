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

import json
import math
from dataclasses import dataclass
from pathlib import Path

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

    @classmethod
    def load(cls, data_dir: str, params: RateParams | None = None) -> "Brain":
        """Load a prepared connectome + interface map from prepare_connectome.py.

        The output map (PFL3 L/R) is complete. The heading/goal input map is only
        populated when the manifest carries preferred angles; until then those
        roles are empty and cue encoding for them is a no-op (spontaneous /
        stimulated recurrent dynamics still run — no fabricated angles).
        """
        data = Path(data_dir)
        W = sp.load_npz(data / "weights_signed.npz")
        manifest = json.loads((data / "manifest.json").read_text())
        im = manifest["interface_map"]
        input_map: dict = {}
        for role, key in (("heading", "heading_input"), ("goal", "goal_input")):
            entry = im.get(key, {})
            angles = entry.get("preferred_angles")
            if angles is not None:
                input_map[role] = (np.array(entry["indices"]),
                                   np.array(angles, dtype=np.float32))
        output_map = {"left": np.array(im["left_output"]["indices"]),
                      "right": np.array(im["right_output"]["indices"])}
        return cls(W, input_map, output_map, params)

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


@dataclass
class Stage2Adapter:
    brain_gain: float
    bias: float
    avoidance_gain: float
    brake_distance: float
    cruise_speed: float = 2.0
    max_steering: float = math.radians(30.0)
    # --- opt-in commit-and-hold recovery (disabled when commit_distance <= 0) ---
    # When the forward sensor clearance falls below commit_distance, commit to the
    # more-open side and hold a decisive turn (overriding the goal pull) until the
    # forward lane reopens past release_distance. recover_speed keeps enough
    # forward motion for the bicycle model to actually rotate out of the trap.
    # Sensor-only: reads ranges/left_open/right_open, never geometry.
    commit_distance: float = 0.0
    release_distance: float = 0.0
    recover_speed: float = 1.0
    # --- opt-in goal-aware commit (disabled when goal_clearance <= 0) ---
    # When choosing the commit side, prefer the side the brain/compass already
    # wants -- the sign of the neural steer term brain_gain*(left-right)+bias --
    # provided that side is at least goal_clearance open (normalized 0..1). If
    # the goal side is walled below that clearance, fall back to the most-open
    # side. This derives the goal direction from the pooled rates already passed
    # in; it reads no geometry (the __call__ signature is unchanged).
    goal_clearance: float = 0.0

    def __post_init__(self) -> None:
        self._committed = 0  # 0 = free, +1 = holding left, -1 = holding right

    def reset(self) -> None:
        """Clear the recovery state. Call at the start of every episode."""
        self._committed = 0

    def __call__(self, left: float, right: float, ranges: tuple[float, ...],
                 speed: float):
        from street import Control, MAX_ACCEL, MAX_BRAKE, SENSOR_RANGE
        normalized = tuple(r / SENSOR_RANGE for r in ranges)
        left_open = sum(normalized[:2]) / 2
        right_open = sum(normalized[-2:]) / 2
        neural_steer = self.brain_gain * (left - right) + self.bias
        steering = neural_steer + self.avoidance_gain * (left_open - right_open)
        steering = max(-self.max_steering, min(self.max_steering, steering))
        desired = self.cruise_speed * min(1.0, ranges[2] / self.brake_distance)
        desired *= max(0.35, 1.0 - abs(steering) / self.max_steering)
        if self.commit_distance > 0.0:
            forward = ranges[2]
            if self._committed == 0:
                if forward < self.commit_distance:
                    open_side = 1 if left_open >= right_open else -1
                    if self.goal_clearance > 0.0:
                        # Prefer the side the brain/compass wants, if it is open
                        # enough; otherwise escape toward the most-open side.
                        goal_side = 1 if neural_steer >= 0.0 else -1
                        goal_open = left_open if goal_side > 0 else right_open
                        self._committed = (goal_side if goal_open >= self.goal_clearance
                                           else open_side)
                    else:
                        self._committed = open_side
            elif forward > self.release_distance:
                self._committed = 0  # forward lane has reopened
            if self._committed != 0:
                steering = self._committed * self.max_steering
                desired = self.recover_speed  # keep moving so the car can rotate
        acceleration = max(-MAX_BRAKE, min(MAX_ACCEL, 2.0 * (desired - speed)))
        return Control(steering, acceleration)


class Stage2NeuralController:
    def __init__(self, brain: Brain, adapter: Stage2Adapter, neural_updates: int = 2):
        self.brain = brain
        self.adapter = adapter
        self.neural_updates = neural_updates

    def reset(self) -> None:
        self.brain.reset()
        self.adapter.reset()

    def __call__(self, obs):
        stimulus = self.brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(self.neural_updates):
            self.brain.step(stimulus)
        left, right = self.brain.outputs()
        return self.adapter(left, right, obs.ranges, obs.speed)
