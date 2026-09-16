"""Stage 2c: a same-observation recurrent/history baseline (the route-memory gate).

Purpose (see HANDOFF Stage 2c). Stage 2b left a bounded question: the
observation-only state machine, carrying no route/topological memory, reached
only 0.62 on the frozen gate while the privileged waypoint witness reached 1.00.
This experiment tests whether *temporal memory over the same observation
interface* closes that gap -- before any dopamine/connectome learning.

Model: a seeded Echo-State Network -- a fixed random leaky-tanh reservoir with a
ridge-regression linear readout. NumPy only, fully deterministic; no BPTT.

Interface discipline (identical to Stage 2b's SM): the controller sees exactly
`heading, goal_bearing, speed, ranges`. `observation_features` maps those to a
normalized vector with *faithful* angles -- heading and goal_bearing each as
sin/cos, with no engineered compass-difference term -- so the network must learn
the heading<->goal relation itself from temporal state. `RecurrentController`
has the `(self, obs)` call signature (asserted in `test_stage2c`); it never sees
geometry, a map, or its pose.

Matched ablation: `EchoStateNetwork(recurrent=False)` disables the reservoir's
temporal state entirely (recurrent weights zero AND unit leak, so the reservoir
becomes a static random-feature map of the *current* observation). Same input
weights, same readout-fitting procedure, same training data. The recurrent-minus-
ablation difference isolates what temporal memory buys.

Labels come from behavior cloning of the Stage 2b `WaypointController` (the
solvability witness): its privileged map/route/pose is used ONLY to generate
target actions and never enters the network's inputs. This is imitation of a
privileged teacher, with the usual covariate-shift limitation -- a documented
caveat, not a claim that a pass proves the interface sufficient in general.

Not neural in the connectome sense; this does not touch the connectome or the
frozen Stage 2 / Stage 2b artifacts.
"""

from __future__ import annotations

import math

import numpy as np

from simulation import MAX_STEERING
from street import Control, MAX_ACCEL, MAX_BRAKE, MAX_SPEED, SENSOR_RANGE

# Feature layout: sin/cos(heading), sin/cos(goal_bearing), speed, 5 ranges.
N_FEATURES = 10


def observation_features(obs) -> np.ndarray:
    """Normalized, observation-only feature vector (faithful sin/cos angles).

    Uses only StreetObservation fields (heading, goal_bearing, speed, ranges).
    Heading and goal_bearing are encoded as sin/cos separately -- no engineered
    difference term -- so the recurrent readout must learn the relation itself.
    """
    ranges = tuple(min(1.0, max(0.0, r / SENSOR_RANGE)) for r in obs.ranges)
    return np.array([
        math.sin(obs.heading), math.cos(obs.heading),
        math.sin(obs.goal_bearing), math.cos(obs.goal_bearing),
        obs.speed / MAX_SPEED, *ranges,
    ], dtype=float)


class EchoStateNetwork:
    """Seeded leaky-tanh reservoir with a ridge-regression readout.

    Reservoir update (leak `a`, input weights W_in, recurrent weights W):
        x_t = (1 - a) x_{t-1} + a * tanh(W_in @ [1; u_t] + W @ x_{t-1})
    Readout is linear over the concatenation [1; u_t; x_t], fit by ridge.

    `recurrent=False` is the matched memoryless ablation: W is zeroed and the
    leak forced to 1, so x_t = tanh(W_in @ [1; u_t]) is a static random-feature
    map of the current observation -- same W_in, same readout fitting.
    """

    def __init__(self, n_reservoir: int = 32, spectral_radius: float = 0.9,
                 leak: float = 0.3, input_scale: float = 1.0, seed: int = 0,
                 recurrent: bool = True):
        self.n = n_reservoir
        self.recurrent = recurrent
        self.leak = leak if recurrent else 1.0
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-1.0, 1.0, (n_reservoir, N_FEATURES + 1)) * input_scale
        if recurrent:
            W = rng.uniform(-1.0, 1.0, (n_reservoir, n_reservoir))
            radius = float(np.max(np.abs(np.linalg.eigvals(W))))
            self.W = W * (spectral_radius / radius) if radius > 0.0 else W
        else:
            self.W = np.zeros((n_reservoir, n_reservoir))
        self.W_out: np.ndarray | None = None    # (2, 1 + N_FEATURES + n)
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(self.n)

    def _readout_vec(self, u: np.ndarray) -> np.ndarray:
        """Advance the reservoir one step on features `u`; return [1; u; x_t]."""
        ub = np.concatenate(([1.0], u))
        pre = self.W_in @ ub + self.W @ self.x
        self.x = (1.0 - self.leak) * self.x + self.leak * np.tanh(pre)
        return np.concatenate((ub, self.x))

    def fit(self, feature_seqs, target_seqs, ridge: float = 1e-3) -> "EchoStateNetwork":
        """Ridge-fit the readout from teacher demos.

        `feature_seqs[i]` is a per-step sequence of feature vectors; the matching
        `target_seqs[i]` holds `(steering, acceleration)` targets. The reservoir
        is reset at the start of each sequence.
        """
        rows, targets = [], []
        for feats, targs in zip(feature_seqs, target_seqs):
            self.reset()
            for u, t in zip(feats, targs):
                rows.append(self._readout_vec(u))
                targets.append(t)
        S = np.asarray(rows, dtype=float)
        Y = np.asarray(targets, dtype=float)
        d = S.shape[1]
        self.W_out = np.linalg.solve(S.T @ S + ridge * np.eye(d), S.T @ Y).T
        self.reset()
        return self

    def predict(self, u: np.ndarray) -> np.ndarray:
        """One-step readout for features `u` (advances reservoir state)."""
        if self.W_out is None:
            raise RuntimeError("EchoStateNetwork.predict called before fit")
        return self.W_out @ self._readout_vec(u)


class RecurrentController:
    """Observation-only controller wrapping a fitted EchoStateNetwork.

    Sees exactly `heading, goal_bearing, speed, ranges` (no map, no pose); the
    call signature is `(self, obs)`, asserted in `test_stage2c`. Carries the
    reservoir's temporal state across steps; `reset()` clears it per episode.
    """

    def __init__(self, esn: EchoStateNetwork):
        self.esn = esn

    def reset(self) -> None:
        self.esn.reset()

    def __call__(self, obs) -> Control:
        y = self.esn.predict(observation_features(obs))
        steering = float(np.clip(y[0], -MAX_STEERING, MAX_STEERING))
        acceleration = float(np.clip(y[1], -MAX_BRAKE, MAX_ACCEL))
        return Control(steering, acceleration)


class TrajectoryRecorder:
    """Wraps a controller, recording per-step (features, [steering, accel]).

    Used to harvest behavior-cloning demonstrations by rolling a teacher through
    the real simulator (`run_street_episode`), so the recorded observations and
    actions match the integrator exactly.
    """

    def __init__(self, inner):
        self.inner = inner
        self.features: list[np.ndarray] = []
        self.targets: list[list[float]] = []

    def reset(self) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self.features = []
        self.targets = []

    def __call__(self, obs) -> Control:
        control = self.inner(obs)
        self.features.append(observation_features(obs))
        self.targets.append([control.steering, control.acceleration])
        return control
