"""Scenario generation, the conventional baseline, calibration, controls, metrics.

The evaluator owns target geometry and rewards. The conventional compass
controller below is a SEPARATELY LABELED baseline: it reads angular error
directly and is not the neural controller.

Adapter calibration (declared grid), the neural controls/interventions
(zero/random steering, cue-withheld, pathway-silenced), and the
shuffled-connectivity control all live here. Running them against the real
graph needs a prepared connectome under `data/` (see SETUP.md); the logic is
unit-tested on synthetic graphs so it is verifiable without that download.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import tempfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from brain import Adapter, Brain, NeuralController, RateParams
from simulation import (
    ARENA_BOUND, MAX_STEERING, TARGET_RADIUS,
    CarState, Scenario, EpisodeResult, run_episode, wrap_angle,
)

ARRIVAL_REWARD = 1.0
BOUNDARY_REWARD = -1.0
TIME_COST = 0.01  # per simulated second


def episode_return(result: EpisodeResult) -> float:
    base = {"arrival": ARRIVAL_REWARD, "boundary": BOUNDARY_REWARD}.get(result.outcome, 0.0)
    return base - TIME_COST * result.elapsed_time


def conventional_baseline(k: float = 4.0):
    """Proportional compass controller. LABELED: reads angular error directly."""
    def controller(obs):
        err = wrap_angle(obs.goal_bearing - obs.heading)
        return max(-MAX_STEERING, min(MAX_STEERING, k * err))
    return controller


def generate_scenarios(n: int, seed: int) -> list[Scenario]:
    """Held-out scenarios: start in [-8,8]^2, target offset 6-15 at uniform
    bearing, uniform initial heading. Reject any target region not fully
    inside the arena; only accepted scenarios are returned."""
    rng = random.Random(seed)
    out: list[Scenario] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError(f"could not sample {n} valid scenarios")
        sx = rng.uniform(-8.0, 8.0)
        sy = rng.uniform(-8.0, 8.0)
        heading = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(6.0, 15.0)
        bearing = rng.uniform(-math.pi, math.pi)
        tx = sx + dist * math.cos(bearing)
        ty = sy + dist * math.sin(bearing)
        # Whole target region must lie inside the arena.
        if abs(tx) + TARGET_RADIUS > ARENA_BOUND or abs(ty) + TARGET_RADIUS > ARENA_BOUND:
            continue
        out.append(Scenario(CarState(sx, sy, heading), tx, ty,
                            label=f"heldout-{len(out)}"))
    return out


def scenario_to_dict(s: Scenario) -> dict:
    return {
        "start": {"x": s.start.x, "y": s.start.y, "heading": s.start.heading},
        "target_x": s.target_x, "target_y": s.target_y,
        "target_radius": s.target_radius, "timeout": s.timeout, "label": s.label,
    }


def scenario_from_dict(d: dict) -> Scenario:
    st = d["start"]
    return Scenario(CarState(st["x"], st["y"], st["heading"]),
                    d["target_x"], d["target_y"],
                    d.get("target_radius", TARGET_RADIUS),
                    d.get("timeout", 30.0), d.get("label", ""))


def save_scenarios(path: str, scenarios: list[Scenario]) -> None:
    with open(path, "w") as f:
        json.dump([scenario_to_dict(s) for s in scenarios], f, indent=2)


def load_scenarios(path: str) -> list[Scenario]:
    with open(path) as f:
        return [scenario_from_dict(d) for d in json.load(f)]


@dataclass
class Summary:
    trials: int
    arrivals: int
    boundaries: int
    timeouts: int
    mean_arrival_time: float | None    # among successes
    mean_route_length: float | None    # among successes
    mean_return: float

    def as_dict(self) -> dict:
        return self.__dict__


def evaluate_controller(scenarios: list[Scenario], controller,
                        reset=None) -> Summary:
    """Run every scenario; report numerators and denominators, incl. failures."""
    arrivals = boundaries = timeouts = 0
    arrival_times: list[float] = []
    route_lengths: list[float] = []
    returns: list[float] = []
    for s in scenarios:
        if reset is not None:
            reset()
        r = run_episode(s, controller)
        r.return_ = episode_return(r)
        returns.append(r.return_)
        if r.outcome == "arrival":
            arrivals += 1
            arrival_times.append(r.elapsed_time)
            route_lengths.append(r.path_length)
        elif r.outcome == "boundary":
            boundaries += 1
        else:
            timeouts += 1
    n = len(scenarios)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return Summary(n, arrivals, boundaries, timeouts,
                   mean(arrival_times), mean(route_lengths),
                   sum(returns) / n if n else 0.0)


# --- Declared adapter calibration grid (plan §Calibration and evaluation) ---
CALIB_GAINS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # rad per unit activity
CALIB_BIASES = (-0.05, 0.0, 0.05)                           # rad

# Declared narrow bracket extension (HANDOFF "exceed 20/100", step 1). The
# original grid's winner (gain=32, bias=-0.05) sat on both edges, so the search
# never bracketed an optimum; this extends beyond it. Selected by arrivals-first
# (see _arrivals_key) on the 32 frozen training scenarios only.
BRACKET_GAINS = (48.0, 64.0, 96.0)
BRACKET_BIASES = (-0.10, -0.05, 0.0)


def _mean_return_key(summ, g, b):
    """Original selection: max mean_return, then min |bias|, then min gain."""
    return (summ.mean_return, -abs(b), -g)


def _arrivals_key(summ, g, b):
    """Bracket selection (step 1): max arrivals, then mean_return, |bias|, gain."""
    return (summ.arrivals, summ.mean_return, -abs(b), -g)


def calibrate_adapter(scenarios, make_controller, gains=CALIB_GAINS,
                      biases=CALIB_BIASES, select_key=_mean_return_key):
    """Exhaust the declared gain×bias grid; return (gain, bias, grid).

    `make_controller(gain, bias) -> (controller, reset)`. `select_key(summ, g, b)`
    produces a sort key maximized over the grid (default: the original
    mean_return-first tie-break). This is adapter calibration, not a claim of
    learning inside the fly brain; the budget is fixed and reported, never
    silently expanded.
    """
    grid = []
    best = None  # (key, gain, bias)
    for g in gains:
        for b in biases:
            controller, reset = make_controller(g, b)
            summ = evaluate_controller(scenarios, controller, reset)
            grid.append({"gain": g, "bias": b, "mean_return": summ.mean_return,
                         "arrivals": summ.arrivals, "trials": summ.trials})
            key = select_key(summ, g, b)
            if best is None or key > best[0]:
                best = (key, g, b)
    return best[1], best[2], grid


def neural_factory(brain: Brain, neural_updates: int = 2):
    """make_controller for calibrate_adapter, backed by a real/synthetic Brain."""
    def make(gain, bias):
        ctrl = NeuralController(brain, Adapter(gain=gain, bias=bias), neural_updates)
        return ctrl, ctrl.reset
    return make


# --- Controls (no neural graph) ---

def zero_steering():
    return (lambda obs: 0.0), None


def random_steering(seed: int):
    """Seeded uniform steering. reset() re-seeds so replay is deterministic."""
    state = {"rng": random.Random(seed)}

    def controller(obs):
        return state["rng"].uniform(-MAX_STEERING, MAX_STEERING)

    def reset():
        state["rng"] = random.Random(seed)
    return controller, reset


# --- Neural interventions (share Brain wiring, lesion one path) ---

def cue_withheld_controller(brain: Brain, adapter: Adapter,
                            neural_updates: int = 2, role: str = "goal"):
    """Neural controller with one input population's cue zeroed (no retuning)."""
    entry = brain.input_map.get(role)
    idx = entry[0] if entry is not None else np.array([], dtype=int)

    def controller(obs):
        stim = brain.encode(obs.heading, obs.goal_bearing)
        stim[idx] = 0.0
        for _ in range(neural_updates):
            brain.step(stim)
        left, right = brain.outputs()
        return adapter(left, right)
    return controller, brain.reset


def pathway_silenced_controller(brain: Brain, adapter: Adapter,
                                neural_updates: int = 2):
    """Neural controller with both mapped output populations clamped to 0.

    Lesions the PFL3 steering readout (also removes its recurrent contribution),
    so the adapter sees (0, 0) and steering collapses to its bias. No retuning.
    """
    silenced = np.concatenate([brain.output_map["left"], brain.output_map["right"]])

    def controller(obs):
        stim = brain.encode(obs.heading, obs.goal_bearing)
        for _ in range(neural_updates):
            brain.step(stim)
            brain.activity[silenced] = 0.0
        left, right = brain.outputs()
        return adapter(left, right)
    return controller, brain.reset


def shuffle_connectivity(W, seed: int):
    """Shuffle anatomical destinations, preserving per-source identity, sign, and
    outgoing-weight multiset (plan's shuffled-connectivity control).

    Preserves: per-source out-degree and its multiset of signed weights.
    Does NOT preserve: destination identity, per-destination in-degree, or any
    dest-side structure. Operate on the pre-normalization signed weights; Brain
    row-normalizes on construction.
    """
    C = sp.csc_array(W)  # columns = presynaptic sources
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    new_rows = np.empty_like(C.indices)
    for c in range(C.shape[1]):
        s, e = C.indptr[c], C.indptr[c + 1]
        k = e - s
        if k:
            # replace=False keeps within-column dest distinct (no CSR summing).
            # ponytail: O(sources) Python loop, run once offline; vectorize if it bites.
            new_rows[s:e] = rng.choice(n, size=k, replace=False)
    shuffled = sp.csc_array((C.data.copy(), new_rows, C.indptr.copy()), shape=C.shape)
    return sp.csr_array(shuffled)


# --- Parallel episode execution (independent scenarios across CPU cores) ---
#
# The controls run is CPU/memory-bandwidth bound on the ~20M-edge sparse matvec,
# but every episode is independent, so scenarios (and calibration grid cells)
# fan out across cores. Each Pool is pinned to ONE Brain variant via its
# initializer, so the 20M-edge load/normalize happens once per worker and the
# brain is reused across all its episodes. Shuffled connectivity is precomputed
# to a temp npz once, so workers skip the O(edges) Python shuffle loop.
# ponytail: worker holds 1 (base) or 2 (base+shuffled) CSR copies (~160MB each);
#           at jobs=8 that's ~2.5GB in the shuffled phase. Lower --jobs if tight.

_WORKER_BRAIN: Brain | None = None


def _init_worker(brain_spec: dict) -> None:
    """Pool initializer: build the pinned Brain once per worker process."""
    global _WORKER_BRAIN
    params = RateParams(**brain_spec["params"])
    base = Brain.load(brain_spec["data_dir"], params)
    npz = brain_spec.get("shuffled_npz")
    if npz is None:
        _WORKER_BRAIN = base
    else:  # shuffled variant reuses base's input/output maps, swaps wiring only
        _WORKER_BRAIN = Brain(sp.load_npz(npz), base.input_map,
                              base.output_map, params)


def _make_controller(mode: str, gain: float, bias: float):
    """Build a (controller, reset) on the worker's pinned Brain."""
    brain = _WORKER_BRAIN
    adapter = Adapter(gain=gain, bias=bias)
    if mode == "neural":
        ctrl = NeuralController(brain, adapter)
        return ctrl, ctrl.reset
    if mode == "cue_withheld":
        return cue_withheld_controller(brain, adapter)
    if mode == "pathway_silenced":
        return pathway_silenced_controller(brain, adapter)
    raise ValueError(f"unknown neural mode {mode!r}")


def _run_one(task: tuple) -> tuple:
    """Run a single episode; return only the picklable scalars we aggregate."""
    mode, gain, bias, s_dict = task
    controller, reset = _make_controller(mode, gain, bias)
    if reset is not None:
        reset()
    r = run_episode(scenario_from_dict(s_dict), controller)
    return (r.outcome, r.elapsed_time, r.path_length, episode_return(r))


def _summary_from_rows(rows: list[tuple]) -> Summary:
    """Aggregate _run_one outputs into a Summary (same math as evaluate_controller)."""
    arrivals = boundaries = timeouts = 0
    arrival_times: list[float] = []
    route_lengths: list[float] = []
    returns: list[float] = []
    for outcome, elapsed, path_length, ret in rows:
        returns.append(ret)
        if outcome == "arrival":
            arrivals += 1
            arrival_times.append(elapsed)
            route_lengths.append(path_length)
        elif outcome == "boundary":
            boundaries += 1
        else:
            timeouts += 1
    n = len(rows)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return Summary(n, arrivals, boundaries, timeouts,
                   mean(arrival_times), mean(route_lengths),
                   sum(returns) / n if n else 0.0)


def _pool_summary(pool, mode: str, gain: float, bias: float,
                  scenarios: list[Scenario]) -> Summary:
    """Parallel evaluate_controller for a neural `mode` over `scenarios`."""
    tasks = [(mode, gain, bias, scenario_to_dict(s)) for s in scenarios]
    return _summary_from_rows(pool.map(_run_one, tasks, chunksize=1))


def _pool_calibrate(pool, scenarios: list[Scenario],
                    gains=CALIB_GAINS, biases=CALIB_BIASES,
                    select_key=_mean_return_key):
    """Parallel calibrate_adapter: flatten grid×scenarios into one balanced batch.

    Same grid and same `select_key` as calibrate_adapter (default: max
    mean_return, then |bias|, then gain), so results are identical to the serial
    path.
    """
    cells = [(g, b) for g in gains for b in biases]
    s_dicts = [scenario_to_dict(s) for s in scenarios]
    tasks = [("neural", g, b, sd) for (g, b) in cells for sd in s_dicts]
    rows = pool.map(_run_one, tasks, chunksize=1)
    per = len(s_dicts)
    grid, best = [], None
    for i, (g, b) in enumerate(cells):
        summ = _summary_from_rows(rows[i * per:(i + 1) * per])
        grid.append({"gain": g, "bias": b, "mean_return": summ.mean_return,
                     "arrivals": summ.arrivals, "trials": summ.trials})
        key = select_key(summ, g, b)
        if best is None or key > best[0]:
            best = (key, g, b)
    return best[1], best[2], grid


# --- Checkpoint I/O ---

def save_checkpoint(path: str, gain: float, bias: float,
                    params: RateParams, data_dir: str) -> None:
    with open(path, "w") as f:
        json.dump({"gain": gain, "bias": bias, "data_dir": data_dir,
                   "params": params.__dict__}, f, indent=2)


def load_checkpoint(path: str) -> dict:
    with open(path) as f:
        cp = json.load(f)
    if "adapter" in cp:
        cp.setdefault("gain", cp["adapter"]["gain"])
        cp.setdefault("bias", cp["adapter"]["bias"])
    if "rate_params" in cp:
        cp.setdefault("params", cp["rate_params"])
    cp["params"] = RateParams(**cp.get("params", {}))
    return cp


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON to `path` via a temp file + rename, so an interrupted write
    never leaves a truncated/corrupt file behind."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_controls(data_dir: str, gain: float, bias: float, params: RateParams,
                 heldout: list[Scenario], calib: list[Scenario],
                 shuffle_seeds=(1, 2, 3), jobs: int | None = None,
                 progress_path: str | None = None,
                 calib_gains=CALIB_GAINS, calib_biases=CALIB_BIASES,
                 select_key=_mean_return_key) -> dict:
    """Held-out neural controller plus every declared control/intervention.

    Shuffled connectivity is RE-calibrated with exactly the same grid, selection
    rule, and interfaces the real-graph candidate got (`calib_gains`,
    `calib_biases`, `select_key`; three fixed seeds) — a matched control. All
    other controls use the frozen adapter. Returns {name: Summary.as_dict()}.

    Neural episodes fan out across `jobs` processes (default: all cores). Results
    are identical to the serial path — episodes are independent and deterministic.

    If `progress_path` is given, each completed condition is flushed there
    atomically as soon as it finishes, and any condition already present in that
    file on entry is skipped. A kill/restart therefore resumes rather than
    discarding the whole (~18 h) run.
    """
    jobs = jobs or os.cpu_count() or 1
    results: dict = {}
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            results = json.load(f)
        if results:
            print(f"resuming: {sorted(results)} already done, skipping")

    def record(name: str, summary: dict) -> None:
        results[name] = summary
        if progress_path:
            _atomic_write_json(progress_path, results)

    # Non-neural controls: no matvec, trivially fast — keep serial.
    if "conventional" not in results:
        record("conventional", evaluate_controller(heldout, conventional_baseline()).as_dict())
    if "zero" not in results:
        record("zero", evaluate_controller(heldout, *zero_steering()).as_dict())
    if "random" not in results:
        record("random", evaluate_controller(heldout, *random_steering(seed=0)).as_dict())

    # Neural + interventions on the base graph: one pool pinned to the base Brain.
    base_spec = {"data_dir": data_dir, "params": params.__dict__}
    base_modes = [m for m in ("neural", "cue_withheld", "pathway_silenced")
                  if m not in results]
    if base_modes:
        with mp.Pool(jobs, initializer=_init_worker, initargs=(base_spec,)) as pool:
            for mode in base_modes:
                record(mode, _pool_summary(pool, mode, gain, bias, heldout).as_dict())

    # Shuffled-connectivity controls: precompute each shuffled W once to a temp
    # npz (so workers skip the O(edges) shuffle loop), then re-calibrate + evaluate.
    pending = [s for s in shuffle_seeds if f"shuffled_seed{s}" not in results]
    if pending:
        W = sp.load_npz(f"{data_dir}/weights_signed.npz")
        for seed in pending:
            fd, npz_path = tempfile.mkstemp(suffix=".npz")
            os.close(fd)
            sp.save_npz(npz_path, sp.csr_array(shuffle_connectivity(W, seed)))
            spec = {**base_spec, "shuffled_npz": npz_path}
            try:
                with mp.Pool(jobs, initializer=_init_worker, initargs=(spec,)) as pool:
                    g, b, _ = _pool_calibrate(pool, calib, gains=calib_gains,
                                              biases=calib_biases, select_key=select_key)
                    summ = _pool_summary(pool, "neural", g, b, heldout)
            finally:
                os.unlink(npz_path)
            record(f"shuffled_seed{seed}",
                   {"calibrated_gain": g, "calibrated_bias": b, **summ.as_dict()})
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 evaluation (baseline slice).")
    p.add_argument("--gen-scenarios", type=int, metavar="N",
                   help="Generate N held-out scenarios and save them.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="scenarios.json")
    p.add_argument("--baseline", action="store_true",
                   help="Run the conventional compass baseline over scenarios.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the declared adapter grid on a real graph; write a checkpoint.")
    p.add_argument("--controls", action="store_true",
                   help="Run held-out neural controller + all controls from a checkpoint.")
    p.add_argument("--bracket", action="store_true",
                   help="Declared step-1 experiment: calibrate the narrow bracket grid "
                        "(gains 48/64/96 x biases -0.10/-0.05/0) on the calibration set "
                        "with arrivals-first selection, then run only the winner on the "
                        "held-out set (neural controller only, no control suite).")
    p.add_argument("--scenarios", default="scenarios.json",
                   help="Held-out scenarios (also the calibration set unless --calib-scenarios).")
    p.add_argument("--calib-scenarios", help="Calibration scenarios (defaults to --scenarios).")
    p.add_argument("--data", help="Prepared connectome dir (needed for --calibrate/--controls).")
    p.add_argument("--checkpoint", default="checkpoint.json")
    p.add_argument("--jobs", type=int, default=None,
                   help="Parallel worker processes for neural episodes "
                        "(default: all CPU cores). Use 1 to force serial.")
    p.add_argument("--select", choices=("return", "arrivals"), default="return",
                   help="Calibration selection rule (--calibrate/--controls shuffled "
                        "recalibration): 'return' = original max mean_return; "
                        "'arrivals' = max arrivals first (bracket experiment).")
    p.add_argument("--calib-grid", choices=("default", "bracket"), default="default",
                   help="Calibration grid: 'default' = original 8x3; 'bracket' = "
                        "the declared narrow extension (48/64/96 x -0.10/-0.05/0).")
    args = p.parse_args()

    _GRIDS = {"default": (CALIB_GAINS, CALIB_BIASES),
              "bracket": (BRACKET_GAINS, BRACKET_BIASES)}
    _SELECTS = {"return": _mean_return_key, "arrivals": _arrivals_key}
    sel_gains, sel_biases = _GRIDS[args.calib_grid]
    sel_key = _SELECTS[args.select]

    if args.gen_scenarios:
        sc = generate_scenarios(args.gen_scenarios, args.seed)
        save_scenarios(args.out, sc)
        print(f"wrote {len(sc)} scenarios to {args.out}")
    elif args.baseline:
        sc = load_scenarios(args.scenarios)
        summary = evaluate_controller(sc, conventional_baseline())
        print(json.dumps(summary.as_dict(), indent=2))
    elif args.calibrate:
        if not args.data:
            p.error("--calibrate needs --data (prepared connectome dir)")
        calib = load_scenarios(args.calib_scenarios or args.scenarios)
        params = RateParams()
        jobs = args.jobs or os.cpu_count() or 1
        spec = {"data_dir": args.data, "params": params.__dict__}
        with mp.Pool(jobs, initializer=_init_worker, initargs=(spec,)) as pool:
            gain, bias, grid = _pool_calibrate(pool, calib, gains=sel_gains,
                                               biases=sel_biases, select_key=sel_key)
        save_checkpoint(args.checkpoint, gain, bias, params, args.data)
        print(json.dumps({"gain": gain, "bias": bias, "budget": len(grid),
                          "checkpoint": args.checkpoint}, indent=2))
    elif args.controls:
        if not args.data:
            p.error("--controls needs --data (prepared connectome dir)")
        cp = load_checkpoint(args.checkpoint)
        heldout = load_scenarios(args.scenarios)
        calib = load_scenarios(args.calib_scenarios or args.scenarios)
        out = args.out if args.out != "scenarios.json" else "results.json"
        # Persist/resume into the output file itself, so a kill mid-run keeps
        # every already-completed condition (plan: no whole-run discard).
        res = run_controls(args.data, cp["gain"], cp["bias"], cp["params"],
                           heldout, calib, jobs=args.jobs, progress_path=out,
                           calib_gains=sel_gains, calib_biases=sel_biases,
                           select_key=sel_key)
        print(json.dumps(res, indent=2))
    elif args.bracket:
        if not args.data:
            p.error("--bracket needs --data (prepared connectome dir)")
        # Step 1 of the "exceed 20/100" plan, run as a DECLARED separate
        # experiment: extended grid, arrivals-first selection, held-out opened
        # exactly once (only the frozen winner). Dynamics come from the frozen
        # Stage 1 checkpoint; only the adapter gain/bias are re-selected here.
        cp = load_checkpoint(args.checkpoint)
        params = cp["params"]
        calib = load_scenarios(args.calib_scenarios or args.scenarios)
        heldout = load_scenarios(args.scenarios)
        out = args.out if args.out != "scenarios.json" else "bracket_results.json"
        spec = {"data_dir": args.data, "params": params.__dict__}
        jobs = args.jobs or os.cpu_count() or 1
        with mp.Pool(jobs, initializer=_init_worker, initargs=(spec,)) as pool:
            gain, bias, grid = _pool_calibrate(
                pool, calib, gains=BRACKET_GAINS, biases=BRACKET_BIASES,
                select_key=_arrivals_key)
            print(json.dumps({"stage": "calibration", "grid": grid,
                              "winner": {"gain": gain, "bias": bias}}, indent=2))
            heldout_summary = _pool_summary(pool, "neural", gain, bias, heldout).as_dict()
        payload = {
            "experiment": "stage1_bracket_step1",
            "selection": "max arrivals, then mean_return, then min abs(bias), then min gain",
            "grid_gains": list(BRACKET_GAINS), "grid_biases": list(BRACKET_BIASES),
            "data_dir": args.data, "rate_params": params.__dict__,
            "n_calib": len(calib), "n_heldout": len(heldout),
            "calibration_grid": grid,
            "winner": {"gain": gain, "bias": bias},
            "heldout": heldout_summary,
            "baseline_reference": {"neural_arrivals": 20, "trials": 100},
        }
        _atomic_write_json(out, payload)
        print(json.dumps({"winner": {"gain": gain, "bias": bias},
                          "heldout": heldout_summary,
                          "beats_20": heldout_summary["arrivals"] > 20,
                          "out": out}, indent=2))
    else:
        p.print_help()
