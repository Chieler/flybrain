"""Checks for the parallel controls path in evaluate.py.

The parallel run must be identical to the serial one; episodes are independent
and deterministic. The only hand-rolled logic is _summary_from_rows (numerators
and denominators including failures), so pin it against evaluate_controller.

Run: python test_parallel.py   (no connectome needed — uses the brain-free
conventional controller so it stays fast).
"""

import json
import os
import tempfile

import evaluate as ev
from evaluate import (
    Summary, _arrivals_key, _atomic_write_json, _mean_return_key,
    _summary_from_rows, episode_return, run_controls,
)
from simulation import run_episode


def test_summary_matches_serial():
    sc = ev.generate_scenarios(16, seed=7)
    serial = ev.evaluate_controller(sc, ev.conventional_baseline())
    rows = []
    for s in sc:
        r = run_episode(s, ev.conventional_baseline())  # stateless controller
        rows.append((r.outcome, r.elapsed_time, r.path_length, episode_return(r)))
    parallel = _summary_from_rows(rows)
    assert serial.as_dict() == parallel.as_dict(), (serial.as_dict(), parallel.as_dict())


def test_summary_edges():
    empty = _summary_from_rows([])
    assert empty.trials == 0 and empty.mean_return == 0.0
    assert empty.mean_arrival_time is None and empty.mean_route_length is None

    rows = [("arrival", 1.0, 2.0, 0.9),
            ("boundary", 0.0, 0.0, -1.0),
            ("timeout", 5.0, 3.0, -0.05)]
    s = _summary_from_rows(rows)
    assert (s.trials, s.arrivals, s.boundaries, s.timeouts) == (3, 1, 1, 1)
    assert s.mean_arrival_time == 1.0 and s.mean_route_length == 2.0


def _summ(arrivals, mean_return):
    # trials/boundaries/timeouts irrelevant to the selection keys under test.
    return Summary(100, arrivals, 0, 0, None, None, mean_return)


def test_arrivals_key_prefers_arrivals_over_return():
    # Bracket selection (step 1) must maximize arrivals first, even when another
    # cell has a higher mean_return.
    hi_arrivals = (_summ(25, -0.5), 64.0, -0.05)
    hi_return = (_summ(18, +0.1), 48.0, 0.0)
    winner = max([hi_arrivals, hi_return], key=lambda t: _arrivals_key(*t))
    assert winner is hi_arrivals
    # Original selection maximizes mean_return first — opposite winner.
    winner0 = max([hi_arrivals, hi_return], key=lambda t: _mean_return_key(*t))
    assert winner0 is hi_return


def test_arrivals_key_tiebreaks_bias_then_gain():
    # Equal arrivals and mean_return: prefer smaller |bias|, then smaller gain.
    a = (_summ(20, -0.4), 96.0, -0.10)
    b = (_summ(20, -0.4), 96.0, 0.0)     # smaller |bias|
    c = (_summ(20, -0.4), 48.0, 0.0)     # same |bias|, smaller gain
    assert max([a, b], key=lambda t: _arrivals_key(*t)) is b
    assert max([b, c], key=lambda t: _arrivals_key(*t)) is c


def test_atomic_write_round_trips_and_cleans_up():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results.json")
        _atomic_write_json(path, {"neural": {"arrivals": 20}})
        assert json.load(open(path)) == {"neural": {"arrivals": 20}}
        # No stray temp files left behind.
        assert os.listdir(d) == ["results.json"]


def test_run_controls_resumes_and_skips_completed():
    # A fully-populated progress file must short-circuit every condition: no
    # Brain is built (data_dir is bogus) and the loaded results come straight
    # back. This is what makes a killed multi-hour run resumable.
    done = {name: {"trials": 100, "arrivals": 20} for name in
            ("conventional", "zero", "random",
             "neural", "cue_withheld", "pathway_silenced")}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "results.json")
        _atomic_write_json(path, done)
        res = run_controls("/nonexistent-connectome", 32.0, -0.05,
                           ev.RateParams(), heldout=[], calib=[],
                           shuffle_seeds=(), progress_path=path)
    assert res == done


if __name__ == "__main__":
    test_summary_matches_serial()
    test_summary_edges()
    test_arrivals_key_prefers_arrivals_over_return()
    test_arrivals_key_tiebreaks_bias_then_gain()
    test_atomic_write_round_trips_and_cleans_up()
    test_run_controls_resumes_and_skips_completed()
    print("ok")
