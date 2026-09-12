"""Checks for the parallel controls path in evaluate.py.

The parallel run must be identical to the serial one; episodes are independent
and deterministic. The only hand-rolled logic is _summary_from_rows (numerators
and denominators including failures), so pin it against evaluate_controller.

Run: python test_parallel.py   (no connectome needed — uses the brain-free
conventional controller so it stays fast).
"""

import evaluate as ev
from evaluate import _summary_from_rows, episode_return
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


if __name__ == "__main__":
    test_summary_matches_serial()
    test_summary_edges()
    print("ok")
