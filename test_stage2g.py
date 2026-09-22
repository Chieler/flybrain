import numpy as np
import stage2g
import stage2d
from stage2d import CEMConfig, cem_maximize
from street import initial_layouts
from evaluate_stage2b import generate_stage2b_split


def test_recurrent_net_has_190_trainable_params():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])  # give W_out a shape
    theta = stage2g.flatten_theta(esn, recurrent=True)
    assert theta.size == 8 * 11 + 8 * 8 + 2 * 19  # 88 + 64 + 38 == 190


def test_ablation_net_has_126_trainable_params_and_zero_frozen_W():
    esn = stage2g.make_net(recurrent=False)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = stage2g.flatten_theta(esn, recurrent=False)
    assert theta.size == 8 * 11 + 2 * 19            # 88 + 38 == 126
    assert np.array_equal(esn.W, np.zeros((8, 8)))  # W removed from search
    assert esn.leak == 1.0                          # memoryless


def test_set_theta_is_inverse_of_flatten():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = np.arange(190, dtype=float)
    stage2g.set_theta(esn, theta, recurrent=True)
    assert np.array_equal(stage2g.flatten_theta(esn, recurrent=True), theta)


def test_ablation_set_theta_leaves_W_zero():
    esn = stage2g.make_net(recurrent=False)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    stage2g.set_theta(esn, np.ones(126), recurrent=False)
    assert np.array_equal(esn.W, np.zeros((8, 8)))


def test_block_scales_formula():
    esn = stage2g.make_net(recurrent=True)
    esn.fit([np.zeros((1, 10))], [np.zeros((1, 2))])
    theta = np.concatenate([np.full(88, 2.0), np.full(64, 0.0), np.full(38, 5.0)])
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta, slices)
    assert np.allclose(scales[:88], 0.1 * 2.0)          # RMS 2.0
    assert np.allclose(scales[88:152], 0.1 * 1e-3)      # RMS 0 -> floor
    assert np.allclose(scales[152:], 0.1 * 5.0)         # RMS 5.0


def test_warm_start_is_deterministic_bit_identical():
    _, t0a = stage2g.warm_start_theta(recurrent=True)
    _, t0b = stage2g.warm_start_theta(recurrent=True)
    assert np.array_equal(t0a, t0b)              # same seed -> identical theta0


def test_warm_start_sizes_match_arm():
    _, t_rec = stage2g.warm_start_theta(recurrent=True)
    _, t_abl = stage2g.warm_start_theta(recurrent=False)
    assert t_rec.size == 190
    assert t_abl.size == 126


def _tiny_scenarios():
    return generate_stage2b_split(
        999, counts={"cross": 0, "regular": 2, "asymmetric": 0})


def test_noop_controller_makes_no_arrivals():
    layouts = initial_layouts()
    arr, _ = stage2g.policy_fitness_arrivals(
        stage2g.NoOpController(), _tiny_scenarios(), layouts)
    assert arr == 0


def test_go_no_go_population_matches_cem_iteration_zero():
    # go/no-go's sampled population is exactly cem_maximize's iteration-0 draw,
    # so the full 30-iter run at the same seed reproduces it bit-for-bit.
    layouts = initial_layouts()
    scenarios = _tiny_scenarios()
    esn, theta0 = stage2g.warm_start_theta(True, layouts)
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta0, slices)
    cfg = CEMConfig(population=4, n_iter=1, init_std=1.0, seed=0)

    probe = stage2g.go_no_go(esn, theta0, scales, scenarios, layouts, cfg, True)

    def fitness_z(z):
        stage2g.set_theta(esn, theta0 + scales * z, True)
        arr, fit = stage2g.policy_fitness_arrivals(
            stage2c_controller := __import__("stage2c").RecurrentController(esn),
            scenarios, layouts)
        return fit

    _, info = cem_maximize(fitness_z, theta0.size, cfg, init_mu=np.zeros(theta0.size))
    assert probe["population_best_fitness"] == info["best"]["fitness"]


def test_cem_maximize_is_untouched_by_stage2g():
    # Stage 2g must reparameterize AROUND cem_maximize, never edit it.
    import inspect
    src = inspect.getsource(stage2d.cem_maximize)
    assert "sigma = np.full(dim, cfg.init_std)" in src   # signature line intact


def test_trainer_guard_keeps_warm_start_when_nothing_beats_it():
    layouts = initial_layouts()
    scenarios = _tiny_scenarios()
    esn, theta0 = stage2g.warm_start_theta(True, layouts)
    slices = stage2g.trainable_slices(esn, recurrent=True)
    scales = stage2g.block_scales(theta0, slices)
    from stage2f import aligned_fitness

    # zero scales => every candidate == theta0 => nothing can beat the warm start
    esn, info = stage2g.train_by_reward_blockscaled(
        esn, theta0, np.zeros_like(scales), scenarios, layouts,
        CEMConfig(population=4, n_iter=2, init_std=1.0, seed=0),
        aligned_fitness, recurrent=True)
    assert info["improved_over_warmstart"] is False
    assert np.allclose(np.asarray(info["best_theta"]), theta0)
    assert info["best_fitness"] == info["warmstart_fitness"]


def test_split_sizes_and_strata():
    import evaluate_stage2g as e2g
    train, dev, gate = e2g.build_all_splits()
    assert len(train) == 66 and len(dev) == 46 and len(gate) == 100
    for split, n_cross in ((train, 6), (dev, 6), (gate, 12)):
        assert sum(s.layout == "cross" for s in split) == n_cross


def test_splits_pairwise_disjoint_and_clear_of_priors():
    import evaluate_stage2g as e2g
    from evaluate_stage2 import _scenario_key
    train, dev, gate = e2g.build_all_splits()
    keys = [{_scenario_key(s) for s in sp} for sp in (train, dev, gate)]
    assert keys[0].isdisjoint(keys[1])
    assert keys[0].isdisjoint(keys[2])
    assert keys[1].isdisjoint(keys[2])
    excluded = {_scenario_key(s) for s in e2g._load(e2g.EXCLUDE_PATHS)}
    for k in keys:
        assert k.isdisjoint(excluded)     # incl. sm_train_split.json (BC demo)


def test_bc_demo_split_is_in_exclusion_set():
    import evaluate_stage2g as e2g
    assert stage2g.BC_DEMO_SPLIT in e2g.EXCLUDE_PATHS


def test_every_prior_scored_gate_is_enforced_excluded():
    import evaluate_stage2g as e2g
    for g in ("runs/stage2b/gate_split.json", "runs/stage2d/gate_split.json",
              "runs/stage2e/gate_split.json", "runs/stage2f/gate_split.json"):
        assert g in e2g.EXCLUDE_PATHS


def test_overlap_provenance_enforced_zero_covers_all_priors():
    import evaluate_stage2g as e2g
    train, dev, gate = e2g.build_all_splits()
    report = e2g.overlap_provenance(train, dev, gate)
    assert set(e2g.EXCLUDE_PATHS).issubset(set(e2g.ALL_PRIOR_SPLITS))
    for name in ("train", "dev", "gate"):
        assert set(report[name]) == set(e2g.ALL_PRIOR_SPLITS)   # every prior split accounted
        for path in e2g.EXCLUDE_PATHS:
            assert report[name][path] == 0                      # enforced overlaps are zero


def test_dev_gate_score_is_min_of_normalized_rates():
    import evaluate_stage2g as e2g
    bd = {"overall": {"arrival_rate": 0.90},
          "by_layout": {"regular": {"arrival_rate": 0.80},
                        "asymmetric": {"arrival_rate": 0.80},
                        "cross": {"arrival_rate": 0.40}}}  # cross half of bar
    assert abs(e2g.dev_gate_score(bd) - 0.5) < 1e-9        # 0.40/0.80 == 0.5


def test_run_arm_escalates_when_dev_arrivals_do_not_improve(monkeypatch):
    # Force the trainer to return the warm start unchanged (zero scales path):
    import evaluate_stage2g as e2g
    import stage2g
    real = stage2g.block_scales
    monkeypatch.setattr(stage2g, "block_scales",
                        lambda theta, slices: real(theta, slices) * 0.0)
    # Force go/no-go to pass so we reach the dev-improvement check:
    monkeypatch.setattr(stage2g, "go_no_go",
                        lambda *a, **k: {"passed": True, "probe_best_arrivals": 1,
                                         "population_best_fitness": 0.0,
                                         "noop_fitness": -1.0, "beats_noop": True})
    layouts = initial_layouts()
    train = e2g.build_split(70, {"cross": 0, "regular": 2, "asymmetric": 0}, [])
    dev = e2g.build_split(71, {"cross": 0, "regular": 2, "asymmetric": 0}, train)
    arm = e2g.run_arm(True, train, dev, layouts,
                      e2g.CEM_BUDGET.__class__(population=2, n_iter=1,
                                               init_std=1.0, seed=0))
    assert arm["escalate"] is True                         # no dev improvement


def test_interpret_both_pass_is_bounded_to_policy_family():
    import evaluate_stage2g as e2g
    passing = {"passes_gate": True, "overall_arrival_rate": 0.95,
               "by_layout_arrival_rate": {"regular": 0.9, "asymmetric": 0.9,
                                          "cross": 0.9}}
    rec = dict(passing)
    abl = dict(passing)
    text = e2g.interpret(rec, abl)
    assert "within this gate and policy family" in text
    assert "not required" not in text or "family" in text


def test_interpret_recurrent_only_pass_beats_ablation():
    import evaluate_stage2g as e2g
    rec = {"passes_gate": True, "overall_arrival_rate": 0.95,
           "by_layout_arrival_rate": {"regular": 0.9, "asymmetric": 0.9, "cross": 0.9}}
    abl = {"passes_gate": False, "overall_arrival_rate": 0.60,
           "by_layout_arrival_rate": {"regular": 0.7, "asymmetric": 0.6, "cross": 0.5}}
    text = e2g.interpret(rec, abl)
    assert "temporal state" in text or "learned temporal" in text
    assert "attribution" in text.lower()      # honest-attribution caveat present
