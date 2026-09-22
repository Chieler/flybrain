import numpy as np
import stage2g
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
