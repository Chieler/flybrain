import numpy as np
import stage2g


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
