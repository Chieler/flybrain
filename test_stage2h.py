import numpy as np
import pytest
import stage2h
from street import StreetCarState, StreetScenario


def one_straight_scenario():
    return StreetScenario(
        "cross", StreetCarState(-21.0, 0.0, 0.0, 0.0), 21.0, 0.0)


def test_env_contract_and_unchanged_observation_boundary():
    env = stage2h.StreetNavigationEnv([one_straight_scenario()], seed=3)
    obs, info = env.reset(seed=3, options={"scenario_index": 0})
    assert obs.shape == (10,)
    assert env.observation_space.contains(obs)
    assert set(info) == {"scenario_index", "layout"}
    assert list(env.observation_space.low) == [-1.0] * 4 + [0.0] * 6
    assert list(env.observation_space.high) == [1.0] * 10


def test_action_mapping_uses_existing_control_limits():
    lo = stage2h.action_to_control(np.array([-1.0, -1.0]))
    hi = stage2h.action_to_control(np.array([1.0, 1.0]))
    assert lo.steering == pytest.approx(-stage2h.MAX_STEERING)
    assert lo.acceleration == pytest.approx(-stage2h.MAX_BRAKE)
    assert hi.steering == pytest.approx(stage2h.MAX_STEERING)
    assert hi.acceleration == pytest.approx(stage2h.MAX_ACCEL)


def test_reward_prefers_progress_and_arrival_and_penalizes_collision():
    assert stage2h.episode_reward(20.0, 19.0, 40.0, None) > -0.001
    assert stage2h.episode_reward(20.0, 19.0, 40.0, "arrival") > 10.0
    assert stage2h.episode_reward(20.0, 19.0, 40.0, "collision") < -3.0


def test_gym_checker_accepts_environment():
    from gymnasium.utils.env_checker import check_env
    check_env(stage2h.StreetNavigationEnv([one_straight_scenario()], seed=3))
