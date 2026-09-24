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


def test_heading_pool_has_32_unique_orientations_and_cardinals():
    import math
    import evaluate_stage2h as e2h
    assert len(e2h.HEADINGS) == len(set(e2h.HEADINGS)) == 32
    for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        assert heading in e2h.HEADINGS


def test_tiny_witnessed_splits_are_disjoint_and_solvable():
    import evaluate_stage2h as e2h
    from evaluate_stage2 import _scenario_key
    from stage2b import WaypointController
    from street import initial_layouts, run_street_episode
    counts = {"cross": 1, "regular": 1, "asymmetric": 1}
    first = e2h.build_split(980, counts, [])
    second = e2h.build_split(981, counts, first)
    assert {_scenario_key(s) for s in first}.isdisjoint(
        {_scenario_key(s) for s in second})
    layouts = initial_layouts()
    for scenario in first + second:
        controller = WaypointController(scenario, layouts[scenario.layout])
        assert run_street_episode(
            scenario, layouts[scenario.layout], controller).outcome == "arrival"


def test_frozen_stage2h_splits_have_declared_sizes_and_no_overlap():
    import evaluate_stage2h as e2h
    from evaluate_stage2 import _scenario_key, load_scenarios
    paths = ["runs/stage2h/train_split.json",
             "runs/stage2h/readiness_split.json",
             "runs/stage2h/gate_split.json"]
    splits = [load_scenarios(path) for path in paths]
    assert [len(split) for split in splits] == [450, 300, 300]
    for split, count in zip(splits, (150, 100, 100)):
        assert {name: sum(s.layout == name for s in split)
                for name in ("cross", "regular", "asymmetric")} == {
                    "cross": count, "regular": count, "asymmetric": count}
    keys = [{_scenario_key(s) for s in split} for split in splits]
    assert keys[0].isdisjoint(keys[1])
    assert keys[0].isdisjoint(keys[2])
    assert keys[1].isdisjoint(keys[2])
    prior = {_scenario_key(s) for s in e2h._load(e2h.PRIOR_SPLITS)}
    assert all(group.isdisjoint(prior) for group in keys)
