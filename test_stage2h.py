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


@pytest.mark.parametrize("recurrent", [False, True])
def test_ppo_arm_learns_and_scores_without_privileged_policy_inputs(tmp_path, recurrent):
    scenarios = [one_straight_scenario()]
    model = stage2h.make_model(recurrent, scenarios, seed=0, n_envs=1,
                               overrides={"n_steps": 32, "batch_size": 32,
                                          "n_epochs": 1})
    model.learn(total_timesteps=32)
    path = tmp_path / ("recurrent" if recurrent else "feedforward")
    model.save(path)
    score = stage2h.score_model(model, scenarios)
    assert score["trials"] == 1
    assert score["arrivals"] + score["collisions"] + score["timeouts"] == 1
    assert set(score["by_layout"]) == {"cross"}


def test_gate_score_is_worst_normalized_threshold():
    score = {"overall_arrival_rate": 0.90,
             "by_layout_arrival_rate": {
                 "cross": 0.40, "regular": 0.80, "asymmetric": 0.80}}
    assert stage2h.gate_score(score) == pytest.approx(0.5)


def test_select_seed_maximizes_worst_spent_gate_then_lower_seed():
    import evaluate_stage2h as e2h
    rows = [
        {"seed": 0, "selection_score": 0.7},
        {"seed": 1, "selection_score": 0.8},
        {"seed": 2, "selection_score": 0.8},
    ]
    assert e2h.select_seed(rows)["seed"] == 1


def test_gate_requires_recurrent_readiness_pass(tmp_path):
    import evaluate_stage2h as e2h
    results = tmp_path / "results.json"
    results.write_text('{"readiness":{"recurrent":{"passes_gate":false}}}')
    with pytest.raises(SystemExit, match="readiness"):
        e2h.require_gate_open(results, tmp_path / "gate_results.json")


def test_gate_refuses_to_overwrite_spent_result(tmp_path):
    import evaluate_stage2h as e2h
    results = tmp_path / "results.json"
    gate = tmp_path / "gate_results.json"
    results.write_text('{"readiness":{"recurrent":{"passes_gate":true}}}')
    gate.write_text('{}')
    with pytest.raises(SystemExit, match="already exists"):
        e2h.require_gate_open(results, gate)
