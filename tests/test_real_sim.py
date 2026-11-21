import pytest

from sbctl.sim import rollout, starter_path


@pytest.mark.skipif(
    not (starter_path() / "smart_buildings/evaluate.py").exists(),
    reason="starter kit unavailable",
)
@pytest.mark.parametrize("policy", ["tuned", "pid_water"])
def test_real(policy):
    if policy == "tuned":
        scenario = (
            starter_path()
            / "smart_buildings/training_scenarios/building_train_1/fairbanks_1"
        )
        result = rollout(scenario_dir=scenario, steps=288, policy=policy)
    else:
        result = rollout("building_train_1", "fairbanks_1", steps=288, policy=policy)
    assert result["num_steps"] == 288
    assert result["mean_reward"] <= 0
    assert result["seconds_per_step"] > 0
    assert result["component_totals"]["normalized_energy_cost"] > 0
