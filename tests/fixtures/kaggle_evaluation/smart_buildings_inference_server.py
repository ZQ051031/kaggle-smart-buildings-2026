"""Synthetic contract test, not a copy of Kaggle's private inference server."""

from pathlib import Path

import numpy as np

from tests.mock_env import MockEnv

LAST_RUN = None


class Policy:
    def __init__(self, building_meta_data, action_spec):
        self.building_meta_data = building_meta_data
        self.action_spec = action_spec


def validate_action(action, metadata):
    """Assert the exact external hybrid action contract."""
    assert set(action) == {"continuous_action", "discrete_action"}
    continuous = action["continuous_action"]
    discrete = action["discrete_action"]
    assert continuous.shape == (metadata["n_continuous_action"],)
    assert discrete.shape == (metadata["n_discrete_action"],)
    assert continuous.dtype == np.float32
    assert discrete.dtype == np.int32
    assert np.isfinite(continuous).all()
    assert ((continuous >= -1) & (continuous <= 1)).all()
    assert np.isin(discrete, [0, 1]).all()


class SmartBuildingsInferenceServer:
    def __init__(self, policy_class):
        self.policy_class = policy_class

    def run(self, competition_data_folder):
        """Exercise the submitted wrapper through 300 synthetic thermal steps."""
        global LAST_RUN
        assert Path(competition_data_folder).is_dir()
        environment = MockEnv(permuted=True)
        policy = self.policy_class(environment.metadata, action_spec=None)
        total_reward = 0.0
        for _ in range(300):
            action = policy.get_action(environment.obs())
            validate_action(action, environment.metadata)
            _, reward = environment.step(action)
            total_reward += reward
        assert np.isfinite(total_reward)
        LAST_RUN = {"steps": 300, "reward": total_reward}
