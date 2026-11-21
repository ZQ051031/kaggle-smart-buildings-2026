import numpy as np
import pytest

from sbctl.policy import DEFAULT_PARAMS, PARAM_BOUNDS, TunedTransferPolicy
from sbctl.sim import rollout
from tests.mock_env import MockEnv


@pytest.mark.parametrize("k,v", [(1, 1), (2, 6), (4, 31), (1, 0), (7, 3)])
def test_valid_deterministic(k, v):
    e = MockEnv(k, v)
    a, b = TunedTransferPolicy(e.metadata), TunedTransferPolicy(e.metadata)
    rng = np.random.default_rng(9)
    for _ in range(30):
        obs = e.obs() + rng.normal(0, 0.1, len(e.obs()))
        x, y = a.get_action(obs), b.get_action(obs)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
        c, d = x["continuous_action"], x["discrete_action"]
        assert c.shape == (3 * k + 2 + 2 * v,) and c.dtype == np.float32
        assert d.shape == (k + 1,) and d.dtype == np.int32
        assert np.isfinite(c).all() and (abs(c) <= 1).all()
        assert np.isin(d, [0, 1]).all()


def test_permuted_scattering():
    original, permuted = MockEnv(), MockEnv(permuted=True)
    a, b = (
        TunedTransferPolicy(original.metadata),
        TunedTransferPolicy(permuted.metadata),
    )
    obs = original.obs()
    obs[::3][:6] = np.linspace(-2, 2, 6)
    x, y = a.get_action(obs), b.get_action(obs)
    for role in [
        "temp_indices",
        "pressure_indices",
        "water_indices",
        "recirculation_indices",
    ]:
        np.testing.assert_allclose(
            x["continuous_action"][original.metadata[role]],
            y["continuous_action"][permuted.metadata[role]],
        )
    for z, fields in original.metadata["zone_action_indices"].items():
        for f, i in fields.items():
            assert (
                x["continuous_action"][i]
                == y["continuous_action"][
                    permuted.metadata["zone_action_indices"][z][f]
                ]
            )


def test_missing_nan_and_params():
    e = MockEnv()
    for names in [e.metadata["observation_field_names"], []]:
        e.metadata["observation_field_names"] = names
        p = TunedTransferPolicy(e.metadata)
        for obs in [
            np.full(len(names), np.nan),
            np.array([]),
            np.full(len(names), np.inf),
        ]:
            assert np.isfinite(p.get_action(obs)["continuous_action"]).all()
    with pytest.raises(ValueError):
        TunedTransferPolicy(e.metadata, params={"kp": float("nan")})
    assert DEFAULT_PARAMS.keys() == PARAM_BOUNDS.keys()


def test_closed_loop():
    controlled, zero = MockEnv(), MockEnv()
    p = TunedTransferPolicy(controlled.metadata)
    errors = [0.0, 0.0]
    for _ in range(864):
        controlled.step(p.get_action(controlled.obs()))
        zero.step(
            dict(
                continuous_action=np.zeros(zero.metadata["n_continuous_action"]),
                discrete_action=np.zeros(zero.k + 1),
            )
        )
        for i, e in enumerate([controlled, zero]):
            errors[i] += (
                np.maximum(294 - e.temp, 0).sum() + np.maximum(e.temp - 297, 0).sum()
            )
    assert errors[0] < errors[1] * 0.5


def test_empty_rollout_and_bad_horizon():
    with pytest.raises(ValueError):
        rollout("unused", "unused", steps=-1)
