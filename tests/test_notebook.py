import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.mock_env import MockEnv
from tools import build_notebook

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def notebook_namespace(monkeypatch):
    """Execute the committed artifact without sbctl imports or notebook globals."""
    monkeypatch.setenv("SB_COMPETITION_DATA_FOLDER", str(FIXTURES))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    module_names = [
        "kaggle_evaluation",
        "kaggle_evaluation.smart_buildings_inference_server",
    ]
    saved = {
        name: sys.modules.pop(name) for name in module_names if name in sys.modules
    }
    namespace = {"__name__": "__kaggle_notebook__"}
    notebook = json.loads(build_notebook.NOTEBOOK_PATH.read_text())
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(
                    compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"),
                    namespace,
                )
        yield namespace
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_notebook_runs_offline(notebook_namespace):
    namespace = notebook_namespace
    assert "__file__" not in namespace
    assert namespace["smart_buildings_inference_server"].LAST_RUN["steps"] == 300
    environment = MockEnv()
    policy = namespace["SubmissionPolicy"](environment.metadata, None)
    assert policy._controller.params == namespace["SUBMISSION_PARAMS"]


def test_exception_fallback_and_copy_isolation(notebook_namespace):
    namespace = notebook_namespace
    environment = MockEnv()
    policy = namespace["SubmissionPolicy"](environment.metadata, None)
    expected = policy.get_action(environment.obs())

    class BrokenController:
        def get_action(self, obs):
            raise RuntimeError("injected controller failure")

    policy._controller = BrokenController()
    actual = policy.get_action(None)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        actual[key][:] = 999
        np.testing.assert_array_equal(policy.get_action(None)[key], expected[key])


@pytest.mark.parametrize("malformation", ["nan", "shape", "missing", "type"])
def test_malformed_actions_fall_back(notebook_namespace, malformation):
    namespace = notebook_namespace
    environment = MockEnv()
    policy = namespace["SubmissionPolicy"](environment.metadata, None)
    expected = policy.get_action(environment.obs())
    malformed = {key: value.copy() for key, value in expected.items()}
    if malformation == "nan":
        malformed["continuous_action"][0] = np.nan
    elif malformation == "shape":
        malformed["discrete_action"] = np.ones((2, 2))
    elif malformation == "missing":
        del malformed["continuous_action"]
    else:
        malformed["continuous_action"] = object()

    class MalformedController:
        def get_action(self, obs):
            return malformed

    policy._controller = MalformedController()
    actual = policy.get_action(environment.obs())
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_sanitizes_bounds_and_dtypes(notebook_namespace):
    namespace = notebook_namespace
    environment = MockEnv()
    policy = namespace["SubmissionPolicy"](environment.metadata, None)

    class OutOfBoundsController:
        def get_action(self, obs):
            return {
                "continuous_action": np.full(
                    environment.metadata["n_continuous_action"], 1e100
                ),
                "discrete_action": np.linspace(
                    -4, 5, environment.metadata["n_discrete_action"]
                ),
            }

    policy._controller = OutOfBoundsController()
    action = policy.get_action(None)
    namespace["smart_buildings_inference_server"].validate_action(
        action, environment.metadata
    )
    assert (action["continuous_action"] == 1).all()


def test_constructor_failure_uses_initial_default(notebook_namespace):
    namespace = notebook_namespace
    environment = MockEnv()

    class BrokenConstructor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("injected initialization failure")

    namespace["TunedTransferPolicy"] = BrokenConstructor
    policy = namespace["SubmissionPolicy"](environment.metadata, None)
    action = policy.get_action(None)
    namespace["smart_buildings_inference_server"].validate_action(
        action, environment.metadata
    )
    assert (action["continuous_action"] == -1).all()
    assert (action["discrete_action"] == 0).all()


def test_builder_is_deterministic_and_check_detects_staleness(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "submission.ipynb"
    policy_path = tmp_path / "policy.py"
    params_path = tmp_path / "params.json"
    policy_path.write_text((build_notebook.ROOT / "sbctl/policy.py").read_text())
    params_path.write_text((build_notebook.ROOT / "sbctl/params.json").read_text())
    original_builder = build_notebook.build_notebook

    def frozen_builder():
        return original_builder(policy_path=policy_path, params_path=params_path)

    assert build_notebook.serialize(frozen_builder()) == build_notebook.serialize(
        frozen_builder()
    )
    monkeypatch.setattr(build_notebook, "build_notebook", frozen_builder)
    assert build_notebook.main(["--output", str(output)]) == 0
    assert build_notebook.main(["--output", str(output), "--check"]) == 0
    parameters = json.loads(params_path.read_text())
    parameters["kp"] += 0.001
    params_path.write_text(json.dumps(parameters))
    assert build_notebook.main(["--output", str(output), "--check"]) == 1
    assert build_notebook.main(["--output", str(output)]) == 0
    policy_path.write_text(policy_path.read_text() + "\n")
    assert build_notebook.main(["--output", str(output), "--check"]) == 1
    output.write_text("{}")
    assert build_notebook.main(["--output", str(output), "--check"]) == 1
    assert "Notebook is stale" in capsys.readouterr().err


def test_inline_loader_change_fails_closed():
    with pytest.raises(ValueError, match="default loader changed"):
        build_notebook.inline_policy(
            "class TunedTransferPolicy:\n    def __init__(self):\n        pass\n"
        )
