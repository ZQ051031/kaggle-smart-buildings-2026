"""Build a dataset-free Kaggle notebook from the controller and current params."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import pprint
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle/submission.ipynb"

HEADER = """# Smart Buildings — portable structured controller

This self-contained submission embeds the NumPy controller and a frozen parameter
snapshot. It needs only the competition data and Kaggle's provided inference
server: **no attached code dataset, pip installation, or internet access**.

1. Create/import this notebook in the NeurIPS 2026 Smart Buildings competition.
2. **Attach the competition data** and turn **Internet off**.
3. Choose **Save Version → Save & Run All**.
4. When the run completes, choose **Submit** for that notebook version.

The server creates a fresh controller for each building. Every action is checked
for shape, finite values, bounds, and dtype. A controller exception or malformed
action reuses the last valid action; before the first valid action it uses
minimum continuous commands with equipment off. This emergency default is
actuator-valid, not a guarantee of comfort during a persistent controller fault.

Maintainers: regenerate with `python tools/build_notebook.py` after changing the
policy or parameters. `--check` detects a stale notebook. The local test-only
`SB_COMPETITION_DATA_FOLDER` override is unnecessary on Kaggle.
"""

SETUP = """import os
import sys

_data_override = os.environ.get("SB_COMPETITION_DATA_FOLDER")
if _data_override:
    if not os.path.isdir(_data_override):
        raise FileNotFoundError(_data_override)
    COMPETITION_DATA_FOLDER = os.path.join(_data_override, "")
else:
    COMPETITION_DATA_FOLDER = next(
        path
        for path in (
            "/kaggle/input/competitions/neurips-2026-smart-buildings/",
            "/kaggle/input/neurips-2026-smart-buildings/",
        )
        if os.path.isdir(path)
    )
sys.path[:0] = [
    COMPETITION_DATA_FOLDER,
    os.path.join(COMPETITION_DATA_FOLDER, "kaggle_evaluation"),
]
from kaggle_evaluation import smart_buildings_inference_server
"""

WRAPPER = '''class SubmissionPolicy(smart_buildings_inference_server.Policy):
    """Guard the portable controller at the competition boundary."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._continuous_shape = (int(building_meta_data["n_continuous_action"]),)
        self._discrete_shape = (int(building_meta_data["n_discrete_action"]),)
        self._last_valid_action = {
            "continuous_action": np.full(self._continuous_shape, -1.0, dtype=np.float32),
            "discrete_action": np.zeros(self._discrete_shape, dtype=np.int32),
        }
        self._controller = None
        try:
            self._controller = TunedTransferPolicy(
                building_meta_data, action_spec, params=SUBMISSION_PARAMS.copy()
            )
        except Exception:
            pass

    def get_action(self, obs):
        """Return a fresh valid action, falling back on any controller error."""
        try:
            action = self._controller.get_action(obs)
            continuous = np.asarray(action["continuous_action"], dtype=np.float64)
            discrete = np.asarray(action["discrete_action"], dtype=np.float64)
            if continuous.shape != self._continuous_shape:
                raise ValueError("Incorrect continuous action shape")
            if discrete.shape != self._discrete_shape:
                raise ValueError("Incorrect discrete action shape")
            if not np.isfinite(continuous).all() or not np.isfinite(discrete).all():
                raise ValueError("Nonfinite action")
            self._last_valid_action = {
                "continuous_action": np.clip(continuous, -1.0, 1.0).astype(np.float32),
                "discrete_action": (np.clip(discrete, 0.0, 1.0) >= 0.5).astype(np.int32),
            }
        except Exception:
            pass
        return {key: value.copy() for key, value in self._last_valid_action.items()}
'''

RUN_SERVER = """smart_buildings_inference_server.SmartBuildingsInferenceServer(
    SubmissionPolicy
).run(competition_data_folder=COMPETITION_DATA_FOLDER)
"""


def inline_policy(source):
    """Replace only the file-based default loader; retain controller source text."""
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TunedTransferPolicy"
    )
    constructor = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignment = ast.parse('path = Path(__file__).with_name("params.json")').body[0]
    loader = ast.parse(
        "if path.exists():\n    self.params.update(json.loads(path.read_text()))"
    ).body[0]
    statements = constructor.body
    matches = [
        index
        for index in range(len(statements) - 1)
        if ast.dump(statements[index]) == ast.dump(assignment)
        and ast.dump(statements[index + 1]) == ast.dump(loader)
    ]
    if len(matches) != 1:
        raise ValueError("Policy default loader changed; update notebook inlining")
    start, end = statements[matches[0]], statements[matches[0] + 1]
    lines = source.splitlines(keepends=True)
    replacements = [
        (
            start.lineno - 1,
            end.end_lineno,
            "        self.params.update(SUBMISSION_PARAMS)\n",
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.Import) and ast.dump(node) == ast.dump(
            ast.parse("import json").body[0]
        ):
            replacements.append((node.lineno - 1, node.end_lineno, ""))
        if isinstance(node, ast.ImportFrom) and ast.dump(node) == ast.dump(
            ast.parse("from pathlib import Path").body[0]
        ):
            replacements.append((node.lineno - 1, node.end_lineno, ""))
    for first, last, replacement in sorted(replacements, reverse=True):
        lines[first:last] = [replacement]
    result = "".join(lines)
    if "__file__" in result:
        raise ValueError("Inlined policy still depends on __file__")
    return result


def make_cell(cell_type, source, cell_id):
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": textwrap.dedent(source).splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook(
    policy_path=ROOT / "sbctl/policy.py", params_path=ROOT / "sbctl/params.json"
):
    """Read one atomic parameter snapshot and produce deterministic notebook JSON."""
    source = Path(policy_path).read_text()
    parameter_source = Path(params_path).read_text()
    parameters = json.loads(parameter_source)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Parameters must be a nonempty dict")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in parameters.values()
    ):
        raise ValueError("Parameters must be finite numeric values")
    literal = pprint.pformat(parameters, sort_dicts=True, width=88)
    return {
        "cells": [
            make_cell("markdown", HEADER, "instructions"),
            make_cell("code", SETUP, "competition-interface"),
            make_cell(
                "code",
                f"SUBMISSION_PARAMS = {literal}\n\n" + inline_policy(source),
                "embedded-controller",
            ),
            make_cell("code", WRAPPER, "guarded-policy"),
            make_cell("code", RUN_SERVER, "inference-server"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "sbctl": {
                "policy_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "params_sha256": hashlib.sha256(parameter_source.encode()).hexdigest(),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def serialize(notebook):
    return json.dumps(notebook, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=NOTEBOOK_PATH)
    args = parser.parse_args(argv)
    expected = serialize(build_notebook())
    if args.check:
        if not args.output.exists() or args.output.read_text() != expected:
            print(
                "Notebook is stale; run: python tools/build_notebook.py",
                file=sys.stderr,
            )
            return 1
        print("Notebook is current: " + str(args.output))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected)
    print("Built " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
