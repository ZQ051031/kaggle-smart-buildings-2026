import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sbctl.policy import DEFAULT_PARAMS
from train import evaluate
from train.es_search import decode, normalize, sample_population, score_candidates


def test_normalization_and_antithetic():
    center = np.full(len(DEFAULT_PARAMS), 0.5)
    population = sample_population(center, 0.01, 8, np.random.default_rng(42))
    np.testing.assert_allclose(population[:4] + population[4:], 1)
    np.testing.assert_allclose(normalize(decode(center)), center)
    np.testing.assert_allclose(
        normalize(DEFAULT_PARAMS), normalize(decode(normalize(DEFAULT_PARAMS)))
    )
    assert np.all((population >= 0) & (population <= 1))
    with pytest.raises(ValueError):
        normalize({**DEFAULT_PARAMS, "kp": float("nan")})


def test_common_random_numbers():
    class RecordingPool:
        def map(self, function, tasks, chunksize):
            self.tasks = tasks
            return [dict(mean_reward=-0.25) for _ in tasks]

    pool = RecordingPool()
    scores = score_candidates(
        pool, [DEFAULT_PARAMS, DEFAULT_PARAMS], ["a", "b"], 864, 123, True
    )
    np.testing.assert_allclose(scores, [0.25, 0.25])
    keys = ["scenario", "steps", "seed"]
    assert [[task[key] for key in keys] for task in pool.tasks[:2]] == [
        [task[key] for key in keys] for task in pool.tasks[2:]
    ]


def test_quick_search_resume(tmp_path):
    parameter_file = Path(__file__).resolve().parents[1] / "sbctl/params.json"
    before = parameter_file.read_bytes()
    command = [
        sys.executable,
        "-m",
        "train.es_search",
        "--quick",
        "--work",
        str(tmp_path / "search"),
        "--workers",
        "2",
        "--population",
        "4",
        "--batch",
        "2",
        "--steps",
        "48",
    ]
    first = subprocess.run(
        command + ["--generations", "1"], check=True, text=True, capture_output=True
    )
    assert '"gen": 1' in first.stdout
    subprocess.run(
        command + ["--generations", "2", "--resume"],
        check=True,
        text=True,
        capture_output=True,
    )
    checkpoint = json.loads((tmp_path / "search/checkpoint.json").read_text())
    logs = [
        json.loads(line)
        for line in (tmp_path / "search/generations.jsonl").read_text().splitlines()
    ]
    assert checkpoint["generation"] == 2
    assert [row["gen"] for row in logs] == [1, 2]
    assert set(checkpoint["contract"]["training"]).isdisjoint(
        checkpoint["contract"]["validation"]
    )
    assert all(row["best_validation"] <= row["previous_validation"] for row in logs)
    for row in logs:
        assert row["promoted"] == (
            row["validation"] < row["previous_validation"] - 1e-12
        )
    assert parameter_file.read_bytes() == before
    uninterrupted = command.copy()
    uninterrupted[uninterrupted.index(str(tmp_path / "search"))] = str(
        tmp_path / "uninterrupted"
    )
    subprocess.run(
        uninterrupted + ["--generations", "2"],
        check=True,
        text=True,
        capture_output=True,
    )
    other = json.loads((tmp_path / "uninterrupted/checkpoint.json").read_text())
    assert checkpoint["center"] == other["center"]
    assert checkpoint["incumbent_params"] == other["incumbent_params"]
    failed = subprocess.run(
        command + ["--generations", "3", "--resume", "--seed", "9"],
        text=True,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert "Resume configuration differs" in failed.stderr


def test_reports_and_scenario_lists(tmp_path, monkeypatch):
    root = tmp_path / "starter"
    scenario = (
        root / "smart_buildings/training_scenarios/sample_building/sample_weather"
    )
    scenario.mkdir(parents=True)
    (scenario / "config.gin").touch()
    manifest = tmp_path / "list.json"
    manifest.write_text(json.dumps([str(scenario)]))
    assert evaluate.load_scenarios(manifest) == [str(scenario)]
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="nonempty"):
        evaluate.load_scenarios(manifest)
    manifest.write_text(json.dumps([str(tmp_path / "absent")]))
    with pytest.raises(FileNotFoundError):
        evaluate.load_scenarios(manifest)
    monkeypatch.setattr(evaluate, "starter_path", lambda: root)
    (root / "baseline_returns.json").write_text(
        json.dumps({"sample_building/sample_weather": {"pid_water": -864.0}})
    )
    rows = [dict(scenario_dir=str(scenario), mean_reward=-0.05, policy="tuned")]
    report = evaluate.write_report(rows, tmp_path / "report", 8640)
    assert report["metric"] == 0.05
    assert report["baseline_metrics"]["pid_water"] == 0.1
    assert "0.05000000" in (tmp_path / "report.md").read_text()
    assert evaluate.baseline_comparison(rows, 288) == {}
    generated_rows = [
        {**rows[0], "scenario_dir": str(tmp_path / "sample_building/sample_weather")}
    ]
    assert evaluate.baseline_comparison(generated_rows, 8640) == {}
    with pytest.raises(ValueError, match="empty"):
        evaluate.write_report([], tmp_path / "empty", 8640)
