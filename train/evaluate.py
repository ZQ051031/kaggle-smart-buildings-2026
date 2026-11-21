"""Parallel headless evaluation with competition-metric JSON/Markdown output."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from sbctl.sim import rollout, starter_path
from sbctl.policy import DEFAULT_PARAMS


def load_scenarios(path):
    """Read a JSON array of scenario directories relative to the list file."""
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError("Scenario list must be a nonempty JSON array")
    scenarios = []
    for value in values:
        directory = Path(value)
        directory = directory if directory.is_absolute() else path.parent / directory
        directory = directory.resolve()
        if not (directory / "config.gin").is_file():
            raise FileNotFoundError(directory / "config.gin")
        scenarios.append(str(directory))
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("Duplicate scenarios in list")
    return scenarios


def evaluate_task(task):
    """Worker entry point; a fixed seed resets simulator randomness per rollout."""
    if task.get("quick"):
        from tests.mock_env import MockEnv
        from sbctl.policy import TunedTransferPolicy

        environment = MockEnv(seed=task["seed"])
        environment.temp[:] += (task["seed"] % 5 - 2) * 0.5
        controller = TunedTransferPolicy(environment.metadata, params=task["params"])
        rewards = []
        for _ in range(task["steps"]):
            _, reward = environment.step(controller.get_action(environment.obs()))
            rewards.append(reward)
        return dict(
            scenario_dir=task["scenario"],
            policy="tuned",
            num_steps=len(rewards),
            mean_reward=float(np.mean(rewards)),
        )
    result = rollout(
        scenario_dir=task["scenario"],
        steps=task["steps"],
        policy=task.get("policy", "tuned"),
        params=task.get("params"),
        seed=task["seed"],
    )
    if result["num_steps"] != task["steps"]:
        raise RuntimeError(f"Short rollout: {result['num_steps']} != {task['steps']}")
    return result


def baseline_comparison(rows, steps, baseline_file=None):
    """Compare only full episodes and only baseline entries for actual shipped paths."""
    if steps != 8640:
        return {}
    root = starter_path()
    baseline_file = (
        Path(baseline_file) if baseline_file else root / "baseline_returns.json"
    )
    if not baseline_file.exists():
        return {}
    reference = json.loads(baseline_file.read_text())
    comparisons = {}
    shipped_root = (root / "smart_buildings/training_scenarios").resolve()
    for row in rows:
        scenario = Path(row["scenario_dir"]).resolve()
        if not scenario.is_relative_to(shipped_root):
            continue
        key = str(scenario.relative_to(shipped_root))
        if key in reference:
            comparisons[key] = {
                policy: float(total) / 8640 for policy, total in reference[key].items()
            }
    return comparisons


def write_report(rows, output, steps, baseline_file=None):
    """Write per-scenario means and the equally weighted negative-mean metric."""
    if not rows:
        raise ValueError("Cannot report empty evaluation")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric = float(np.mean([-row["mean_reward"] for row in rows]))
    comparisons = baseline_comparison(rows, steps, baseline_file)
    baseline_metrics = {}
    if comparisons:
        for policy in sorted(
            set.intersection(*(set(values) for values in comparisons.values()))
        ):
            baseline_metrics[policy] = -float(
                np.mean([values[policy] for values in comparisons.values()])
            )
    result = dict(
        steps=steps,
        metric=metric,
        scenarios=rows,
        baseline_mean_rewards=comparisons,
        baseline_metrics=baseline_metrics,
        baseline_scenario_count=len(comparisons),
    )
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        f"# Evaluation ({steps} steps)",
        "",
        f"Overall metric: **{metric:.8f}** (lower is better; {len(rows)} scenarios).",
        "",
        "| Scenario | Policy | Mean reward |",
        "|---|---|---:|",
    ]
    for row in rows:
        scenario = Path(row["scenario_dir"])
        lines.append(
            f"| {scenario.parent.name}/{scenario.name} | {row['policy']} | {row['mean_reward']:.8f} |"
        )
    if comparisons:
        lines += [
            "",
            f"Full-episode shipped baselines ({len(comparisons)} matching scenarios only):",
            "",
            "| Baseline | Metric |",
            "|---|---:|",
        ]
        lines += [
            f"| {policy} | {value:.8f} |" for policy, value in baseline_metrics.items()
        ]
        lines += ["", "| Scenario | Baseline | Mean reward |", "|---|---|---:|"]
        for scenario, values in comparisons.items():
            lines += [
                f"| {scenario} | {policy} | {value:.8f} |"
                for policy, value in values.items()
            ]
    elif steps != 8640:
        lines += ["", "No baseline_returns.json comparison: horizon is not 8640 steps."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8640)
    parser.add_argument("--policy", default="tuned")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--output", type=Path, default=Path("work/evaluation"))
    args = parser.parse_args()
    if args.steps <= 0 or args.workers <= 0:
        parser.error("steps and workers must be positive")
    params = None
    if args.policy == "tuned":
        parameter_path = (
            args.params or Path(__file__).resolve().parents[1] / "sbctl/params.json"
        )
        params = {**DEFAULT_PARAMS, **json.loads(parameter_path.read_text())}
    tasks = [
        dict(
            scenario=scenario,
            steps=args.steps,
            seed=args.seed,
            policy=args.policy,
            params=params,
        )
        for scenario in load_scenarios(args.scenarios)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(evaluate_task, tasks, chunksize=1)
    report = write_report(rows, args.output, args.steps)
    print(
        json.dumps(
            dict(metric=report["metric"], scenarios=len(rows), output=str(args.output))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
