"""Fast headless adapter. Run with python -m sbctl.sim BUILDING WEATHER."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .policy import TunedTransferPolicy


def starter_path():
    """Locate the external starter checkout without importing it."""
    return Path(
        os.environ.get("SB_STARTER_KIT", "/home/ubuntu/repos/starter-kit")
    ).resolve()


@contextmanager
def starter_cwd():
    """Resolve the starter's relative gin assets within one worker process."""
    previous = Path.cwd()
    root = starter_path()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(previous)


def rollout(
    building=None,
    weather=None,
    steps=288,
    start_offset=0,
    policy="tuned",
    params=None,
    audit=None,
    scenario_dir=None,
    seed=0,
):
    """Roll out a policy headlessly, collecting rewards and normalized terms."""
    if steps < 0 or start_offset < 0:
        raise ValueError("steps and start_offset must be nonnegative")
    scenario = Path(scenario_dir).resolve() if scenario_dir is not None else None
    audit_path = Path(audit).resolve() if audit else None
    random.seed(seed)
    np.random.seed(seed)
    with starter_cwd() as root:
        from smart_buildings.evaluate import (
            load_hybrid_environment,
            create_building_meta_data,
            validate_action,
            POLICY_MAP,
        )

        if scenario is None:
            if building is None or weather is None:
                raise ValueError("Provide scenario_dir or both building and weather")
            scenario = (
                root / "smart_buildings" / "training_scenarios" / building / weather
            )
        if policy != "tuned" and policy not in POLICY_MAP:
            raise ValueError(f"Unknown policy: {policy}")
        env = load_hybrid_environment(scenario / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        meta = create_building_meta_data(env, scenario)
        cls = TunedTransferPolicy if policy == "tuned" else POLICY_MAP[policy]
        controller = (
            cls(meta, env.action_spec())
            if policy != "tuned"
            else cls(meta, env.action_spec(), params=params)
        )
        obs, _ = env.reset()
        if audit_path:
            audit_path.write_text(
                json.dumps(
                    dict(
                        observations=dict(
                            zip(meta["observation_field_names"], map(float, obs))
                        ),
                        normalization=meta["observation_normalization_metadata"],
                    ),
                    indent=2,
                )
            )
        rewards = []
        totals = dict(
            normalized_productivity_regret=0.0,
            normalized_energy_cost=0.0,
            normalized_carbon_emission=0.0,
        )
        last_components = {}
        original_reward = env.get_reward_info_and_response

        def capture_reward():
            info, response = original_reward()
            last_components.update(
                {key: float(getattr(response, key)) for key in totals}
            )
            return info, response

        env.get_reward_info_and_response = capture_reward
        elapsed = 0.0
        control_time = 0.0
        for i in range(steps + start_offset):
            t = time.perf_counter()
            action = validate_action(controller.get_action(obs), env.action_spec())
            control_time += time.perf_counter() - t
            t = time.perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            elapsed += time.perf_counter() - t
            if not np.isfinite(reward):
                raise ValueError("Nonfinite simulator reward")
            if i >= start_offset:
                rewards.append(float(reward))
                for key in totals:
                    totals[key] += last_components.get(key, 0.0)
            if terminated or truncated:
                break
        count = i + 1 if steps + start_offset else 0
        del env.get_reward_info_and_response
        return dict(
            building=building or scenario.parent.name,
            weather=weather or scenario.name,
            scenario_dir=str(scenario),
            policy=policy,
            num_steps=len(rewards),
            total_reward=sum(rewards),
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            component_totals=totals,
            seconds_per_step=elapsed / max(count, 1),
            controller_seconds_per_step=control_time / max(count, 1),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("building", nargs="?")
    parser.add_argument("weather", nargs="?")
    parser.add_argument("--scenario-dir", type=Path)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--policy", default="tuned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit")
    args = vars(parser.parse_args())
    print(json.dumps(rollout(**args)), flush=True)


if __name__ == "__main__":
    main()
