"""Resumable normalized (mu, lambda)-ES with antithetic common-random rollouts."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from sbctl.policy import DEFAULT_PARAMS, PARAM_BOUNDS
from sbctl.sim import starter_path
from train.evaluate import evaluate_task, load_scenarios

PARAMETER_NAMES = tuple(PARAM_BOUNDS)
LOWER = np.array([PARAM_BOUNDS[name][0] for name in PARAMETER_NAMES])
UPPER = np.array([PARAM_BOUNDS[name][1] for name in PARAMETER_NAMES])
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "sbctl/params.json"


def normalize(parameters):
    """Map physical parameters to the bounded unit search cube."""
    vector = np.array([parameters[name] for name in PARAMETER_NAMES], dtype=float)
    if (
        not np.isfinite(vector).all()
        or np.any(vector < LOWER)
        or np.any(vector > UPPER)
    ):
        raise ValueError("Initial parameters must be finite and within PARAM_BOUNDS")
    return (vector - LOWER) / (UPPER - LOWER)


def decode(vector):
    """Map bounded search coordinates back to portable controller parameters."""
    values = LOWER + np.clip(vector, 0, 1) * (UPPER - LOWER)
    return dict(zip(PARAMETER_NAMES, map(float, values)))


def atomic_json(path, value):
    """Replace a complete checkpoint/parameter file, never exposing partial JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sample_population(center, sigma, population, rng):
    """Reflect antithetic perturbations at bounds rather than clipping their mass."""
    noise = rng.normal(size=(population // 2, len(center)))
    raw = center + sigma * np.concatenate([noise, -noise])
    reflected = 1 - np.abs((raw % 2) - 1)
    return reflected


def score_candidates(pool, candidates, scenarios, steps, seed, quick):
    """All candidates share scenario order, horizon, and per-scenario seed."""
    tasks = [
        dict(
            scenario=scenario,
            params=parameters,
            steps=steps,
            seed=seed + index,
            quick=quick,
        )
        for parameters in candidates
        for index, scenario in enumerate(scenarios)
    ]
    rows = pool.map(evaluate_task, tasks, chunksize=1)
    scores = (
        np.array([-row["mean_reward"] for row in rows])
        .reshape(len(candidates), len(scenarios))
        .mean(axis=1)
    )
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite objective; refusing to update search state")
    return scores


def persist(work, state):
    """Checkpoint at generation boundaries and reconstruct the matching JSONL log."""
    atomic_json(work / "checkpoint.json", state)
    temporary = work / "generations.jsonl.tmp"
    temporary.write_text(
        "".join(
            json.dumps(record, allow_nan=False) + "\n" for record in state["records"]
        )
    )
    os.replace(temporary, work / "generations.jsonl")


def run_search(args):
    """Optimize generated scenarios; promote only on the fixed held-out subset."""
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # One writer per checkpoint, including across resumed invocations.
    with (work / "search.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(args, work)


def _run_locked(args, work):
    if args.population < 2 or args.population % 2 or args.batch < 1 or args.workers < 1:
        raise ValueError("Need an even population >= 2, positive batch and workers")
    if args.sigma <= 0 or args.hours <= 0:
        raise ValueError("sigma and hours must be positive")
    if args.quick:
        training = [f"mock-train-{index}" for index in range(6)]
        validation = ["mock-validation-0", "mock-validation-1"]
        promotion_path = work / "quick_params.json"
    else:
        if args.training is None or args.validation is None:
            raise ValueError("Real search requires --training and --validation lists")
        training = load_scenarios(args.training)
        validation = load_scenarios(args.validation)
        if set(training) & set(validation):
            raise ValueError("Training and validation scenarios overlap")
        if any(Path(path).is_relative_to(args.starter_scenarios) for path in training):
            raise ValueError("Shipped scenarios are held out and cannot enter training")
        promotion_path = DEFAULT_PARAMS_PATH
    if args.batch > len(training):
        raise ValueError("Batch exceeds training pool")
    steps = args.steps if args.steps is not None else (96 if args.quick else 864)
    generations = (
        args.generations if args.generations is not None else (3 if args.quick else 100)
    )
    if steps <= 0 or generations < 1:
        raise ValueError("steps and generations must be positive")
    contract = dict(
        training=training,
        validation=validation,
        steps=steps,
        population=args.population,
        batch=args.batch,
        seed=args.seed,
        quick=args.quick,
        parameter_bounds=PARAM_BOUNDS,
        initial_sigma=args.sigma,
        policy_sha256=hashlib.sha256(
            (DEFAULT_PARAMS_PATH.parent / "policy.py").read_bytes()
        ).hexdigest(),
    )
    # JSON roundtrip makes tuple/list comparison stable when resuming.
    contract = json.loads(json.dumps(contract))
    checkpoint = work / "checkpoint.json"
    rng = np.random.default_rng(args.seed)
    state = None
    if checkpoint.exists():
        if not args.resume:
            raise ValueError("Checkpoint exists; pass --resume or use a new work dir")
        state = json.loads(checkpoint.read_text())
        if state["contract"] != contract:
            raise ValueError("Resume configuration differs from checkpoint")
        rng.bit_generator.state = state["rng_state"]
        if (
            promotion_path.exists()
            and json.loads(promotion_path.read_text()) != state["incumbent_params"]
        ):
            if json.loads(promotion_path.read_text()) != state.get("pending_promotion"):
                raise ValueError(
                    "Promoted params changed externally; refusing stale validation"
                )
        if state.get("pending_promotion") is not None:
            atomic_json(promotion_path, state["incumbent_params"])
            del state["pending_promotion"]
        persist(work, state)
    elif args.resume:
        raise FileNotFoundError(checkpoint)
    starting_params = DEFAULT_PARAMS.copy()
    starting_params.update(json.loads(DEFAULT_PARAMS_PATH.read_text()))
    start_time = time.monotonic()
    prior_elapsed = state["elapsed_seconds"] if state else 0.0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        if state is None:
            baseline = float(
                score_candidates(
                    pool, [starting_params], validation, steps, args.seed, args.quick
                )[0]
            )
            state = dict(
                contract=contract,
                generation=0,
                center=normalize(starting_params).tolist(),
                sigma=args.sigma,
                rng_state=rng.bit_generator.state,
                incumbent_params=starting_params,
                incumbent_validation=baseline,
                records=[],
                elapsed_seconds=time.monotonic() - start_time,
            )
            persist(work, state)
            print(
                json.dumps(
                    dict(
                        event="initial_validation",
                        metric=baseline,
                        seconds=state["elapsed_seconds"],
                        scenarios=validation,
                    )
                ),
                flush=True,
            )
        for generation in range(state["generation"] + 1, generations + 1):
            elapsed = prior_elapsed + time.monotonic() - start_time
            previous_time = max(
                (record["seconds"] for record in state["records"][-3:]), default=0.0
            )
            if elapsed + previous_time >= args.hours * 3600:
                print(
                    json.dumps(
                        dict(
                            event="budget_stop",
                            elapsed_seconds=elapsed,
                            generation=state["generation"],
                        )
                    ),
                    flush=True,
                )
                break
            generation_start = time.monotonic()
            scenario_batch = [
                training[index]
                for index in rng.choice(len(training), args.batch, replace=False)
            ]
            population = sample_population(
                np.array(state["center"]), state["sigma"], args.population, rng
            )
            candidates = [decode(vector) for vector in population]
            scores = score_candidates(
                pool,
                candidates,
                scenario_batch,
                steps,
                args.seed + generation * 10000,
                args.quick,
            )
            ranking = np.argsort(scores, kind="stable")
            champion = candidates[int(ranking[0])]
            validation_score = float(
                score_candidates(
                    pool, [champion], validation, steps, args.seed, args.quick
                )[0]
            )
            promoted = validation_score < state["incumbent_validation"] - 1e-12
            previous_validation = state["incumbent_validation"]
            if promoted:
                if (
                    promotion_path.exists()
                    and json.loads(promotion_path.read_text())
                    != state["incumbent_params"]
                ):
                    raise RuntimeError(
                        "Params changed externally during search; refusing overwrite"
                    )
                state["pending_promotion"] = state["incumbent_params"]
                state["incumbent_validation"] = validation_score
                state["incumbent_params"] = champion
            elite_count = args.population // 2
            weights = np.log(elite_count + 0.5) - np.log(np.arange(1, elite_count + 1))
            weights /= weights.sum()
            state["center"] = np.sum(
                population[ranking[:elite_count]] * weights[:, None], axis=0
            ).tolist()
            state["sigma"] = max(0.015, state["sigma"] * 0.99)
            state["generation"] = generation
            state["rng_state"] = rng.bit_generator.state
            state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start_time
            record = dict(
                gen=generation,
                best=float(scores[ranking[0]]),
                mean=float(scores.mean()),
                params=champion,
                validation=validation_score,
                previous_validation=previous_validation,
                best_validation=state["incumbent_validation"],
                promoted=promoted,
                sigma=state["sigma"],
                scenarios=scenario_batch,
                steps=steps,
                seconds=time.monotonic() - generation_start,
                elapsed_seconds=state["elapsed_seconds"],
            )
            state["records"].append(record)
            persist(work, state)
            if promoted:
                atomic_json(promotion_path, champion)
                del state["pending_promotion"]
                persist(work, state)
            print(json.dumps(record, allow_nan=False), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--work", type=Path, default=Path("work/search"))
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.starter_scenarios = (
        starter_path() / "smart_buildings/training_scenarios"
    ).resolve()
    run_search(args)


if __name__ == "__main__":
    main()
