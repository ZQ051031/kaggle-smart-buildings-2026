# Offline search and evaluation

Use the installed Python explicitly. No extra optimizer dependency is required:
the search is a NumPy `(mu, lambda)` evolution strategy, not CMA-ES. The policy
remains NumPy-only. Ruff is a development-only tool installed in the venv.

## Data separation

```sh
/home/ubuntu/sbenv/bin/python -m train.generate_pool --output work --count 12 --start-seed 12000
```

This calls the starter kit's building and scenario generation functions; it
does not modify that checkout. Geometry ranges are 300–400 by 180–250 feet,
2–3 floors and an AHU floor of 3–7. Materials are stratified over the batch.
Seeds overlapping existing starter building configs are skipped, as are
geometries rejected by the starter's minimum-zone-area check.

The initial pool contains 12 accepted buildings (seeds 12001–12012), 11–46
zones and 3–7 AHUs. Seed 12000 was rejected by the minimum-area check. Each is
crossed with a recorded strip from every discovered station, cycling through
strips across buildings: **72 scenarios across all six stations**, including
Minneapolis. No station or building identity enters the controller.

All assets, generated configs, manifests, search state and logs live in
gitignored `work/`:

* `training.json`: 72 generated scenario directories, the ES objective pool.
* `validation.json`: all 30 shipped scenario directories, held out of ES.
* `promotion.json`: five fixed shipped scenarios, one per available station,
  used only for best-by-validation parameter promotion. The remaining 25 are
  not used in search or promotion.
* `pool.json`: seeds, geometry statistics, rejected seeds and data manifests.

Manifest paths are absolute in the generated output. Regenerate the pool when
moving machines. The scenario generator references recorded weather assets in
the external starter checkout; none are vendored into tracked source files.

## Search

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup /home/ubuntu/sbenv/bin/python -u -m train.es_search \
  --training work/training.json --validation work/promotion.json \
  --work work/search --workers 7 --population 8 --batch 3 --steps 864 \
  --hours 3.5 --generations 100 --seed 2026 \
  > work/search.log 2>&1 < /dev/null &
```

The ES initializes at the current `sbctl/params.json`, normalizes all 16
parameters through `PARAM_BOUNDS`, samples four antithetic pairs, reflects at
bounds, and recombines the top four candidates with logarithmic rank weights.
Normalized sigma starts at 0.08, decays by 0.99 per generation and floors at
0.015. Every generation samples three scenarios without replacement; all eight
candidates share those scenarios, the 864-step horizon and per-scenario random
seeds. The objective is the equally weighted mean of negative per-step reward.

Only that generation's training champion is evaluated on the fixed promotion
set, with fixed seeds and the same horizon. `sbctl/params.json` is atomically
replaced only when its validation metric improves on the incumbent. The
incumbent is evaluated initially. The search never uses promotion rewards to
rank candidates or update its ES center. Validation selection still risks
overfitting this small subset; use the full 30 for a final independent check.

`work/search/generations.jsonl` logs `gen`, `best`, `mean`, physical `params`,
training batch, validation metrics, promotion decisions and timings.
`checkpoint.json` stores the center, sigma, RNG state, validation incumbent,
configuration, elapsed budget and generation records. Checkpoints are atomic;
pending parameter promotions are recoverable. Failures stop rather than hiding
bad rollouts behind an arbitrary penalty.

Resume with the same arguments plus `--resume`. The wall-time budget includes
previously checkpointed elapsed runtime; increase `--hours` to extend it. The
script stops before starting a generation whose recent maximum duration would
exceed the budget. One in-flight generation may overrun if its runtime grows.
The generation cap is 100; the wall budget is normally the binding limit.
The checkpoint rejects changed scenario lists, policy source, seed, search
configuration or externally changed promoted parameters. Workers use spawn;
do not use threads for the simulator's global gin configuration and cwd.

For a seconds-long synthetic smoke test that **cannot modify production
parameters**:

```sh
/home/ubuntu/sbenv/bin/python -m train.es_search --quick --work work/quick
```

Quick mode writes promotions only to its own `quick_params.json`. Tests also
check resume equivalence to an uninterrupted quick run.

## Timing and budget

A seven-worker pilot, seven generated scenarios at 864 steps, measured mean
simulator time **0.06846 s/step**, maximum **0.07402 s/step**. Using the maximum
for every job, four training waves plus one validation wave cost roughly
`5 × 864 × 0.07402 + 10 = 330 s/generation`. A 3.5-hour budget therefore permits
about 38 generations, comfortably above the 15-generation target. This is an
estimate, not a guarantee; batch geometry and parallel contention vary.

## Evaluation

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ubuntu/sbenv/bin/python -m train.evaluate \
  --scenarios work/validation.json --steps 8640 --policy tuned \
  --workers 7 --output work/validation_tuned
```

Policies include `tuned` and every policy registered in the external starter's
`POLICY_MAP` (`pid_water`, `pid`, `bang_bang`, `fixed_setpoint`, `all_off`,
`transfer`). A tuned evaluation snapshots the parameter file before dispatch,
so a concurrent search cannot mix generations in the evaluation report.
Use `--params PATH` to evaluate a frozen tuned parameter file.

Outputs are JSON and Markdown, with per-scenario mean reward and the overall
competition metric. At exactly 8640 steps, matching shipped paths are compared
with `baseline_returns.json / 8640`; no invalid short-horizon baseline
comparison is made. Generated paths with coincidentally matching names are
not treated as shipped scenarios.

## Checks

```sh
/home/ubuntu/sbenv/bin/ruff format --check sbctl/policy.py sbctl/sim.py train tests/test_training.py tests/test_real_sim.py
/home/ubuntu/sbenv/bin/python -m pytest -q
```

The readability-only cleanup was separately checked against 500 saved action
steps: continuous/discrete arrays were bit-identical before and after. Its
pytest run reported `11 passed, 14 warnings in 29.66s`. The warnings originate
from the installed matplotlib/pyparsing versions, not an unloaded environment.
