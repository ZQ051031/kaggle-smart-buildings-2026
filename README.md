# Smart Buildings structured control

Portable NumPy PI zone controller for the NeurIPS 2026 Smart Buildings task.
No climate/building identifiers enter the controller. No remote is configured.

For procedural training, resumable parameter search and parallel evaluation,
see [TRAINING.md](TRAINING.md). The benchmarks below record the initial defaults;
an active search may subsequently promote a new `sbctl/params.json`.

## Kaggle submission

`kaggle/submission.ipynb` is a standalone notebook with the controller and a
frozen parameter literal embedded. Attach competition data, disable internet,
then **Save & Run All** and **Submit**. No code dataset or pip install is needed.

Regenerate after changing policy/parameters (including ES promotions):

```sh
/home/ubuntu/sbenv/bin/python tools/build_notebook.py
/home/ubuntu/sbenv/bin/python tools/build_notebook.py --check
```

The wrapper validates output shapes/dtypes, clips finite actions, and falls back
to the last valid action on exceptions or malformed outputs. Before the first
valid output it uses minimum continuous commands and equipment off; persistent
fallback is not guaranteed to preserve comfort. Offline tests execute the
notebook against a synthetic inference-server fixture, including failure cases.
The actual Kaggle server is not included or tested here.

Development/CI dependencies are in `requirements-dev.txt`. GitHub Actions runs
Ruff, synthetic pytest tests (real simulator tests skip without the starter),
and notebook freshness checking on Python 3.12. To run only synthetic tests
locally while a search is active:

```sh
SB_STARTER_KIT=/nonexistent-starter-kit /home/ubuntu/sbenv/bin/python -m pytest -q
```

## Run

```sh
/home/ubuntu/sbenv/bin/python -m pytest -q
/home/ubuntu/sbenv/bin/python -m sbctl.sim building_train_1 fairbanks_1 --steps 2016
/home/ubuntu/sbenv/bin/python -m sbctl.sim building_train_1 fairbanks_1 --steps 288 --policy pid_water
```

Set `SB_STARTER_KIT` to override `/home/ubuntu/repos/starter-kit`. Only the
simulator adapter imports the kit; `sbctl.policy` needs NumPy and the standard
library. Default parameters load from `sbctl/params.json` over `DEFAULT_PARAMS`;
constructor `params=` overrides both. `PARAM_BOUNDS` defines the search space.
Physical unit conversions and actuator limits are not tuning parameters.
`start_offset` advances the same controller from reset and excludes those
steps from rewards; this is not an instantaneous jump or a fresh controller.
The adapter temporarily changes cwd because shipped gin files use relative
paths: parallelize using processes, not threads. No charts or timeseries are
retained. Component totals are the three **unweighted normalized** reward terms
(comfort regret negative, cost and carbon positive).

## Controller

Per-zone center tracking PI, conditional anti-windup and bounded integral;
absolute demand drives the damper and positive demand drives reheat. Mean AHU
demand resets supply temperature; maximum zone effort resets fan pressure;
maximum reheat demand resets water temperature. Cooling calls refresh an AHU
minimum hold timer. Every continuous slot is initialized to its minimum each
step, then scattered using metadata indices (including the water pressure,
which the starter metadata places at the end of `pressure_indices`).
Unknown/missing sensors have finite neutral defaults. Zone arrays and mappings
are precomputed; AHU aggregation is a short loop.

Sensor bands and comfort flags reflect the real schedule, including holidays
and time zones; the cooling ceiling is never widened at night and heating
setback is limited to 2 K. An additional weekday prestart uses the observed
cyclic clock and metadata morning start. **Limitation:** observations have no
calendar date, so the extra two-hour prestart cannot identify holidays from
the static holiday list; `comfort_mode_soon` only exposes one-hour lookahead.
It can therefore run unnecessarily before a holiday. It does not hardcode a
scenario start date to resolve that ambiguity.

## Observation audit (actual initial Fairbanks observation)

`results/observation_audit.json` records every field/value and the full
normalization metadata, produced using `--audit`.

* `VAV_1_zone_air_temperature_sensor = -0.2`: normalized, mean 295 K,
  std 5 K, hence 294 K.
* Zone heating/cooling setpoint sensors are raw: 294 and 297 K.
* `ahs_ahu_1_outside_air_temperature_sensor = 272.54998779296875`:
  **raw Kelvin**, despite an `outside_air_temperature_sensor` normalization
  entry with mean 295, std 10.
* VAV damper/reheat command observations are normalized (-0.8 / -1.0 at reset).
  Airflow setpoints and AHU prefixed fields are raw unless their actual
  measurement name matches a normalizer key. Cyclic time features are sin/cos;
  comfort flags are raw 0/1; occupancy has a separate environment transform.

The normalizer's `_get_normalization_constants` in
`smart_buildings/smart_control/utils/observation_normalizer.py` performs **exact
match only**, notwithstanding its keyword-match docstring. AHU measurement
names carry a prefix, so bare normalization keys do not transform them.
Do not denormalize outside temperature a second time.

The claimed pid_plus economizer bug is **not present in this checkout**:
`PIDEconomizerPolicy._recirculation` reads `outside = float(obs[oat_idx])`
directly. `PIDWaterResetPolicy` inherits it. The baseline comparison uses that
unmodified implementation; improvements here do not rely on fixing its OAT.

## Seven-day checks

2016 steps from reset, six parallel processes, CPU simulator. Higher (less
negative) mean reward is better; official score negates it and is lower-better.

| Scenario | pid_water mean | tuned mean | loss reduction | sim s/step (pid / tuned) |
|---|---:|---:|---:|---:|
| building_train_1/fairbanks_1 | -0.378666 | -0.266243 | 29.7% | 0.04701 / 0.04707 |
| building_train_2/phoenix_6 | -0.092585 | -0.035602 | 61.5% | 0.09753 / 0.09443 |
| building_train_5/singapore_4 | -0.078865 | -0.019963 | 74.7% | 0.06987 / 0.07056 |

Tuned controller including action validation: 0.406–0.503 ms/step. Timings
include parallel contention and first-step compilation. Defaults won the first
comparison on all three scenarios, so no post-comparison tuning was performed.
These are development horizons, not a 30-day leaderboard score or a claim of
held-out performance. Full results and component totals: `results/comparison.json`.

## Tests and provenance

Synthetic RC mock tests exercise varied sizes, permuted actuator layouts,
determinism, nonfinite/missing observations, invalid parameters and closed-loop
comfort. Real tests run both controllers for 288 steps and skip when the
starter checkout is absent. Synthetic dynamics are not calibrated to the real
plant. `pytest` was added to the supplied venv (it was absent); no starter files
were modified. Its matplotlib dependency emits pyparsing deprecation warnings.

Interface and evaluation conventions follow the Apache-2.0 public starter kit
at `/home/ubuntu/repos/starter-kit`; controller implementation is independent.
