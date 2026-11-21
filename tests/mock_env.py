"""Synthetic RC building, deliberately not a substitute for real evaluation."""

import numpy as np


class MockEnv:
    def __init__(self, k=2, v=6, seed=0, permuted=False):
        rng = np.random.default_rng(seed)
        n = 3 * k + 2 + 2 * v
        order = rng.permutation(n) if permuted else np.arange(n)
        zones = [f"VAV_{j + 1}" for j in range(v)]
        names = []
        for z in zones:
            names += [
                z + s
                for s in (
                    "_zone_air_temperature_sensor",
                    "_zone_air_heating_temperature_setpoint_sensor",
                    "_zone_air_cooling_temperature_setpoint_sensor",
                )
            ]
        names += [f"ahs_ahu_{a + 1}_outside_air_temperature_sensor" for a in range(k)]
        names += [
            "comfort_mode_now",
            "comfort_mode_soon",
            "hod_sin_000",
            "hod_cos_000",
            "dow_sin_000",
            "dow_cos_000",
        ]
        self.metadata = dict(
            n_continuous_action=n,
            n_discrete_action=k + 1,
            temp_indices=list(order[np.arange(k) * 3]),
            pressure_indices=list(order[list(np.arange(k) * 3 + 1) + [3 * k + 1]]),
            recirculation_indices=list(order[np.arange(k) * 3 + 2]),
            water_indices=[int(order[3 * k])],
            zone_action_indices={
                z: dict(
                    supply_air_damper_percentage_command=int(order[3 * k + 2 + 2 * j]),
                    reheat_valve_percentage_command=int(order[3 * k + 3 + 2 * j]),
                )
                for j, z in enumerate(zones)
            },
            ahu_zone_assignments={
                f"ahu_{a + 1}": [f"room_{j + 1}" for j in range(v) if j % k == a]
                for a in range(k)
            },
            observation_field_names=names,
            zone_temp_indices=list(np.arange(v) * 3),
            observation_normalization_metadata={
                "zone_air_temperature_sensor": dict(
                    sample_mean=295.0, sample_std=5.0, sample_variance=25.0
                ),
                "outside_air_temperature_sensor": dict(
                    sample_mean=295.0, sample_std=10.0, sample_variance=100.0
                ),
            },
            comfort_temp_window=(294.0, 297.0),
            eco_temp_window=(290.0, 301.0),
            morning_start_hour=8,
            evening_start_hour=18,
            holidays=[],
            time_zone="UTC",
        )
        self.k, self.v = k, v
        self.temp = np.full(v, 294.0)
        self.steps = 0

    def obs(self):
        result = []
        for t in self.temp:
            result += [(t - 295) / 5, 294.0, 297.0]
        result += [281.0] * self.k + [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        return np.array(result, dtype=np.float32)

    def step(self, action):
        m = self.metadata
        c, d = action["continuous_action"], action["discrete_action"]
        for j, z in enumerate(m["zone_action_indices"]):
            a = j % self.k
            idx = m["zone_action_indices"][z]
            damper = max(0.1, (c[idx["supply_air_damper_percentage_command"]] + 1) / 2)
            valve = (c[idx["reheat_valve_percentage_command"]] + 1) / 2
            sat = 295 + 10 * c[m["temp_indices"][a]]
            flow = d[a] * damper
            self.temp[j] += (
                0.001 * (281 - self.temp[j])
                + 0.03 * flow * (sat - self.temp[j])
                + 0.5 * flow * valve * d[-1]
            )
        deviation = np.maximum(294 - self.temp, 0) + np.maximum(self.temp - 297, 0)
        comfort = (
            np.where(
                deviation <= 5, deviation / 5, 1 + ((deviation - 5) / 5) ** 2
            ).mean()
            if self.v
            else 0
        )
        energy = float(np.mean((c + 1) / 2)) * 0.05
        self.steps += 1
        return self.obs(), -(0.6 * comfort + 0.2 * energy + 0.2 * energy)
