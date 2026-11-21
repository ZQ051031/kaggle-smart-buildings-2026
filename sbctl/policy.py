"""NumPy-only controller; all physical temperatures are Kelvin."""

import json
from pathlib import Path
import re

import numpy as np

DEFAULT_PARAMS = dict(
    kp=0.07,
    ki=0.0035,
    integral_limit=1.0,
    prestart_hours=2.0,
    cooling_hold_steps=288.0,
    heating_setback=2.0,
    damper_gain=1.0,
    reheat_gain=1.0,
    fan_base=0.1,
    fan_gain=0.7,
    sat_base=295.0,
    sat_gain=10.0,
    water_base=0.0,
    water_gain=1.0,
    on_threshold=0.03,
    econ_advantage=2.0,
)
PARAM_BOUNDS = dict(
    kp=(0.01, 0.3),
    ki=(0.0001, 0.02),
    integral_limit=(0.2, 2.0),
    prestart_hours=(0.0, 4.0),
    cooling_hold_steps=(0.0, 576.0),
    heating_setback=(0.0, 5.0),
    damper_gain=(0.2, 3.0),
    reheat_gain=(0.2, 3.0),
    fan_base=(0.1, 0.5),
    fan_gain=(0.0, 1.0),
    sat_base=(290.0, 300.0),
    sat_gain=(0.0, 20.0),
    water_base=(0.0, 1.0),
    water_gain=(0.0, 2.0),
    on_threshold=(0.0, 0.3),
    econ_advantage=(0.0, 10.0),
)


class TunedTransferPolicy:
    """Coordinate zone PI, AHU air-side reset, and plant water reset loops."""

    name = "tuned"

    def __init__(self, building_meta_data, action_spec=None, params=None):
        metadata = self.metadata = building_meta_data
        self.params = DEFAULT_PARAMS.copy()
        path = Path(__file__).with_name("params.json")
        if path.exists():
            self.params.update(json.loads(path.read_text()))
        self.params.update(params or {})
        for key, value in self.params.items():
            if key not in PARAM_BOUNDS or not np.isfinite(value):
                raise ValueError(f"Invalid parameter: {key}")
        self.n_continuous = int(metadata["n_continuous_action"])
        self.n_discrete = int(metadata["n_discrete_action"])
        self.temperature_indices = np.array(metadata.get("temp_indices", []), dtype=int)
        self.pressure_indices = np.array(
            metadata.get("pressure_indices", []), dtype=int
        )
        self.recirculation_indices = np.array(
            metadata.get("recirculation_indices", []), dtype=int
        )
        self.water_indices = np.array(metadata.get("water_indices", []), dtype=int)
        self.n_ahus = len(self.temperature_indices)
        names = metadata.get("observation_field_names", [])
        self.fields = {n: i for i, n in enumerate(names)}
        self.zones = list(metadata.get("zone_action_indices", {}))
        self.zone_temperature_indices = np.array(
            [
                self.fields.get(zone_id + "_zone_air_temperature_sensor", -1)
                for zone_id in self.zones
            ]
        )
        self.heating_sensor_indices = np.array(
            [
                self.fields.get(
                    zone_id + "_zone_air_heating_temperature_setpoint_sensor", -1
                )
                for zone_id in self.zones
            ]
        )
        self.cooling_sensor_indices = np.array(
            [
                self.fields.get(
                    zone_id + "_zone_air_cooling_temperature_setpoint_sensor", -1
                )
                for zone_id in self.zones
            ]
        )
        self.damper_indices = np.array(
            [
                metadata["zone_action_indices"][zone_id].get(
                    "supply_air_damper_percentage_command", -1
                )
                for zone_id in self.zones
            ],
            dtype=int,
        )
        self.reheat_indices = np.array(
            [
                metadata["zone_action_indices"][zone_id].get(
                    "reheat_valve_percentage_command", -1
                )
                for zone_id in self.zones
            ],
            dtype=int,
        )
        assignments = metadata.get("ahu_zone_assignments", {})
        ahus = sorted(
            assignments,
            key=lambda s: int(re.search(r"\d+", s)[0]) if re.search(r"\d+", s) else 0,
        )
        self.groups = [
            np.array(
                [
                    j
                    for j, zone_id in enumerate(self.zones)
                    if zone_id
                    in [r.replace("room_", "VAV_") for r in assignments[ahu_index]]
                ],
                dtype=int,
            )
            for ahu_index in ahus
        ]
        self.outside_temperature_indices = [
            self.fields.get("ahs_" + ahu_index + "_outside_air_temperature_sensor", -1)
            for ahu_index in ahus
        ]
        norm = metadata.get("observation_normalization_metadata", {}).get(
            "zone_air_temperature_sensor", {}
        )
        self.mean = norm.get("sample_mean", 295.0)
        self.std = norm.get("sample_std", 5.0)
        self.integral = np.zeros(len(self.zones))
        self.hold = np.zeros(self.n_ahus)

    @staticmethod
    def _read(obs, indices, default):
        """Read indexed sensors with finite fallbacks for absent measurements."""
        indices = np.asarray(indices, dtype=int)
        result = np.full(indices.shape, default, dtype=float)
        valid = (indices >= 0) & (indices < len(obs))
        result[valid] = obs[indices[valid]]
        return np.where(np.isfinite(result), result, default)

    def get_action(self, obs):
        """Track zone band centers with anti-windup PI.

        AHUs reset supply temperature and fan effort from zone demand, holding
        cooling on to avoid cycling. The plant resets water from peak reheat.
        """
        obs = np.asarray(obs).reshape(-1)
        parameters = self.params
        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        def field(name, default):
            return self._read(obs, [self.fields.get(name, -1)], default)[0]

        hour = (
            (np.arctan2(field("hod_sin_000", 0), field("hod_cos_000", 1)) % (2 * np.pi))
            * 24
            / (2 * np.pi)
        )
        day = (
            int(
                round(
                    (
                        np.arctan2(field("dow_sin_000", 0), field("dow_cos_000", 1))
                        % (2 * np.pi)
                    )
                    * 7
                    / (2 * np.pi)
                )
            )
            % 7
        )
        morning = self.metadata.get("morning_start_hour", 8)
        evening = self.metadata.get("evening_start_hour", 18)
        occupied = (
            field("comfort_mode_now", float(day < 5 and morning <= hour < evening))
            > 0.5
        )
        soon = field("comfort_mode_soon", 0) > 0.5
        # Sensor bands and comfort flags are authoritative for holidays/time zones.
        scheduled = (
            occupied
            or soon
            or (day < 5 and morning - parameters["prestart_hours"] <= hour < morning)
        )
        low, high = self.metadata.get("comfort_temp_window", (294.0, 297.0))
        heating_band = self._read(obs, self.heating_sensor_indices, low)
        cooling_band = self._read(obs, self.cooling_sensor_indices, high)
        valid = cooling_band > heating_band
        heating_band = np.where(valid, heating_band, low)
        cooling_band = np.where(valid, cooling_band, high)
        heating_band = np.maximum(heating_band, low - parameters["heating_setback"])
        cooling_band = np.minimum(cooling_band, high)
        if scheduled:
            heating_band = np.full_like(heating_band, low)
        target = (heating_band + cooling_band) / 2
        zone_temperature = (
            self._read(
                obs,
                self.zone_temperature_indices,
                (np.mean(target) - self.mean) / self.std if len(target) else 0,
            )
            * self.std
            + self.mean
        )
        error = target - zone_temperature
        candidate_integral = np.clip(
            self.integral + parameters["ki"] * error,
            -parameters["integral_limit"],
            parameters["integral_limit"],
        )
        unclipped_demand = parameters["kp"] * error + candidate_integral
        self.integral = np.where(
            (np.abs(unclipped_demand) <= 1) | (error * unclipped_demand < 0),
            candidate_integral,
            self.integral,
        )
        zone_demand = np.clip(parameters["kp"] * error + self.integral, -1, 1)
        damper = np.clip(np.abs(zone_demand) * parameters["damper_gain"], 0, 1)
        reheat = np.clip(zone_demand * parameters["reheat_gain"], 0, 1)
        for indices, values in (
            (self.damper_indices, damper),
            (self.reheat_indices, reheat),
        ):
            mask = indices >= 0
            continuous[indices[mask]] = 2 * values[mask] - 1
        for ahu_index in range(self.n_ahus):
            group = (
                self.groups[ahu_index]
                if ahu_index < len(self.groups)
                else np.array([], dtype=int)
            )
            demand = float(np.mean(zone_demand[group])) if len(group) else 0.0
            effort = float(np.max(np.abs(zone_demand[group]))) if len(group) else 0.0
            if demand < -parameters["on_threshold"]:
                self.hold[ahu_index] = parameters["cooling_hold_steps"]
            active = (
                scheduled
                or effort > parameters["on_threshold"]
                or self.hold[ahu_index] > 0
            )
            self.hold[ahu_index] = max(0, self.hold[ahu_index] - 1)
            if ahu_index < self.n_discrete - 1:
                discrete[ahu_index] = active
            supply_temperature = np.clip(
                parameters["sat_base"] + parameters["sat_gain"] * demand, 285, 305
            )
            continuous[self.temperature_indices[ahu_index]] = (
                supply_temperature - 285
            ) / 10 - 1
            if ahu_index < len(self.pressure_indices):
                continuous[self.pressure_indices[ahu_index]] = (
                    2
                    * np.clip(
                        parameters["fan_base"] + parameters["fan_gain"] * effort, 0, 1
                    )
                    - 1
                )
            if ahu_index < len(self.recirculation_indices):
                recirculation = 0.7
                outside = self._read(
                    obs,
                    [
                        self.outside_temperature_indices[ahu_index]
                        if ahu_index < len(self.outside_temperature_indices)
                        else -1
                    ],
                    295,
                )[0]
                inside = (
                    float(np.mean(zone_temperature[group])) if len(group) else 295.0
                )
                if demand < 0 and inside - outside > parameters["econ_advantage"]:
                    recirculation = np.clip(
                        1 - (inside - supply_temperature) / (inside - outside), 0, 0.7
                    )
                continuous[self.recirculation_indices[ahu_index]] = (
                    2 * recirculation - 1
                )
        heating_demand = float(np.max(reheat)) if len(reheat) else 0.0
        continuous[self.water_indices] = (
            2
            * np.clip(
                parameters["water_base"] + parameters["water_gain"] * heating_demand,
                0,
                1,
            )
            - 1
        )
        if len(self.pressure_indices) > self.n_ahus:
            continuous[self.pressure_indices[self.n_ahus :]] = (
                1 if heating_demand > parameters["on_threshold"] else -1
            )
        if self.n_discrete:
            discrete[-1] = heating_demand > parameters["on_threshold"]
        return dict(
            continuous_action=np.clip(continuous, -1, 1), discrete_action=discrete
        )
