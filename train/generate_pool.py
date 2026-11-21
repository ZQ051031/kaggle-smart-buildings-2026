"""Generate small procedural buildings; keep all shipped scenarios held out."""

import argparse
import json
from pathlib import Path
import random

from sbctl.sim import starter_cwd


def generate_pool(output, count=12, start_seed=12000):
    """Cross each accepted geometry with one recorded strip per station."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with starter_cwd() as root:
        from smart_buildings.run_floor_generator import (
            build_config,
            save_config,
            DEFAULT_RANGES,
        )
        from smart_buildings.smart_control.utils.materials import draw_materials
        from smart_buildings.smart_control.utils.scenario_generator import (
            check_zone_floor_areas,
            scenario_config_from_dict,
            list_recorded_weather_presets,
            load_scenario_from_parts,
            generate_scenario_from_config,
        )

        original_ranges = DEFAULT_RANGES.copy()
        DEFAULT_RANGES.update(
            width=(300, 400), height=(180, 250), num_floors=(2, 3), num_ahus=(3, 7)
        )
        materials = draw_materials(random.Random(start_seed), count)
        weather_by_station = {}
        for weather in sorted(list_recorded_weather_presets()):
            weather_by_station.setdefault(weather.rsplit("_", 1)[0], []).append(weather)
        shipped = sorted(
            (root / "smart_buildings/training_scenarios").glob("*/*/config.gin")
        )
        existing_seeds = set()
        import yaml

        for path in (root / "smart_buildings/buildings").glob("*.yaml"):
            existing_seeds.add(yaml.safe_load(path.read_text()).get("seed"))
        scenarios = []
        buildings = []
        rejected = []
        seed = start_seed
        try:
            while len(buildings) < count:
                if seed >= start_seed + count * 100:
                    raise RuntimeError("Too many rejected geometries")
                current_seed = seed
                seed += 1
                if current_seed in existing_seeds:
                    continue
                config = build_config(
                    current_seed,
                    f"generated_{current_seed}",
                    "composite",
                    materials[len(buildings)],
                )
                try:
                    check_zone_floor_areas(scenario_config_from_dict(config))
                except ValueError as error:
                    if "area" not in str(error).lower():
                        raise
                    rejected.append(dict(seed=current_seed, reason=str(error)))
                    continue
                building_path = save_config(
                    config, output / "buildings", save_pdf=False, show=False
                )[0]
                building_info = dict(
                    seed=current_seed, config=str(building_path), scenarios=[]
                )
                for station, weathers in sorted(weather_by_station.items()):
                    weather = weathers[len(buildings) % len(weathers)]
                    merged = load_scenario_from_parts(
                        str(building_path),
                        weather,
                        output_base_dir=str(output / "scenarios"),
                    )
                    result = generate_scenario_from_config(merged)
                    path = str(Path(result["output_dir"]).resolve())
                    scenarios.append(path)
                    building_info["scenarios"].append(
                        dict(
                            path=path,
                            station=station,
                            weather=weather,
                            zones=result["num_rooms"],
                            ahus=result["num_ahus"],
                        )
                    )
                buildings.append(building_info)
                print(
                    f"Accepted {len(buildings)}/{count}: seed {current_seed}",
                    flush=True,
                )
        finally:
            DEFAULT_RANGES.clear()
            DEFAULT_RANGES.update(original_ranges)
        validation = [str(path.parent.resolve()) for path in shipped]
        # Fixed station-stratified promotion subset, never part of the ES objective.
        selected = []
        stations = sorted({Path(path).name.rsplit("_", 1)[0] for path in validation})
        for index, station in enumerate(stations):
            choices = [
                path
                for path in validation
                if Path(path).name.rsplit("_", 1)[0] == station
            ]
            selected.append(choices[index % len(choices)])
        manifest = dict(
            seed=start_seed,
            training=scenarios,
            validation=validation,
            promotion=selected,
            buildings=buildings,
            rejected=rejected,
        )
        (output / "pool.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name, values in [
            ("training", scenarios),
            ("validation", validation),
            ("promotion", selected),
        ]:
            (output / f"{name}.json").write_text(json.dumps(values, indent=2) + "\n")
        print(
            f"{len(scenarios)} training / {len(validation)} held-out / {len(selected)} promotion",
            flush=True,
        )
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start-seed", type=int, default=12000)
    args = parser.parse_args()
    generate_pool(args.output, args.count, args.start_seed)


if __name__ == "__main__":
    main()
