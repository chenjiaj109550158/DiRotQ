#!/usr/bin/env python3
"""Count-weighted Four Over Six stats aggregation for Pilot32 + missing96."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.fouroversix_utils import FourOverSixStats


SCALARS = (
    "m4_count", "m6_count", "block_count", "m4_sse", "m6_sse",
    "adaptive_sse", "m6_gscale2688_sse", "m6_gscale1536_sse", "e0_sse",
    "e2_tile_sse", "e0_tile_sse", "tile_selected_sse", "e0_tile_count",
    "e2_tile_count", "tile_count", "m4_saturation_count",
    "m6_saturation_count", "selected_saturation_count", "valid_value_count",
    "signal_energy", "reconstruction_sse",
)
VECTORS = ("m4_occupancy", "m6_occupancy")


def raw(row: dict) -> dict:
    return {key: row[key] for key in SCALARS + VECTORS}


def add(left: dict, right: dict) -> dict:
    output = {key: float(left[key]) + float(right[key]) for key in SCALARS}
    for key in VECTORS:
        if len(left[key]) != len(right[key]):
            raise RuntimeError(f"{key}: incompatible histogram lengths")
        output[key] = [float(a) + float(b) for a, b in zip(left[key], right[key])]
    return output


def derived(row: dict) -> dict:
    return FourOverSixStats._derived(raw(row))


def aggregate_named(first: dict, latter: dict) -> dict:
    if set(first) != set(latter):
        raise RuntimeError("per-layer key mismatch between Pilot32 and missing96")
    return {name: derived(add(first[name], latter[name])) for name in sorted(first)}


def timestep_map(rows: list[dict]) -> dict:
    output = {}
    for row in rows:
        key = (row["layer"], int(row["timestep_index"]), float(row["timestep"]))
        if key in output:
            raise RuntimeError(f"duplicate layer/timestep row: {key}")
        output[key] = row
    return output


def aggregate_timesteps(first: list[dict], latter: list[dict]) -> list[dict]:
    first_map, latter_map = timestep_map(first), timestep_map(latter)
    if set(first_map) != set(latter_map):
        raise RuntimeError("per-layer/timestep key mismatch between segments")
    return [
        {
            "layer": key[0], "timestep_index": key[1], "timestep": key[2],
            **derived(add(first_map[key], latter_map[key])),
        }
        for key in sorted(first_map, key=lambda item: (item[0], item[1]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first32", type=Path, required=True)
    parser.add_argument("--new96", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first32.read_text())
    latter = json.loads(args.new96.read_text())
    if first["selection_unit"] != latter["selection_unit"]:
        raise RuntimeError("selection-unit mismatch")
    combined_raw = add(first, latter)
    result = {
        "aggregation": "raw counters and SSE sums; ratios are recomputed, never averaged",
        "selection_unit": first["selection_unit"],
        "segments": {
            "indices_0_31": derived(first),
            "indices_32_127": derived(latter),
            "indices_0_127": FourOverSixStats._derived(combined_raw),
        },
        "per_layer_indices_0_127": aggregate_named(first["per_layer"], latter["per_layer"]),
        "per_layer_timestep_indices_0_127": aggregate_timesteps(
            first["per_layer_timestep"], latter["per_layer_timestep"]
        ),
        "sources": {"first32": str(args.first32), "new96": str(args.new96)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["segments"], indent=2))


if __name__ == "__main__":
    main()
