"""Attach geographic reference points to an existing simulated shedding trace.

This is a reporting layer: locations never enter observations or rewards.
Partial shedding is a fraction of zone load, not a list of affected customers.
Residual imbalance is system-wide and cannot be located by this model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

NOTICE = (
    "Simulated controlled shedding only. Points represent aggregate zones, "
    "not affected premises or feeders. Residual imbalance is not located. "
    "Episode times are illustrative, not a real outage forecast."
)


def location_rows(cfg: dict, trace: dict, metadata: dict, policy: str, seed: int) -> list[dict]:
    """One row per zone per hour, including hours with no controlled shedding.

    Match by zone name rather than metadata order to prevent assigning an
    outage to the wrong location after a configuration reorder.
    """
    names = [z["name"] for z in cfg["zones"]]
    if set(names) != set(metadata["locations"]) or len(set(names)) != len(names):
        raise ValueError("Location names must match the configured zones exactly")
    shed = np.asarray(trace["shed"])
    hours = cfg["env"]["episode_hours"]
    if shed.shape != (len(names), hours) or not np.isfinite(shed).all():
        raise ValueError("Invalid shedding trace shape or nonfinite values")
    if np.any((shed < 0) | (shed > 1)):
        raise ValueError("Shed fractions must lie between zero and one")
    if cfg["env"]["timestep_hours"] != 1:
        raise ValueError("Location schedule requires the model's one-hour steps")
    rows = []
    for i, name in enumerate(names):
        point = metadata["locations"][name]
        lat, lon = float(point["latitude"]), float(point["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Invalid coordinates for {name}")
        for h in range(hours):
            fraction = float(shed[i, h])
            rows.append({
                "policy": policy, "episode_seed": seed, "zone": name,
                "reference_place": point["place"], "latitude": lat, "longitude": lon,
                "episode_hour_start": h, "episode_hour_end": h + 1,
                "episode_day": h // 24 + 1, "hour_of_day": h % 24,
                "shed_fraction": fraction,
                "status": ("no controlled shedding" if fraction == 0 else
                           "full zone shedding" if fraction == 1 else "partial zone shedding"),
                "scope": NOTICE, "coordinate_source": metadata["source"],
            })
    return sorted(rows, key=lambda r: (r["episode_hour_start"], names.index(r["zone"])))


def export_locations(cfg: dict, trace: dict, policy: str, seed: int, hour: int,
                     out_dir: Path, locations_path: str = "config/zone_locations.yaml") -> list[Path]:
    """Export the full schedule and a selected hour as GeoJSON and a static plot."""
    with open(locations_path) as f:
        metadata = yaml.safe_load(f)
    rows = location_rows(cfg, trace, metadata, policy, seed)
    selected = [r for r in rows if r["episode_hour_start"] == hour]
    if not selected:
        raise ValueError(f"hour must be between 0 and {cfg['env']['episode_hours'] - 1}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"locations_{policy}_seed{seed}"
    csv_path = out_dir / f"{stem}_schedule.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    features = []
    for row in selected:
        fraction = row["shed_fraction"]
        features.append({
            "type": "Feature",
            # GeoJSON uses longitude FIRST, unlike the CSV's named columns.
            "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
            "properties": {**row, "title": f"{row['zone']}: {fraction:.0%} load shed",
                           "marker-color": "#26824a" if fraction == 0 else
                           "#c0392b" if fraction == 1 else "#c77c00"},
        })
    geo_path = out_dir / f"{stem}_hour{hour}.geojson"
    geo_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    points = ax.scatter([r["longitude"] for r in selected], [r["latitude"] for r in selected],
                        c=[r["shed_fraction"] for r in selected], cmap="YlOrRd",
                        vmin=0, vmax=1, s=180, edgecolors="#333333", zorder=3)
    for row in selected:
        ax.annotate(f"{row['zone']} ({row['reference_place']})\n"
                    f"{row['shed_fraction']:.0%} load shed",
                    (row["longitude"], row["latitude"]), xytext=(10, 9),
                    textcoords="offset points", fontsize=9)
    ax.margins(x=0.6, y=0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.grid(alpha=0.2)
    fig.colorbar(points, ax=ax, shrink=0.65, label="Fraction of zone load shed")
    ax.set_title(f"Simulated zone locations | {policy.upper()} | seed {seed}\n"
                 f"Episode day {hour // 24 + 1}, {hour % 24:02d}:00 "
                 f"to {(hour + 1) % 24:02d}:00")
    fig.text(0.07, 0.025,
             "City reference points, not affected premises or feeder boundaries.\n"
             "Controlled shedding only; residual imbalance is not located.\n"
             "Illustrative episode, not a real outage forecast. Coordinates: GeoNames.", fontsize=8)
    fig.subplots_adjust(bottom=0.15)
    png_path = out_dir / f"{stem}_hour{hour}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [csv_path, geo_path, png_path]
