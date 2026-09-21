#!/usr/bin/env python3
"""Generate a small synthetic ADS-B-like trajectory for demos and smoke tests.

The real experiments in the paper use historical OpenSky Network ADS-B data
for Hartsfield-Jackson Atlanta International Airport (ATL), which is too
large to redistribute here. This script fabricates a smooth, physically
plausible single-flight trajectory with the same columns so the pipeline in
this repo can be run end-to-end without external data.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate_synthetic_trajectory(n_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    t = np.arange(n_rows, dtype="float64")
    unix_start = 1_478_874_137.0
    time = unix_start + t  # ~1 Hz ADS-B pings

    # Smooth lat/lon drift (a gentle descent path) plus small sensor noise.
    lat = 33.6407 + 0.15 * np.sin(t / 400.0) + rng.normal(0, 0.0005, n_rows)
    lon = -84.4277 + 0.15 * np.cos(t / 400.0) + rng.normal(0, 0.0005, n_rows)
    heading = (90 + 30 * np.sin(t / 250.0) + rng.normal(0, 1.0, n_rows)) % 360
    velocity = 230 + 15 * np.sin(t / 300.0) + rng.normal(0, 2.0, n_rows)
    vertrate = 5 * np.sin(t / 150.0) + rng.normal(0, 0.5, n_rows)
    hour = ((time // 3600) % 24).astype("float64")

    df = pd.DataFrame({
        "time": time, "lat": lat, "lon": lon, "heading": heading,
        "velocity": velocity, "vertrate": vertrate, "hour": hour,
    })

    # Sprinkle in a few missing values, mirroring real ADS-B dropouts, to
    # exercise the cubic-spline interpolation step.
    for col in ["lat", "lon", "velocity"]:
        drop_idx = rng.choice(n_rows, size=max(1, n_rows // 200), replace=False)
        df.loc[drop_idx, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/sample_flight_data.csv")
    args = parser.parse_args()

    df = generate_synthetic_trajectory(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
