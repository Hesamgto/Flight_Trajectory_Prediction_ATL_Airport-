#!/usr/bin/env python3
"""Generate a small synthetic ADS-B-like trajectory for demos and smoke tests.

The paper uses real OpenSky Network ADS-B data for an ATL route, which is
too large to redistribute here. This fabricates a plausible single-flight
trajectory with the same columns (time, lat, lon, heading, velocity,
vertrate, hour) so the pipeline can be run end-to-end without external data.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate_synthetic_trajectory(n_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    t = np.arange(n_rows, dtype="float64")
    unix_start = 1_478_874_137.0
    time = unix_start + t

    # A long-haul-style route drifting steadily northeast (e.g. ATL -> STL).
    lat = 33.64 + 0.02 * t / n_rows * n_rows / 50 + rng.normal(0, 0.0008, n_rows)
    lon = -84.43 - 0.03 * t / n_rows * n_rows / 50 + rng.normal(0, 0.0008, n_rows)
    heading = (310 + 10 * np.sin(t / 500.0) + rng.normal(0, 1.0, n_rows)) % 360
    velocity = 420 + 20 * np.sin(t / 350.0) + rng.normal(0, 3.0, n_rows)
    vertrate = 8 * np.sin(t / 200.0) + rng.normal(0, 0.5, n_rows)
    hour = ((time // 3600) % 24).astype("float64")

    df = pd.DataFrame({
        "time": time, "lat": lat, "lon": lon, "heading": heading,
        "velocity": velocity, "vertrate": vertrate, "hour": hour,
    })

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
