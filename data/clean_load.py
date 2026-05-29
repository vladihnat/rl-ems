"""Clean and resample the realistic load profile for RL training.

Reads ``data/raw/energy_community.csv`` (hourly kW, ``;``-separated, year 2025) and
produces ``data/clean/load_profile.csv`` resampled to ``DELTA_T`` with a ``Conso_tot``
column, in the same ``Time`` format as the Pyranometer data (year 2023).

Run directly:
    python data/clean_load.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "raw/energy_community.csv"
TARGET_CSV = Path(__file__).resolve().parent / "clean/load_profile.csv"

# Resampling step. The load is upsampled from 1h to this resolution; it must be a
# divisor of 60 min and match the timestep used elsewhere (e.g. the Pyrano grid).
DELTA_T = "15min"

# Year used by the Pyrano data — the load calendar is shifted onto it so the two
# datasets share identical timestamps and can be joined on "Time".
TARGET_YEAR = 2023

CONSO_COLS = ["Conso_resident", "Conso_bureau", "Conso_ecole"]


def clean_load() -> pd.DataFrame:
    """Return the cleaned, resampled load profile (indexed by Time)."""
    df = pd.read_csv(SOURCE_CSV, sep=";", parse_dates=["date"], dayfirst=True)

    df = df.drop(columns=["prod_pv"])
    df["date"] = df["date"].apply(lambda d: d.replace(year=TARGET_YEAR))
    df = df.rename(
        columns={
            "date": "Time",
            "conso_residentiel": "Conso_resident",
            "conso_bureau": "Conso_bureau",
            "conso_ecole": "Conso_ecole",
        }
    )

    df = df.set_index("Time").sort_index()

    # Zero-order hold: each hourly kW is repeated across the 4 quarter-hour slots,
    # so every hour's average is exactly its original value and energy is preserved
    # with zero error.
    resampled = df.resample(DELTA_T).ffill()

    resampled["Conso_tot"] = resampled[CONSO_COLS].sum(axis=1)
    resampled.index.name = "Time"
    return resampled, df


def main() -> int:
    resampled, original = clean_load()

    resampled.to_csv(TARGET_CSV, index_label="Time")

    # Energy sanity check: Σ kW·Δt should be preserved within a small epsilon.
    dt_h_new = pd.Timedelta(DELTA_T).total_seconds() / 3600.0
    energy_new = resampled["Conso_tot"].sum() * dt_h_new
    energy_orig = original[CONSO_COLS].sum(axis=1).sum() * 1.0  # original step = 1h
    rel_err = abs(energy_new - energy_orig) / energy_orig if energy_orig else 0.0

    out = TARGET_CSV.relative_to(Path.cwd()) if TARGET_CSV.is_relative_to(Path.cwd()) else TARGET_CSV
    print(
        f"Wrote {len(resampled)} rows ({DELTA_T}) | "
        f"{resampled.index[0]:%Y-%m-%d %H:%M} -> {resampled.index[-1]:%Y-%m-%d %H:%M} | "
        f"energy orig {energy_orig:.1f} kWh, resampled {energy_new:.1f} kWh "
        f"(rel. err {rel_err:.4%}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
