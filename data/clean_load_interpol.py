"""Clean and resample the realistic load profile for RL training (interpolation).

Reads ``data/raw/energy_community.csv`` (hourly kW, ``;``-separated, year 2025) and
produces ``data/clean/load_profile_interpol.csv`` resampled to ``DELTA_T`` with a
``Conso_tot`` column, in the same ``Time`` format as the Pyranometer data (year 2023).

Upsampling strategy: linear interpolation between consecutive hourly values, which
produces a smooth profile (unlike the ZOH / forward-fill of clean_load.py).

Run directly:
    python data/clean_load_interpol.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "raw/energy_community.csv"
TARGET_CSV = Path(__file__).resolve().parent / "clean/load_profile_interpol.csv"

DELTA_T = "15min"
TARGET_YEAR = 2023

CONSO_COLS = ["Conso_resident", "Conso_bureau", "Conso_ecole"]


def clean_load_interpol() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned, interpolated load profile (indexed by Time)."""
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

    # Linear interpolation: new 15-min slots are filled by interpolating between the
    # two surrounding hourly values, producing a smooth transition rather than a step.
    resampled = df.resample(DELTA_T).interpolate(method="linear")

    resampled["Conso_tot"] = resampled[CONSO_COLS].sum(axis=1)
    resampled.index.name = "Time"
    return resampled, df


def main() -> int:
    resampled, original = clean_load_interpol()

    resampled.to_csv(TARGET_CSV, index_label="Time")

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
