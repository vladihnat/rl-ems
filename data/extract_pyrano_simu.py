"""Extract a contiguous block of days from clean/Pyrano1Y_clean.csv.

Generates two aligned files at the maximum shared timestep (15 min, imposed
by the inner-join with the load profile):

  --usage train  →  irradiance_training.csv   +  load_training.csv
  --usage simu   →  irradiance_simulation.csv  +  load_simulation.csv

Usage:
    python data/extract_pyrano_simu.py --nbD <int> --usage {train,simu} \
        [--month <1-12>] [--startDate <1-28>]

Example:
    # 3 days starting Jan 1, for RL training
    python data/extract_pyrano_simu.py --nbD 3 --month 1 --startDate 1 --usage train

    # Different window for simulation/deployment
    python data/extract_pyrano_simu.py --nbD 3 --month 2 --startDate 1 --usage simu

The start year is taken from the CSV itself (the first matching ``month`` /
``startDate`` is selected); no ``--year`` argument is exposed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "clean/Pyrano1Y_clean.csv"
LOAD_CSV   = Path(__file__).resolve().parent / "clean/load_profile.csv"

TRAIN_IRR_CSV  = Path(__file__).resolve().parent / "irradiance_training.csv"
TRAIN_LOAD_CSV = Path(__file__).resolve().parent / "load_training.csv"
SIMU_IRR_CSV   = Path(__file__).resolve().parent / "irradiance_simulation.csv"
SIMU_LOAD_CSV  = Path(__file__).resolve().parent / "load_simulation.csv"


def extract_window(nb_days: int, month: int, start_date: int) -> pd.DataFrame:
    """Return rows from SOURCE_CSV starting at (month, start_date) for nb_days.

    Args:
        nb_days: Number of consecutive days to extract (>= 1).
        month: Month (1-12) where extraction starts.
        start_date: Day of month (1-28) where extraction starts.

    Returns:
        A DataFrame with all original columns at native 5-min resolution.

    Raises:
        ValueError: If the requested window extends past the end of the CSV.
        FileNotFoundError: If the source CSV is missing.
    """
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV, parse_dates=["Time"])
    times = df["Time"]

    year = int(times.iloc[0].year)
    start_ts = pd.Timestamp(year=year, month=month, day=start_date,
                            hour=0, minute=0, second=0)
    end_ts = start_ts + pd.Timedelta(days=int(nb_days))

    last_ts = times.iloc[-1]
    if end_ts > last_ts + pd.Timedelta(seconds=1):
        raise ValueError(
            f"Requested window exceeds available data: "
            f"end {end_ts:%Y-%m-%d %H:%M} > last available "
            f"{last_ts:%Y-%m-%d %H:%M} in {SOURCE_CSV.name}."
        )

    mask = (times >= start_ts) & (times < end_ts)
    window = df.loc[mask].reset_index(drop=True)

    if window.empty:
        raise ValueError(
            f"No rows match the requested window starting at "
            f"{start_ts:%Y-%m-%d %H:%M}. Check --month/--startDate values."
        )
    return window


def extract_aligned(
    irr_out: Path,
    load_out: Path,
    nb_days: int,
    month: int,
    start_date: int,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Extract aligned PV + load at the maximum shared timestep (15 min).

    Inner-joins the 5-min Pyrano window with the 15-min load profile, keeping
    only timestamps present in both.  Writes two equal-length, row-aligned CSVs.

    Args:
        irr_out: Output path for the irradiance CSV.
        load_out: Output path for the load CSV.
        nb_days: Number of consecutive days to extract.
        month: Month (1-12) where extraction starts.
        start_date: Day of month (1-28) where extraction starts.

    Returns:
        (n_rows, irr_df, load_df) for the matched window.

    Raises:
        FileNotFoundError: If the load profile CSV is missing.
        ValueError: If no timestamp overlaps between the two windows.
    """
    pyrano = extract_window(nb_days, month, start_date)

    if not LOAD_CSV.is_file():
        raise FileNotFoundError(
            f"Load profile not found: {LOAD_CSV}. Run data/clean_load.py first."
        )

    load = pd.read_csv(LOAD_CSV, parse_dates=["Time"])
    start_ts = pyrano["Time"].iloc[0]
    end_ts   = pyrano["Time"].iloc[-1]
    load = load.loc[(load["Time"] >= start_ts) & (load["Time"] <= end_ts)]

    common = pyrano.merge(load[["Time"]], on="Time", how="inner")["Time"]
    if common.empty:
        raise ValueError(
            "No overlapping timestamps between Pyrano and load profile. "
            "Check that both share the same Time format and 15-min grid."
        )

    irr_df  = pyrano[pyrano["Time"].isin(common)].reset_index(drop=True)
    load_df = load[load["Time"].isin(common)].reset_index(drop=True)

    irr_out.parent.mkdir(parents=True, exist_ok=True)
    irr_df.to_csv(irr_out, index=False)
    load_df.to_csv(load_out, index=False)

    return len(common), irr_df, load_df


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a Unix-style exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        description="Extract aligned irradiance + load CSVs at 15-min resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbD", type=int, required=True,
                        help="Number of consecutive days to extract")
    parser.add_argument("--usage", choices=["train", "simu"], required=True,
                        help="'train' → irradiance_training.csv + load_training.csv ; "
                             "'simu'  → irradiance_simulation.csv + load_simulation.csv")
    parser.add_argument("--month", type=int, default=1, choices=range(1, 13),
                        metavar="{1-12}",
                        help="Month where extraction starts")
    parser.add_argument("--startDate", type=int, default=1, choices=range(1, 29),
                        metavar="{1-28}",
                        help="Day of month where extraction starts")
    args = parser.parse_args(argv)

    if args.nbD <= 0:
        parser.error(f"--nbD must be >= 1 (got {args.nbD})")

    if args.usage == "train":
        irr_out, load_out = TRAIN_IRR_CSV, TRAIN_LOAD_CSV
    else:
        irr_out, load_out = SIMU_IRR_CSV, SIMU_LOAD_CSV

    n_rows, irr_df, load_df = extract_aligned(
        irr_out, load_out, args.nbD, args.month, args.startDate
    )
    assert len(irr_df) == len(load_df) == n_rows

    start_str  = irr_df["Time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_str    = irr_df["Time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    max_global = irr_df["Global30_kW"].max()
    print(
        f"[{args.usage}] {n_rows} aligned rows (15 min) | "
        f"start: {start_str} | end: {end_str} | "
        f"max Global30_kW: {max_global:.3f} kW\n"
        f"  → {irr_out.name}\n"
        f"  → {load_out.name}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
