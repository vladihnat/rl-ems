"""Extract a contiguous block of days from Pyrano1Y_clean.csv into pyrano_simu.csv.

Simulates a "deployment" dataset for the agent: it sees data it has never been
trained on, mirroring real-world rollout conditions. The output file
``data/pyrano_simu.csv`` is overwritten on every call.

Usage:
    python data/extract_pyrano_simu.py --nbD <int> [--month <1-12>] [--startDate <1-28>]

Example:
    # 3 consecutive days starting on January 1
    python data/extract_pyrano_simu.py --nbD 3 --month 1 --startDate 1

The start year is taken from the CSV itself (the first matching ``month`` /
``startDate`` is selected); no ``--year`` argument is exposed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(__file__).resolve().parent / "Pyrano1Y_clean.csv"
TARGET_CSV = Path(__file__).resolve().parent / "pyrano_simu.csv"


def extract_window(nb_days: int, month: int, start_date: int) -> pd.DataFrame:
    """Return rows from SOURCE_CSV starting at (month, start_date) for nb_days.

    Args:
        nb_days: Number of consecutive days to extract (>= 1).
        month: Month (1-12) where extraction starts.
        start_date: Day of month (1-28) where extraction starts.

    Returns:
        A DataFrame with all original columns intact (no rename, no add).

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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a Unix-style exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        description="Extract a contiguous block of days from Pyrano1Y_clean.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbD", type=int, required=True,
                        help="Number of consecutive days to extract")
    parser.add_argument("--month", type=int, default=1, choices=range(1, 13),
                        metavar="{1-12}",
                        help="Month where extraction starts")
    parser.add_argument("--startDate", type=int, default=1, choices=range(1, 29),
                        metavar="{1-28}",
                        help="Day of month where extraction starts")
    args = parser.parse_args(argv)

    if args.nbD <= 0:
        parser.error(f"--nbD must be >= 1 (got {args.nbD})")

    window = extract_window(args.nbD, args.month, args.startDate)

    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(TARGET_CSV, index=False)

    start_str = window["Time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_str = window["Time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    print(
        f"Extracted {len(window)} rows | start: {start_str} | end: {end_str} "
        f"→ {TARGET_CSV.relative_to(Path.cwd()) if TARGET_CSV.is_relative_to(Path.cwd()) else TARGET_CSV}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
