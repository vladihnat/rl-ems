"""Extract a contiguous block of days from clean/Pyrano1Y_clean.csv.

Generates two aligned files at the maximum shared timestep (15 min, imposed
by the inner-join with the load profile):

  --usage train  →  irradiance_training.csv   +  load_training.csv
  --usage simu   →  irradiance_simulation.csv  +  load_simulation.csv

Usage:
    # Single-window mode (backward-compatible)
    python data/extract_pyrano_simu.py --nbD <int> --usage {train,simu} \
        [--month <1-12>] [--startDate <1-28>]

    # Multi-window mode: repeat --window MONTH START NBDAYS; windows are
    # concatenated in order (must not overlap).
    python data/extract_pyrano_simu.py --usage {train,simu} \
        --window <m> <start> <nbD> [--window <m> <start> <nbD> ...]

Example:
    # 3 days starting Jan 1, for RL training
    python data/extract_pyrano_simu.py --nbD 3 --month 1 --startDate 1 --usage train

    # 4 seasonal windows (Jan/Apr/Jul/Oct), 7 days each, for multi-season training
    python data/extract_pyrano_simu.py --usage train \
        --window 1 1 7 --window 4 1 7 --window 7 1 7 --window 10 1 7

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

# Two options for the load profile : Interpolation or ZOH approaches 
LOAD_CSV   = Path(__file__).resolve().parent / "clean/load_profile_interpol.csv"
# LOAD_CSV   = Path(__file__).resolve().parent / "clean/load_profile_zoh.csv"

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


def _align_one(
    pyrano: pd.DataFrame, load: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join one Pyrano window with the load profile on ``Time`` (15 min).

    Args:
        pyrano: A 5-min Pyrano window (output of :func:`extract_window`).
        load:   The full 15-min load profile DataFrame.

    Returns:
        (irr_df, load_df) — equal-length, row-aligned on ``Time``.

    Raises:
        ValueError: If no timestamp overlaps between the window and the load profile.
    """
    start_ts = pyrano["Time"].iloc[0]
    end_ts   = pyrano["Time"].iloc[-1]
    load_win = load.loc[(load["Time"] >= start_ts) & (load["Time"] <= end_ts)]

    common = pyrano.merge(load_win[["Time"]], on="Time", how="inner")["Time"]
    if common.empty:
        raise ValueError(
            "No overlapping timestamps between Pyrano and load profile. "
            "Check that both share the same Time format and 15-min grid."
        )

    irr_df  = pyrano[pyrano["Time"].isin(common)].reset_index(drop=True)
    load_df = load_win[load_win["Time"].isin(common)].reset_index(drop=True)
    return irr_df, load_df


def extract_aligned(
    irr_out: Path,
    load_out: Path,
    windows: list[tuple[int, int, int]],
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Extract one or more aligned PV + load windows at the shared 15-min step.

    Each window is inner-joined with the load profile, then all windows are
    concatenated (in the given order) into two equal-length, row-aligned CSVs.

    Args:
        irr_out:  Output path for the irradiance CSV.
        load_out: Output path for the load CSV.
        windows:  List of (month, start_date, nb_days) tuples to extract and
                  concatenate. Windows must not overlap in time.

    Returns:
        (n_rows, irr_df, load_df) for the concatenated result.

    Raises:
        FileNotFoundError: If the load profile CSV is missing.
        ValueError: If a window overlaps another or has no matching timestamps.
    """
    if not LOAD_CSV.is_file():
        raise FileNotFoundError(
            f"Load profile not found: {LOAD_CSV}. Run data/clean_load.py first."
        )

    load = pd.read_csv(LOAD_CSV, parse_dates=["Time"])

    irr_parts: list[pd.DataFrame] = []
    load_parts: list[pd.DataFrame] = []
    for month, start_date, nb_days in windows:
        pyrano = extract_window(nb_days, month, start_date)
        irr_df, load_df = _align_one(pyrano, load)
        irr_parts.append(irr_df)
        load_parts.append(load_df)
        print(
            f"  window m{month:02d} d{start_date:02d} x{nb_days}j → "
            f"{len(irr_df)} rows "
            f"[{irr_df['Time'].iloc[0]:%Y-%m-%d %H:%M} → "
            f"{irr_df['Time'].iloc[-1]:%Y-%m-%d %H:%M}]"
        )

    irr_df  = pd.concat(irr_parts, ignore_index=True)
    load_df = pd.concat(load_parts, ignore_index=True)

    if irr_df["Time"].duplicated().any():
        raise ValueError(
            "Overlapping windows: duplicate timestamps detected after concat. "
            "Check that the requested --window ranges do not overlap."
        )
    assert len(irr_df) == len(load_df)

    irr_out.parent.mkdir(parents=True, exist_ok=True)
    irr_df.to_csv(irr_out, index=False)
    load_df.to_csv(load_out, index=False)

    return len(irr_df), irr_df, load_df


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a Unix-style exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        description="Extract aligned irradiance + load CSVs at 15-min resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--usage", choices=["train", "simu"], required=True,
                        help="'train' → irradiance_training.csv + load_training.csv ; "
                             "'simu'  → irradiance_simulation.csv + load_simulation.csv")
    parser.add_argument("--window", action="append", nargs=3, type=int,
                        metavar=("MONTH", "START", "NBDAYS"),
                        help="A window to extract: MONTH(1-12) START(1-28) NBDAYS(>=1). "
                             "Repeatable — windows are concatenated in order. "
                             "Mutually exclusive with --nbD/--month/--startDate.")
    parser.add_argument("--nbD", type=int, default=None,
                        help="Single-window mode: number of consecutive days to extract")
    parser.add_argument("--month", type=int, default=1, choices=range(1, 13),
                        metavar="{1-12}",
                        help="Single-window mode: month where extraction starts")
    parser.add_argument("--startDate", type=int, default=1, choices=range(1, 29),
                        metavar="{1-28}",
                        help="Single-window mode: day of month where extraction starts")
    args = parser.parse_args(argv)

    if args.window and args.nbD is not None:
        parser.error("Use either --window (repeatable) or --nbD/--month/--startDate, not both.")
    if not args.window and args.nbD is None:
        parser.error("Provide at least one --window, or --nbD for single-window mode.")

    if args.window:
        windows = [(m, s, d) for (m, s, d) in args.window]
    else:
        windows = [(args.month, args.startDate, args.nbD)]

    for month, start_date, nb_days in windows:
        if not 1 <= month <= 12:
            parser.error(f"--window MONTH must be in 1-12 (got {month})")
        if not 1 <= start_date <= 28:
            parser.error(f"--window START must be in 1-28 (got {start_date})")
        if nb_days <= 0:
            parser.error(f"--window NBDAYS must be >= 1 (got {nb_days})")

    if args.usage == "train":
        irr_out, load_out = TRAIN_IRR_CSV, TRAIN_LOAD_CSV
    else:
        irr_out, load_out = SIMU_IRR_CSV, SIMU_LOAD_CSV

    print(f"[{args.usage}] extracting {len(windows)} window(s):")
    n_rows, irr_df, load_df = extract_aligned(irr_out, load_out, windows)
    assert len(irr_df) == len(load_df) == n_rows

    start_str  = irr_df["Time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_str    = irr_df["Time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    max_global = irr_df["Global30_kW"].max()
    print(
        f"[{args.usage}] {n_rows} aligned rows total (15 min) | "
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
