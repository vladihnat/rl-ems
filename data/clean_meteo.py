"""Clean meteorological irradiance CSV files from pyranometer exports."""

import math
import pandas as pd
from pathlib import Path

# Mapping from raw column names to output names
COLUMN_RENAME = {
    "Diffus":         "Diffus_kW",
    "Global 0°":      "Global0_kW",
    "Global 45°":     "Global45_kW",
    "Global 60°":     "Global60_kW",
    "Global 30°":     "Global30_kW",
    "ClearSky 0° F":  "ClearSky0_kW",
}

OUTPUT_COLUMNS = [
    "Time", "date",
    "hour", "hour_sin", "hour_cos",
    "timestep", "timestep_sin", "timestep_cos",
    "day_of_year", "doy_sin", "doy_cos",
    "Diffus_kW", "Global0_kW", "Global45_kW",
    "Global60_kW", "Global30_kW", "ClearSky0_kW",
]


def _parse_irradiance(value: str) -> float:
    """Convert a single irradiance string cell to kW/m².

    Handles fW/m², µW/m², mW/m², W/m², and kW/m² suffixes.
    Longer/more-specific suffixes are checked first to avoid partial matches
    (e.g. mW/m² must be matched before W/m²).
    """
    s = str(value).strip()
    if "fW/m²" in s:   # femtowatts → kW: ÷ 1e18
        return float(s.replace("fW/m²", "").strip()) / 1e18
    if "nW/m²" in s:   # nanowatts → kW: ÷ 1e12
        return float(s.replace("nW/m²", "").strip()) / 1e12
    if "µW/m²" in s:   # microwatts → kW: ÷ 1e9
        return float(s.replace("µW/m²", "").strip()) / 1e9
    if "mW/m²" in s:   # milliwatts → kW: ÷ 1e6
        return float(s.replace("mW/m²", "").strip()) / 1_000_000
    if "kW/m²" in s:   # kilowatts → kW: ÷ 1
        return float(s.replace("kW/m²", "").strip())
    if "W/m²" in s:    # watts → kW: ÷ 1e3
        return float(s.replace("W/m²", "").strip()) / 1_000
    return float(s)


def _sin_cos(series: pd.Series, period: float):
    theta = 2 * math.pi * series / period
    return theta.apply(math.sin), theta.apply(math.cos)


def clean_file(input_path: str | Path, output_path: str | Path) -> None:
    """Clean a single pyranometer CSV file and write the result.

    Steps performed:
    - Parse the Time column to datetime.
    - Derive calendar columns (date, hour, timestep, day_of_year).
    - Encode periodic features as sin/cos pairs.
    - Convert all irradiance values from W/m² or mW/m² strings to float kW/m².
    - Rename columns to standardised names.
    - Write the cleaned CSV with a fixed column order.

    Parameters
    ----------
    input_path:
        Path to the raw CSV file.
    output_path:
        Destination path for the cleaned CSV file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # --- Time column ---------------------------------------------------------
    df["Time"] = pd.to_datetime(df["Time"])

    df["date"] = df["Time"].dt.date.astype(str)
    df["hour"] = df["Time"].dt.hour
    # df["timestep"] = df["Time"].dt.minute // 5
    df["day_of_year"] = df["Time"].dt.day_of_year

    t0 = df["Time"].iloc[0]
    t_hours = (df["Time"] - t0).dt.total_seconds() / 3600.0

    df["hour_sin"], df["hour_cos"] = _sin_cos(t_hours, 24)
    # Uncomment if sin_cos timestep encoding is desired :
    # df["timestep_sin"], df["timestep_cos"] = _sin_cos(df["timestep"], 12)
    df["doy_sin"], df["doy_cos"] = _sin_cos(t_hours, 24 * 365.2425)

    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- Numeric columns -----------------------------------------------------
    irradiance_cols = [c for c in COLUMN_RENAME if c in df.columns]
    for col in irradiance_cols:
        df[col] = df[col].apply(_parse_irradiance).astype("float64")

    df = df.rename(columns=COLUMN_RENAME)

    # --- ClearSky0_kW may be absent (e.g. yearly file) ----------------------
    if "ClearSky0_kW" not in df.columns:
        df["ClearSky0_kW"] = float("nan")

    df = df[[c for c in OUTPUT_COLUMNS if c in df.columns]]

    df.to_csv(output_path, index=False)

    # --- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"File : {input_path.name}  →  {output_path.name}")
    print(f"Rows : {len(df)}")
    print(f"Date range : {df['date'].min()}  →  {df['date'].max()}")
    print(f"Time range : {df['Time'].iloc[0]}  →  {df['Time'].iloc[-1]}")


FILES = [
    ("Pyrano1w.csv",  "Pyrano1w_clean.csv"),
    ("Pyrano1M.csv",  "Pyrano1M_clean.csv"),
    ("Pyrano1Y.csv",  "Pyrano1Y_clean.csv"),
]

if __name__ == "__main__":
    base = Path(__file__).parent
    for src, dst in FILES:
        clean_file(base / src, base / dst)
    print("\nDone.")
