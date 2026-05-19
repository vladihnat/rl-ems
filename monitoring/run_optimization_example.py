"""Run the trained RL agent on one concrete optimization example (e.g. a 24 h day).

This script simulates real-world deployment: load the saved policy, walk the
env step-by-step (the env's `monitoring_table` is auto-populated on each
`step()`), dump the resulting monitoring table to CSV, and open the two
interactive matplotlib windows.

Usage:
    python -m monitoring.run_optimization_example \
        --config configs/exp01_perfect_foresight.yaml \
        --model  results/exp01_perfect_foresight/sac_model.zip \
        --forecast data/Pyrano1w_clean.csv \
        --out monitoring/runs/monitoring_table.csv

The forecast CSV represents what the agent "sees" (observations); the
ground-truth CSV (optional, defaults to the forecast) carries the values used
to overlay the forecast-error shading on the comparison plot.

Variable-name mapping between RL code and MATLAB conventions:
    RL code              MATLAB / monitoTable
    -------              --------------------
    info["pv_t"]      ↔  Pp     (PV power, kW)
    info["load_t"]    ↔  Pl     (load, kW magnitude)
    env.battery.soc   ↔  SoC    (x 100 → percent)
    info["Pb_effective"] ↔ Pb   (kW, >0 discharge)
    info["P_grid"]    ↔  Pg     (kW, >0 import)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `envs`, `agents`, etc. importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402

from envs.registry import make_env  # noqa: E402
from monitoring.plot_monitoring import plot_monitoring  # noqa: E402
from monitoring.plot_power import plot_power  # noqa: E402


def _load_model(model_path: str, algo: str):
    """Load a Stable-Baselines3 checkpoint. `algo` matches config['training']['algorithm']."""
    algo_lower = algo.lower()
    if algo_lower == "sac":
        from stable_baselines3 import SAC
        return SAC.load(model_path)
    if algo_lower == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    if algo_lower == "td3":
        from stable_baselines3 import TD3
        return TD3.load(model_path)
    raise ValueError(f"Unknown algorithm {algo!r}; extend _load_model() to support it.")


def _build_forecast_df(forecast_csv: Path | None, env) -> pd.DataFrame | None:
    """Build a forecast DataFrame aligned to the env's timestamps.

    Columns expected from the forecast CSV: at least `Time`, `pv_forecast`,
    `load_forecast`. Falls back to the env's own perfect-foresight forecasts
    if no file is provided.
    """
    n = env.max_steps
    times = pd.to_datetime(env.pv.timestamps[:n])

    if forecast_csv is None:
        # No external forecast — use the env's perfect-foresight (= actual) values.
        pv_fc   = np.array([env.pv.get_irradiance(k) for k in range(n)])
        load_fc = np.array([env.load.get_load(k)     for k in range(n)])
    else:
        df = pd.read_csv(forecast_csv, parse_dates=["Time"]) if "Time" in pd.read_csv(forecast_csv, nrows=1).columns else pd.read_csv(forecast_csv)
        # Pick best-effort columns; fall back to the actual env values if missing.
        if "pv_forecast" in df.columns:
            pv_fc = df["pv_forecast"].to_numpy()[:n]
        else:
            pv_fc = np.array([env.pv.get_irradiance(k) for k in range(n)])
        if "load_forecast" in df.columns:
            load_fc = df["load_forecast"].to_numpy()[:n]
        else:
            load_fc = np.array([env.load.get_load(k) for k in range(n)])

    return pd.DataFrame(
        {"Pp": pv_fc, "Pl": load_fc,
         "SoC": np.nan, "Pb": np.nan, "Pg": np.nan},
        index=pd.DatetimeIndex(times, name="t"),
    )


def run(
    config_path: str,
    model_path: str,
    forecast_csv: str | None = None,
    out_csv: str = "monitoring/runs/monitoring_table.csv",
    n_steps: int | None = None,
    deterministic: bool = True,
    show: bool = True,
):
    """End-to-end deployment-style rollout + plotting.

    Args:
        config_path: YAML config used to (re)build the env.
        model_path: SB3 checkpoint (.zip) saved by train_*().
        forecast_csv: optional forecast CSV to overlay in plot_monitoring.
        out_csv: where to dump the monitoring table.
        n_steps: limit the rollout to this many decision steps (e.g. 144 for
            a 24 h day at Δt = 10 min). Defaults to env.max_steps.
        deterministic: pass-through to model.predict().
        show: open the matplotlib windows.

    Returns:
        (monitoring_df, total_cost) — also written to `out_csv`.
    """
    # ---- 1. build the test env (the env factory returns (train, test, cfg)) -
    _, env, cfg = make_env(config_path)

    # ---- 2. load the trained model ----------------------------------------
    algo = cfg["training"]["algorithm"]
    print(f"[1/4] Loading {algo} model from {model_path}")
    model = _load_model(model_path, algo)

    # ---- 3. roll out step by step -----------------------------------------
    if n_steps is None:
        n_steps = int(24 * 60 / env.delta_t_min)
    n_steps = min(int(n_steps), env.max_steps)
    print(f"[2/4] Rolling out {n_steps} decision steps (Δt = {env.delta_t_min} min)")

    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    monitoring_df = env.monitoring_table.to_dataframe().iloc[:n_steps]

    # ---- 4. persist + plot ------------------------------------------------
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitoring_df.to_csv(out_path)
    print(f"[3/4] Monitoring table written to {out_path}")

    total_cost = env.monitoring_table.get_total_cost(
        buy_price=env.price_import, sell_price=env.price_export,
    )
    print(f"       Total optimization cost = {total_cost:.4f} €")

    forecast_df = _build_forecast_df(
        Path(forecast_csv) if forecast_csv else None, env,
    )

    print(f"[4/4] Opening interactive plots (close windows to exit)")
    plot_power(
        monitoring_df,
        delta_t_minutes=env.delta_t_min,
        cost=total_cost,
        base_load_kw=cfg.get("load", {}).get("base_load_kw"),
        show=False,
    )
    plot_monitoring(
        monitoring_df,
        forecast_df=forecast_df,
        delta_t_minutes=env.delta_t_min,
        show=False,
    )
    if show:
        plt.show()

    return monitoring_df, total_cost


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="YAML config matching the trained model")
    p.add_argument("--model",  required=True, help="Path to the saved SB3 .zip checkpoint")
    p.add_argument("--forecast", default=None,
                   help="Optional forecast CSV (with pv_forecast/load_forecast columns)")
    p.add_argument("--out", default="monitoring/runs/monitoring_table.csv",
                   help="Destination CSV for the monitoring table")
    p.add_argument("--steps", type=int, default=None,
                   help="Number of decision steps (default: env.max_steps)")
    args = p.parse_args()

    run(
        config_path=args.config,
        model_path=args.model,
        forecast_csv=args.forecast,
        out_csv=args.out,
        n_steps=args.steps,
        show=True,
    )


if __name__ == "__main__":
    main()
