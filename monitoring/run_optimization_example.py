"""Run the trained RL agent on one concrete optimization example.

This script simulates real-world deployment: load the saved policy, walk the
env step-by-step (the env's `monitoring_table` is auto-populated on each
`step()`), dump the resulting monitoring table to CSV, and open the two
interactive matplotlib windows.

Usage:
    python -m monitoring.run_optimization_example \
        --config configs/exp01_perfect_foresight.yaml \
        --model  results/exp01_perfect_foresight/sac_model.zip \
        --forecast data/pyrano_simu.csv \
        [--measures data/pyrano_simu.csv] \
        --out monitoring/runs/rl_exp01_monitoring_table.csv

Deployment semantics:
    * ``--measures`` is the ground-truth measured PV / load / prices that the
      env sees step-by-step. If omitted, falls back to ``--forecast`` (perfect
      foresight, current development phase).
    * ``--forecast`` is what the agent reads as predicted future values
      (overlaid on the comparison plot). If omitted, the env's own perfect-
      foresight values are used.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make `envs`, `agents`, etc. importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402

from envs.base_microgrid_env import MicrogridEnv  # noqa: E402
from envs.components.battery import BatteryModel  # noqa: E402
from envs.components.load import LoadModel  # noqa: E402
from envs.components.price_signal import PriceSignal  # noqa: E402
from envs.components.pv_source import PVSource  # noqa: E402
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


def _build_deployment_env(config_path: str, deployment_csv: str | None):
    """Build a single MicrogridEnv consuming the deployment CSV end-to-end.

    Unlike ``envs.registry.make_env`` (which produces a train/test split), the
    deployment env exposes the full CSV — the user picks the rollout length by
    choosing how many days they extracted with ``data/extract_pyrano_simu.py``.

    Args:
        config_path: YAML config matching the trained model.
        deployment_csv: Optional path to the CSV the env should consume. When
            None, falls back to ``cfg["data"]["pv_csv"]``.

    Returns:
        (env, cfg) — a fresh MicrogridEnv plus the resolved config dict.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if deployment_csv is not None:
        cfg["data"]["pv_csv"] = str(deployment_csv)

    pv = PVSource(cfg["pv"], cfg["data"])
    load = LoadModel(
        cfg["load"],
        n_steps=pv.n_steps,
        delta_t_min=cfg["time"]["delta_t_min"],
        timestamps=pv.timestamps,
    )
    battery = BatteryModel(cfg["battery"])
    price_signal = PriceSignal(
        cfg["grid"], pv.timestamps, cfg["time"]["delta_t_min"]
    )
    env = MicrogridEnv(pv, load, battery, price_signal, cfg)
    # Deployment covers the whole CSV. The training-time cap
    # `max_steps = n_steps - horizon_steps` would crop the last horizon_h hours
    # off the rollout (e.g. 6h missing from the plots). Forecast accessors
    # already edge-pad past the end, so we can safely walk to n_steps here.
    env.max_steps = env.pv.n_steps
    return env, cfg


def _build_forecast_df(forecast_csv: Path | None, env) -> pd.DataFrame:
    """Build a forecast DataFrame aligned to the env's timestamps.

    Columns expected from the forecast CSV: at least ``Time``, ``pv_forecast``,
    ``load_forecast``. Falls back to the env's own perfect-foresight forecasts
    if a column (or the whole file) is missing.
    """
    n = env.max_steps
    times = pd.to_datetime(env.pv.timestamps[:n])

    if forecast_csv is None:
        pv_fc = np.array([env.pv.get_irradiance(k) for k in range(n)])
        load_fc = np.array([env.load.get_load(k) for k in range(n)])
    else:
        header = pd.read_csv(forecast_csv, nrows=1).columns
        if "Time" in header:
            df = pd.read_csv(forecast_csv, parse_dates=["Time"])
        else:
            df = pd.read_csv(forecast_csv)
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
    measures_csv: str | None = None,
    n_steps: int | None = None,  # noqa: ARG001 — deprecated, kept for backward compat
    deterministic: bool = True,
    show: bool = True,
):
    """End-to-end deployment-style rollout + plotting.

    Args:
        config_path: YAML config used to (re)build the env.
        model_path: SB3 checkpoint (.zip) saved by train_*().
        forecast_csv: Optional forecast CSV. Used both for the plot overlay and
            as the env's data source when ``measures_csv`` is None.
        out_csv: Where to dump the monitoring table.
        measures_csv: Optional measured-state CSV consumed by the env step by
            step. Defaults to ``forecast_csv`` (perfect foresight). If neither
            is given, the env uses ``cfg["data"]["pv_csv"]`` unchanged.
        n_steps: Deprecated no-op. The rollout now spans the full deployment
            data; the visualisation window has been removed.
        deterministic: Pass-through to model.predict().
        show: Open the matplotlib windows.

    Returns:
        (monitoring_df, total_cost) — also written to ``out_csv``.
    """
    deployment_csv = measures_csv if measures_csv is not None else forecast_csv

    env, cfg = _build_deployment_env(config_path, deployment_csv)

    algo = cfg["training"]["algorithm"]
    print(f"[1/4] Loading {algo} model from {model_path}")
    model = _load_model(model_path, algo)

    rollout_steps = int(env.max_steps)
    print(f"[2/4] Rolling out {rollout_steps} decision steps (Δt = {env.delta_t_min} min)")

    obs, _ = env.reset()
    for _ in range(rollout_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    monitoring_df = env.monitoring_table.to_dataframe().iloc[:rollout_steps]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitoring_df.to_csv(out_path)
    print(f"[3/4] Monitoring table written to {out_path}")

    total_cost = env.monitoring_table.get_total_cost(
        buy_price=env.price_signal.import_prices,
        sell_price=env.price_signal.export_prices,
    )
    print(f"       Total optimization cost = {total_cost:.4f} €")

    forecast_df = _build_forecast_df(
        Path(forecast_csv) if forecast_csv else None, env,
    )

    print("[4/4] Opening interactive plots (close windows to exit)")
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
                   help="Forecast CSV (with pv_forecast/load_forecast columns) — "
                        "used both for the plot overlay and, when --measures is "
                        "omitted, as the env's deployment data.")
    p.add_argument("--measures", default=None,
                   help="Measured-state CSV consumed by the env step by step. "
                        "Defaults to --forecast (perfect foresight).")
    p.add_argument("--out", default="monitoring/runs/monitoring_table.csv",
                   help="Destination CSV for the monitoring table")
    args = p.parse_args()

    run(
        config_path=args.config,
        model_path=args.model,
        forecast_csv=args.forecast,
        measures_csv=args.measures,
        out_csv=args.out,
        show=True,
    )


if __name__ == "__main__":
    main()
