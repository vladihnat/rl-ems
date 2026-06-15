"""Run the trained RL agent on one concrete optimization example.

This script simulates real-world deployment: load the saved policy, walk the
env step-by-step (the env's `monitoring_table` is auto-populated on each
`step()`), dump the resulting monitoring table to CSV, and open the two
interactive matplotlib windows.

Usage:
    python -m monitoring.run_optimization_example \
        --config configs/exp08_realLoad.yaml \
        --model  results/exp08_realLoad_300Bs_g99/sac_model.zip \
        --forecast data/irradiance_simulation.csv \
        [--measures data/irradiance_simulation.csv] \
        [--load-csv data/load_simulation.csv] \
        --out monitoring/runs/exp08_monitoring_table.csv

Generate simulation data first:
    python data/extract_pyrano_simu.py --nbD 7 --month 2 --startDate 1 --usage simu

Deployment semantics:
    * ``--measures`` is the ground-truth measured PV / load / prices that the
      env sees step-by-step. If omitted, falls back to ``--forecast`` (perfect
      foresight, current development phase).
    * ``--forecast`` is what the agent reads as predicted future values
      (overlaid on the comparison plot). If omitted, the env's own perfect-
      foresight values are used.
    * ``--load-csv`` overrides the load CSV for deployment (real-load configs
      only). When omitted the config's ``data.load_csv`` is used unchanged.

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
    if algo_lower == "ddpg":
        from stable_baselines3 import DDPG
        return DDPG.load(model_path)
    raise ValueError(f"Unknown algorithm {algo!r}; extend _load_model() to support it.")


def _build_deployment_env(
    config_path: str,
    deployment_csv: str | None,
    deployment_load_csv: str | None = None,
):
    """Build a single MicrogridEnv consuming the deployment CSV end-to-end.

    Unlike ``envs.registry.make_env`` (which produces a train/test split), the
    deployment env exposes the full CSV — the user picks the rollout length by
    choosing how many days they extracted with ``data/extract_pyrano_simu.py``.

    Args:
        config_path: YAML config matching the trained model.
        deployment_csv: Optional path to the PV CSV the env should consume.
            When None, falls back to ``cfg["data"]["pv_csv"]``.
        deployment_load_csv: Optional path to the load CSV for deployment.
            When None, falls back to ``cfg["data"]["load_csv"]`` (real-load
            configs) or the synthetic generator (fixed-load configs).

    Returns:
        (env, cfg) — a fresh MicrogridEnv plus the resolved config dict.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if deployment_csv is not None:
        cfg["data"]["pv_csv"] = str(deployment_csv)
    if deployment_load_csv is not None:
        cfg["data"]["load_csv"] = str(deployment_load_csv)

    pv = PVSource(cfg["pv"], cfg["data"])
    load = LoadModel(
        cfg["load"],
        cfg["data"],
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


def _build_split_env(config_path: str, split: str):
    """Build the env for a registry train/test split, matching ``make_env`` exactly.

    Unlike ``_build_deployment_env`` (which exposes the full CSV), this reproduces
    the temporal split used by ``experiments/run_experiment.py`` so the monitoring
    rollout replays the *exact* train or test set that produced ``comparison.json``.

    Args:
        config_path: YAML config.
        split: ``"train"`` or ``"test"``.

    Returns:
        (env, cfg) — the chosen split env, with ``max_steps`` extended to cover all
        its steps (the training-time horizon cap is dropped for monitoring, like
        ``_build_deployment_env``).
    """
    from envs.registry import make_env

    train_env, test_env, cfg = make_env(config_path)
    env = test_env if split == "test" else train_env
    env.max_steps = env.pv.n_steps
    return env, cfg


def compute_score_steps(cfg: dict, total_steps: int, score_days: int | None) -> int | None:
    """Nombre de pas de tête à scorer pour l'évaluation N-jours / scoring sur N−1.

    On simule/optimise la fenêtre complète (``total_steps``) mais on n'évalue le
    coût que sur les ``score_days`` premiers jours. Le(s) dernier(s) jour(s)
    servent de tampon « lendemain » : ni le RL ni le MILP n'y adoptent un
    comportement de fin d'horizon (dump de batterie), donc la zone scorée reste
    en régime permanent (cf. plan exp19).

    Returns:
        Le nombre de pas correspondant aux ``score_days`` premiers jours, borné à
        ``total_steps``. ``None`` (option ``--score-days`` absente) → on score
        toute la fenêtre (comportement historique).
    """
    if score_days is None:
        return None
    steps_per_day = round(24 * 60 / cfg["time"]["delta_t_min"])
    return min(total_steps, int(score_days) * steps_per_day)


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


def _to_soc0_csv(monitoring_df: pd.DataFrame, init_soc: float,
                 delta_t_minutes: float) -> pd.DataFrame:
    """Réécrit la table en convention SoC(0)->SoC(n) pour le CSV.

    La table de monitoring ne stocke que le SoC d'APRÈS chaque pas
    (SoC(1)..SoC(n)), daté au DÉBUT de l'intervalle. Pour le CSV — comme pour le
    plot — on veut la trajectoire complète SoC(0)..SoC(n) :

      * colonne ``SoC`` réalignée sur le DÉBUT d'intervalle : ligne k porte
        SoC(k) (SoC(0)=init_soc, puis SoC(1)..SoC(n-1)). C'est exactement la
        convention du CSV MILP, qui démarre déjà à SoC(0)=50%.
      * une ligne terminale (t_n = dernier timestamp + Δt) porte l'état final
        SoC(n) ; ses autres colonnes restent vides (aucune décision après le
        dernier pas).

    Les colonnes de puissance/prix/reward ne sont PAS décalées : elles restent
    datées au début de leur intervalle.
    """
    df = monitoring_df.copy()
    soc_post = df["SoC"].to_numpy(dtype=float)            # SoC(1)..SoC(n)
    df["SoC"] = np.concatenate([[init_soc * 100.0], soc_post[:-1]])  # SoC(0)..SoC(n-1)

    delta_t = pd.Timedelta(minutes=delta_t_minutes)
    term = pd.DataFrame(
        {"SoC": [soc_post[-1]]},                          # SoC(n)
        index=pd.DatetimeIndex([df.index[-1] + delta_t], name=df.index.name),
    )
    return pd.concat([df, term])


def run(
    config_path: str,
    model_path: str,
    forecast_csv: str | None = None,
    out_csv: str = "monitoring/runs/monitoring_table.csv",
    measures_csv: str | None = None,
    load_csv: str | None = None,
    split: str = "full",
    n_steps: int | None = None,  # noqa: ARG001 — deprecated, kept for backward compat
    score_days: int | None = None,
    deterministic: bool = True,
    show: bool = True,
    vec_normalize: str | None = None,
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
        load_csv: Optional deployment load CSV (real-load configs). Overrides
            ``cfg["data"]["load_csv"]`` so simulation and training data stay
            separate. Ignored for fixed/synthetic load configs.
        split: ``"full"`` (default) replays the whole CSV / deployment data;
            ``"test"`` or ``"train"`` replays the exact registry split (same data
            ``experiments/run_experiment.py`` evaluated), ignoring the CSV
            override args.
        n_steps: Deprecated no-op. The rollout now spans the full deployment
            data; the visualisation window has been removed.
        score_days: Évaluation N-jours / scoring sur N−1. On rejoue toujours la
            fenêtre complète, mais le coût rapporté ne couvre que les
            ``score_days`` premiers jours (cf. ``compute_score_steps``). ``None``
            score toute la fenêtre (historique).
        deterministic: Pass-through to model.predict().
        show: Open the matplotlib windows.
        vec_normalize: Observation-normalisation stats for norm_obs models.
            None (default) auto-detects ``vec_normalize.pkl`` next to the model;
            ``"none"`` forces raw observations; a path overrides the location.
            Required for a faithful replay of norm_obs=true models — without it
            the policy receives un-normalised obs and dispatches incorrectly.

    Returns:
        (monitoring_df, total_cost) — also written to ``out_csv``.
    """
    if split == "full":
        deployment_csv = measures_csv if measures_csv is not None else forecast_csv
        env, cfg = _build_deployment_env(config_path, deployment_csv, load_csv)
    else:
        env, cfg = _build_split_env(config_path, split)

    algo = cfg["training"]["algorithm"]
    print(f"[1/4] Loading {algo} model from {model_path}")
    model = _load_model(model_path, algo)

    # Observation normalisation : indispensable pour rejouer fidèlement un modèle
    # norm_obs=true (sinon obs brutes -> dispatch erroné). On ne normalise QUE l'obs
    # (eval mode), l'env Gymnasium brut reste utilisé pour le step — idem evaluate_sac.
    vec_norm = None
    if vec_normalize != "none":
        if vec_normalize is not None:
            vn_path = Path(vec_normalize)
        else:
            vn_path = Path(model_path).parent / "vec_normalize.pkl"
        if vn_path.exists():
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            vec_norm = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: env]))
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"       VecNormalize stats loaded from {vn_path}")
        elif vec_normalize is not None:
            raise FileNotFoundError(f"vec_normalize file not found: {vn_path}")

    rollout_steps = int(env.max_steps)
    print(f"[2/4] Rolling out {rollout_steps} decision steps (Δt = {env.delta_t_min} min)")

    obs, _ = env.reset()
    # SoC(0) : état initial réel APRÈS reset (= init_soc, ou tirage aléatoire si
    # random_soc=true). La table de monitoring ne stocke que les SoC post-pas
    # (SoC(1)..SoC(n)) ; on le passe au plot pour reconstituer SoC(0)->SoC(n).
    soc0 = float(env.battery.soc)
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    for _ in range(rollout_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        if terminated or truncated:
            break

    monitoring_df = env.monitoring_table.to_dataframe().iloc[:rollout_steps]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # CSV en convention SoC(0)->SoC(n) (cf. _to_soc0_csv), alignée sur le CSV MILP.
    _to_soc0_csv(monitoring_df, soc0, env.delta_t_min).to_csv(out_path)
    print(f"[3/4] Monitoring table written to {out_path}")

    buy = env.price_signal.import_prices
    sell = env.price_signal.export_prices
    full_cost = env.monitoring_table.get_total_cost(buy_price=buy, sell_price=sell)
    score_steps = compute_score_steps(cfg, rollout_steps, score_days)
    # Coût headline : scoré sur N−1 jours si --score-days, sinon pleine fenêtre.
    total_cost = env.monitoring_table.get_total_cost(
        buy_price=buy, sell_price=sell, n_steps=score_steps,
    )
    if score_steps is not None and score_steps < rollout_steps:
        win_start = monitoring_df.index[0]
        win_end = monitoring_df.index[score_steps - 1]
        print(f"       Coût scoré (jours 1..{score_days}, {score_steps} pas, "
              f"{win_start.date()} → {win_end.date()}) = {total_cost:.4f} €")
        print(f"       Coût pleine fenêtre ({rollout_steps} pas) = {full_cost:.4f} €")
    else:
        print(f"       Total optimization cost = {total_cost:.4f} €")

    forecast_df = _build_forecast_df(
        Path(forecast_csv) if forecast_csv else None, env,
    )

    print("[4/4] Opening interactive plots (close windows to exit)")
    plot_power(
        monitoring_df,
        delta_t_minutes=env.delta_t_min,
        cost=full_cost,  # le plot trace la fenêtre complète → on l'annote au coût complet
        show=False,
    )
    plot_monitoring(
        monitoring_df,
        forecast_df=forecast_df,
        delta_t_minutes=env.delta_t_min,
        init_soc=soc0,
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
    p.add_argument("--load-csv", default=None,
                   help="Deployment load CSV (real-load configs). Overrides the "
                        "config's data.load_csv so simulation data stays separate "
                        "from training data. Typically data/load_simulation.csv.")
    p.add_argument("--split", default="full", choices=["full", "test", "train"],
                   help="'full' replays the deployment CSV; 'test'/'train' replay "
                        "the exact registry split (same data run_experiment.py "
                        "evaluated). With test/train the CSV args are ignored.")
    p.add_argument("--out", default="monitoring/runs/monitoring_table.csv",
                   help="Destination CSV for the monitoring table")
    p.add_argument("--score-days", type=int, default=None,
                   help="Évaluation N-jours / scoring sur N−1 : rejoue la fenêtre "
                        "complète mais ne calcule le coût que sur les N premiers "
                        "jours (ex. --score-days 1 pour une fenêtre 2 jours). "
                        "Exclut le dump de batterie de fin d'horizon. Défaut : "
                        "score toute la fenêtre.")
    p.add_argument("--vec-normalize", default=None,
                   help="Obs-normalisation stats for norm_obs models. Default: "
                        "auto-detect vec_normalize.pkl next to the model. "
                        "'none' forces raw obs; or pass an explicit .pkl path.")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() — useful for batch runs over several "
                        "simulation cases (CSV is still written).")
    args = p.parse_args()

    run(
        config_path=args.config,
        model_path=args.model,
        forecast_csv=args.forecast,
        measures_csv=args.measures,
        load_csv=args.load_csv,
        split=args.split,
        out_csv=args.out,
        score_days=args.score_days,
        show=not args.no_show,
        vec_normalize=args.vec_normalize,
    )


if __name__ == "__main__":
    main()
