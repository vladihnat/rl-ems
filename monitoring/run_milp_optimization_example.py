"""Run the MILP baseline on one concrete optimization example.

MILP analogue of `monitoring/run_optimization_example.py`. The MILP solver
(``baselines.milp_solver.run_milp``) plans the entire dispatch trajectory
up-front with perfect foresight; we then replay those Pb setpoints through
the env step-by-step to get a "realized" trace, which lets the comparison
plot expose any divergence between the MILP's linearised battery+hard-bounds
model and the env's nonlinear one.

Usage:
    python -m monitoring.run_milp_optimization_example \
        --config configs/exp01_perfect_foresight.yaml \
        --forecast data/pyrano_simu.csv \
        [--measures data/pyrano_simu.csv] \
        --out monitoring/runs/milp_monitoring_table.csv \
        --plan-out monitoring/runs/milp_plan.csv

Deployment semantics mirror ``run_optimization_example.py``:
    * ``--measures`` is the ground-truth data the env consumes step-by-step.
      Falls back silently to ``--forecast`` (perfect foresight) when omitted.
    * ``--forecast`` is also the data the MILP solves over.

Variable-name mapping is identical to the RL example
(see ``run_optimization_example.py`` for the table). The only RL→MILP
difference is that there is no ``--model`` argument: the MILP is self-
contained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `envs`, `baselines`, etc. importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402

from baselines.milp_solver import run_milp  # noqa: E402
from evaluation.metrics import boundary_soc_credit  # noqa: E402
from monitoring.plot_monitoring_milp import plot_monitoring_milp  # noqa: E402
from monitoring.plot_power import plot_power  # noqa: E402
from monitoring.run_optimization_example import (  # noqa: E402
    _build_deployment_env,
    _build_split_env,
    compute_score_steps,
)


def _pb_to_action(pb_kw: float, max_charge_kw: float, max_discharge_kw: float) -> np.ndarray:
    """Invert the env's action→Pb mapping (`base_microgrid_env.py:99-102`).

    For ``pb_kw >= 0`` (discharge) the env uses ``Pb = action × max_discharge_kw``;
    for ``pb_kw < 0`` (charge) it uses ``Pb = action × max_charge_kw`` (action
    negative, max_charge_kw positive, product negative). Inverting gives the
    same |action| ∈ [0, 1].
    """
    if pb_kw >= 0.0:
        a = pb_kw / max_discharge_kw
    else:
        a = pb_kw / max_charge_kw
    return np.array([float(np.clip(a, -1.0, 1.0))], dtype=np.float32)


def _build_milp_plan_df(hist: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Wrap the MILP `history` dict into a MonitoringTable-shaped DataFrame.

    Sign + unit conversions mirror `MonitoringTable.insert` (SoC fraction →
    percent; everything else passes through). We reuse the env's monitoring
    DatetimeIndex directly so the two DataFrames are guaranteed to align
    under `reindex` (matching tz + dtype).
    """
    n = len(hist["Pb_effective"])
    return pd.DataFrame(
        {
            "Pp":  np.asarray(hist["pv_t"][:n], dtype=float),
            "Pl":  np.asarray(hist["load_t"][:n], dtype=float),
            "SoC": np.asarray(hist["soc"][:n], dtype=float) * 100.0,
            "Pb":  np.asarray(hist["Pb_effective"][:n], dtype=float),
            "Pg":  np.asarray(hist["P_grid"][:n], dtype=float),
        },
        index=index[:n],
    )


def run(
    config_path: str,
    out_csv: str = "monitoring/runs/milp_monitoring_table.csv",
    plan_csv: str = "monitoring/runs/milp_plan.csv",
    forecast_csv: str | None = None,
    measures_csv: str | None = None,
    load_csv: str | None = None,
    split: str = "full",
    n_steps: int | None = None,  # noqa: ARG001 — deprecated, kept for backward compat
    offset: int = 0,  # noqa: ARG001 — deprecated, kept for backward compat
    score_days: int | None = None,
    show: bool = True,
    cap_horizon: bool = False,
):
    """End-to-end MILP example: solve + env-replay + dump + plot.

    Args:
        config_path: YAML config matching the env to build (same one used to
            train the RL agent works fine — the MILP just reads ``battery``,
            ``grid``, ``time``, ``pv``, ``load`` sections).
        out_csv: Where to dump the env-replayed monitoring table.
        plan_csv: Where to dump the MILP setpoint table.
        forecast_csv: Optional forecast CSV. Used as the env's deployment data
            when ``measures_csv`` is None.
        measures_csv: Optional measured-state CSV consumed by the env. Defaults
            to ``forecast_csv`` (perfect foresight).
        load_csv: Optional deployment load CSV (real-load configs). Overrides
            ``cfg["data"]["load_csv"]`` so simulation data stays separate from
            training data. Typically ``data/load_simulation.csv``.
        split: ``"full"`` (default) solves over the whole CSV / deployment data;
            ``"test"`` or ``"train"`` solves over the exact registry split (same
            data ``experiments/run_experiment.py`` evaluated), ignoring the CSV
            override args.
        n_steps: Deprecated no-op. The MILP now solves the full trajectory and
            both plots cover it entirely.
        offset: Deprecated no-op (the visualisation window has been removed).
        score_days: Évaluation N-jours / scoring sur N−1. Le MILP résout TOUJOURS
            l'horizon complet (sinon il re-dumperait au pas N−1), mais le coût
            env-replay rapporté ne couvre que les ``score_days`` premiers jours
            (cf. ``compute_score_steps``). ``None`` score toute la fenêtre.
        show: Open the matplotlib windows.

    Returns:
        ``(monitoring_df, milp_plan_df, replay_cost)`` — also written to the
        two CSV paths above.
    """
    if split == "full":
        deployment_csv = measures_csv if measures_csv is not None else forecast_csv
        env, cfg = _build_deployment_env(config_path, deployment_csv, load_csv,
                                         cap_horizon=cap_horizon)
    else:
        env, cfg = _build_split_env(config_path, split, cap_horizon=cap_horizon)

    print(f"[1/4] Solving MILP (T = {env.max_steps} steps, "
          f"Δt = {env.delta_t_min} min)")
    metrics = run_milp(env, cfg)
    hist = metrics["history"]
    print(f"       solver status   = {metrics['solver_status']}")
    print(f"       objective value = {metrics['objective_value']:.4f} € "
          f"(inclut la pénalité fantôme)")
    print(f"       grid net cost   = {metrics['net_cost']:.4f} € (économie réseau réelle)")
    if metrics.get("phantom_energy_kwh", 0.0) > 1e-9:
        print(f"       ⚠ puissance fantôme = {metrics['phantom_energy_kwh']:.2f} kWh "
              f"sur {metrics['phantom_steps']} pas — système sous-dimensionné")

    print("[2/4] Replaying MILP actions through the env")
    n_plan = len(hist["Pb_effective"])
    env.reset()
    for k in range(n_plan):
        action = _pb_to_action(
            float(hist["Pb_effective"][k]),
            env.max_charge_kw,
            env.max_discharge_kw,
        )
        _, _, terminated, truncated, _ = env.step(action)
        # Overwrite the env-derived monitoring row with MILP-side columns so the
        # CSV reflects the planner's intent, not the env-replay deviation, for
        # economics/auxiliaries. The five base power columns stay from env.step.
        mt = env.monitoring_table
        if mt is not None and k < mt.n_steps:
            mt.insert(
                k,
                {
                    "pp":           float(hist["pv_t"][k]),
                    "pl":           float(hist["load_t"][k]),
                    "soc":          float(hist["soc"][k]) * 100.0,
                    "pb":           float(hist["Pb_effective"][k]),
                    "pg":           float(hist["P_grid"][k]),
                    "p_imp":        float(hist["P_imp"][k]),
                    "p_exp":        float(hist["P_exp"][k]),
                    "price_imp":    float(hist["price_imp"][k]),
                    "price_exp":    float(hist["price_exp"][k]),
                    "r_eco":        float(hist["r_eco"][k]),
                    "r_soc":        float(hist["r_soc"][k]),
                    "reward":       float(hist["reward"][k]),
                    "pb_charge":    float(hist["Pb_charge"][k]),
                    "pb_discharge": float(hist["Pb_discharge"][k]),
                    "b_int":        float(hist["b_int"][k]),
                    "pcurt":        float(hist["Pcurt"][k]),
                    # Planner's phantom (slack), consistent with the planner Pb/Pg
                    # overwritten above. Without this, the env-replay phantom from
                    # env.step survives and contradicts the displayed Pb/Pg.
                    "pph":          float(hist["P_phantom"][k]),
                },
            )
        if terminated or truncated:
            break

    monitoring_df = env.monitoring_table.to_dataframe().iloc[:n_plan]
    milp_plan_df = _build_milp_plan_df(hist, monitoring_df.index)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitoring_df.to_csv(out_path)

    plan_path = Path(plan_csv)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    milp_plan_df.to_csv(plan_path)

    print(f"[3/4] Wrote env-replay CSV   → {out_path}")
    print(f"       Wrote MILP plan  CSV → {plan_path}")

    buy = env.price_signal.import_prices
    sell = env.price_signal.export_prices
    full_replay_cost = env.monitoring_table.get_total_cost(buy_price=buy, sell_price=sell)
    score_steps = compute_score_steps(cfg, n_plan, score_days)
    # Coût headline : scoré sur N−1 jours si --score-days, sinon pleine fenêtre.
    replay_cost = env.monitoring_table.get_total_cost(
        buy_price=buy, sell_price=sell, n_steps=score_steps,
    )
    if score_steps is not None and score_steps < n_plan:
        # Crédit de SoC de bord (cf. evaluation.metrics.boundary_soc_credit / exp22) : même
        # correction que côté RL (run_optimization_example) pour que le coût scoré MILP reste
        # symétrique et matche le net_cost de run_experiment. SoC table en POURCENT ⇒ /100.
        replay_cost -= boundary_soc_credit(
            soc_end_frac=float(monitoring_df["SoC"].iloc[score_steps - 1]) / 100.0,
            soc_init_frac=float(cfg["battery"]["init_soc"]),
            capacity_kwh=float(cfg["battery"]["capacity_kwh"]),
            eta_discharge=float(cfg["battery"]["efficiency_discharge"]),
            store_value=float(env._store_value(score_steps - 1)),
        )
        win_start = monitoring_df.index[0]
        win_end = monitoring_df.index[score_steps - 1]
        print(f"       env-replay cost scoré (jours 1..{score_days}, {score_steps} pas, "
              f"{win_start.date()} → {win_end.date()}) = {replay_cost:.4f} €")
        print(f"       env-replay cost pleine fenêtre ({n_plan} pas) = "
              f"{full_replay_cost:.4f} €  "
              f"(MILP grid net cost = {metrics['net_cost']:.4f} €, "
              f"gap = {full_replay_cost - metrics['net_cost']:+.4f} €)")
    else:
        print(f"       env-replay cost = {replay_cost:.4f} €  "
              f"(MILP grid net cost = {metrics['net_cost']:.4f} €, "
              f"gap = {replay_cost - metrics['net_cost']:+.4f} €)")

    print(f"[4/4] Opening interactive plots ({n_plan} steps shown — "
          f"{monitoring_df.index[0]} → {monitoring_df.index[-1]} — "
          f"close windows to exit)")
    plot_power(
        monitoring_df,
        delta_t_minutes=env.delta_t_min,
        cost=full_replay_cost,  # le plot trace la fenêtre complète → annoté au coût complet
        show=False,
    )
    plot_monitoring_milp(
        monitoring_df,
        milp_plan_df,
        delta_t_minutes=env.delta_t_min,
        soc_safe_min=env.soc_safe_min,
        soc_safe_max=env.soc_safe_max,
        show=False,
    )
    if show:
        plt.show()

    return monitoring_df, milp_plan_df, replay_cost


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True,
                   help="YAML config (e.g. configs/exp01_perfect_foresight.yaml)")
    p.add_argument("--forecast", default=None,
                   help="Forecast CSV — also used as the env's deployment data "
                        "when --measures is omitted.")
    p.add_argument("--measures", default=None,
                   help="Measured-state CSV consumed by the env step by step. "
                        "Defaults to --forecast (perfect foresight).")
    p.add_argument("--load-csv", default=None,
                   help="Deployment load CSV (real-load configs). Overrides the "
                        "config's data.load_csv so simulation data stays separate "
                        "from training data. Typically data/load_simulation.csv.")
    p.add_argument("--split", default="full", choices=["full", "test", "train"],
                   help="'full' solves over the deployment CSV; 'test'/'train' "
                        "solve over the exact registry split (same data "
                        "run_experiment.py evaluated). With test/train the CSV "
                        "args are ignored.")
    p.add_argument("--out", default="monitoring/runs/milp_monitoring_table.csv",
                   help="CSV path for the env-replayed monitoring table")
    p.add_argument("--plan-out", default="monitoring/runs/milp_plan.csv",
                   help="CSV path for the MILP setpoint table")
    p.add_argument("--score-days", type=int, default=None,
                   help="Évaluation N-jours / scoring sur N−1 : le MILP résout "
                        "l'horizon complet mais le coût rapporté ne couvre que les "
                        "N premiers jours (ex. --score-days 1 pour 2 jours). Exclut "
                        "le dump de batterie de fin d'horizon. Défaut : toute la "
                        "fenêtre.")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() — useful in CI or tests")
    p.add_argument("--cap-horizon", action="store_true",
                   help="Garde le cap max_steps = n_steps - horizon (le MILP planifie "
                        "le même horizon que l'entraînement RL). À utiliser pour le "
                        "sanity-check overfit (MILP ≡ horizon vu par le RL).")
    args = p.parse_args()

    run(
        config_path=args.config,
        out_csv=args.out,
        plan_csv=args.plan_out,
        forecast_csv=args.forecast,
        measures_csv=args.measures,
        load_csv=args.load_csv,
        split=args.split,
        score_days=args.score_days,
        show=not args.no_show,
        cap_horizon=args.cap_horizon,
    )


if __name__ == "__main__":
    main()
