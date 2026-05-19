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
        --out monitoring/runs/milp_monitoring_table.csv \
        --plan-out monitoring/runs/milp_plan.csv

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
from envs.registry import make_env  # noqa: E402
from monitoring.plot_monitoring_milp import plot_monitoring_milp  # noqa: E402
from monitoring.plot_power import plot_power  # noqa: E402


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
    n_steps: int | None = None,
    offset: int = 0,
    show: bool = True,
):
    """End-to-end MILP example: solve + env-replay + dump + plot.

    Args:
        config_path: YAML config matching the env to build (same one used to
            train the RL agent works fine — the MILP just reads `battery`,
            `grid`, `time`, `pv`, `load` sections).
        out_csv: where to dump the env-replayed monitoring table.
        plan_csv: where to dump the MILP setpoint table.
        n_steps: number of steps in the visualisation window. Defaults to 24h
            (= ``24 × 60 / delta_t_min``) so daytime PV is visible — matches
            the default in ``run_optimization_example.py``. Pass a smaller
            value (e.g. ``env.horizon_steps`` for a 6h horizon view) to zoom.
        offset: index of the first step to show (default 0). Shift forward to
            frame any sub-window of the full MILP trajectory.
        show: open the matplotlib windows.

    Returns:
        ``(monitoring_df, milp_plan_df, replay_cost)`` — also written to the
        two CSV paths above.
    """
    # ---- 1. build the test env -------------------------------------------
    _, env, cfg = make_env(config_path)

    # ---- 2. solve the MILP over the full env horizon ---------------------
    print(f"[1/4] Solving MILP (T = {env.max_steps} steps, "
          f"Δt = {env.delta_t_min} min)")
    metrics = run_milp(env, cfg)
    hist = metrics["history"]
    print(f"       solver status   = {metrics['solver_status']}")
    print(f"       objective value = {metrics['objective_value']:.4f} €")

    # ---- 3. replay the MILP actions through the env ----------------------
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
        if terminated or truncated:
            break

    monitoring_df = env.monitoring_table.to_dataframe().iloc[:n_plan]
    milp_plan_df = _build_milp_plan_df(hist, monitoring_df.index)

    # ---- 4. choose visualisation window ----------------------------------
    # Default to 24h (matches run_optimization_example.py) so daytime PV is
    # visible. The first n_steps of the test dataset start at midnight, so
    # the default 6h-from-start window we used previously fell entirely in
    # pre-dawn (Pp = 0 everywhere). Users can still pass --steps 72 + an
    # --offset to frame a tight 6h view of any part of the trajectory.
    if n_steps is None:
        n_steps = int(24 * 60 / env.delta_t_min)
    offset = max(0, min(int(offset), n_plan - 1))
    n_steps = max(1, min(int(n_steps), n_plan - offset))
    monitoring_view = monitoring_df.iloc[offset : offset + n_steps]
    milp_view = milp_plan_df.iloc[offset : offset + n_steps]

    # ---- 5. persist both tables ------------------------------------------
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitoring_df.to_csv(out_path)

    plan_path = Path(plan_csv)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    milp_plan_df.to_csv(plan_path)

    print(f"[3/4] Wrote env-replay CSV   → {out_path}")
    print(f"       Wrote MILP plan  CSV → {plan_path}")

    replay_cost = env.monitoring_table.get_total_cost(
        buy_price=env.price_import, sell_price=env.price_export,
    )
    print(f"       env-replay cost = {replay_cost:.4f} €  "
          f"(MILP optimum = {metrics['objective_value']:.4f} €, "
          f"gap = {replay_cost - metrics['objective_value']:+.4f} €)")

    # ---- 6. plot ---------------------------------------------------------
    print(f"[4/4] Opening interactive plots ({n_steps} steps shown, "
          f"offset={offset} — {monitoring_view.index[0]} → "
          f"{monitoring_view.index[-1]} — close windows to exit)")
    plot_power(
        monitoring_view,
        delta_t_minutes=env.delta_t_min,
        cost=replay_cost,
        base_load_kw=cfg.get("load", {}).get("base_load_kw"),
        show=False,
    )
    plot_monitoring_milp(
        monitoring_view,
        milp_view,
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
    p.add_argument("--out", default="monitoring/runs/milp_monitoring_table.csv",
                   help="CSV path for the env-replayed monitoring table")
    p.add_argument("--plan-out", default="monitoring/runs/milp_plan.csv",
                   help="CSV path for the MILP setpoint table")
    p.add_argument("--steps", type=int, default=None,
                   help="Visualisation window (steps). "
                        "Default: 24h (= 24 × 60 / delta_t_min). "
                        "Pass env.horizon_steps for a 6h horizon view.")
    p.add_argument("--offset", type=int, default=0,
                   help="Index of the first step to show (default 0). Use "
                        "to skip nighttime or zoom into a specific period.")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() — useful in CI or tests")
    args = p.parse_args()

    run(
        config_path=args.config,
        out_csv=args.out,
        plan_csv=args.plan_out,
        n_steps=args.steps,
        offset=args.offset,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
