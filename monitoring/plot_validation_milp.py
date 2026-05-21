"""4-panel MILP validation/diagnostic figure.

Companion to ``plot_monitoring_milp.py``. Where that plot compares planned
setpoints to env-replayed actuals, this one zooms in on the MILP's internal
auxiliary variables so any constraint violation or unexpected decision is
visible at a glance:

  1. ``Pb`` charge/discharge decomposition vs net Pb.
  2. ``b_int`` mutual-exclusion flag (1 = discharge, 0 = charge).
  3. Grid import/export split with the import price overlay.
  4. Per-step economic cost / revenue decomposition + cumulative net cost.

All four panels share the x-axis. Colour palette follows the convention used
by ``plot_power.py`` and ``plot_monitoring_milp.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_C_ACTUAL    = "#2c3e50"
_C_MILP_PLAN = "#9b59b6"
_C_BAT_DISCH  = "#2ecc71"
_C_BAT_CHARGE = "#3498db"
_C_GRID_IMP   = "#7f8c8d"
_C_GRID_EXP   = "#34495e"
_C_PRICE      = "#e67e22"
_C_COST_IMP   = "#e74c3c"
_C_COST_EXP   = "#16a085"
_C_VIOLATION  = "#c0392b"


def _extend_for_step_post(t: np.ndarray, y: np.ndarray, delta_t: pd.Timedelta):
    t_ext = np.concatenate([t, [t[-1] + delta_t]])
    y_ext = np.concatenate([y, [y[-1]]])
    return t_ext, y_ext


def _column_or_nan(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(n, np.nan, dtype=float)


def plot_validation_milp(
    monitoring_df: pd.DataFrame,
    milp_plan_df: pd.DataFrame,
    delta_t_minutes: float = 5.0,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Render the 4-panel MILP validation figure.

    Args:
        monitoring_df: env-replayed monitoring DataFrame (from
            ``MonitoringTable.to_dataframe()``), MILP-row-overwritten by
            ``run_milp_optimization_example.py``. Expected columns include the
            new extended schema: Pb, Pg, P_imp, P_exp, price_imp, price_exp,
            r_eco, Pb_charge, Pb_discharge, b_int.
        milp_plan_df: MILP setpoint DataFrame (same index as monitoring_df).
            Used only for the net-Pb cross-check overlay on panel 1.
        delta_t_minutes: timestep size; used to extend the trailing stair and
            for any per-step area calculations.
        show: when True, call ``plt.show()``. Tests pass False.

    Returns:
        ``(fig, (ax1, ax2, ax3, ax4))``.
    """
    df = monitoring_df.copy()
    valid = ~df["Pp"].isna()
    df = df.loc[valid]
    if df.empty:
        raise ValueError("monitoring_df is empty or fully NaN.")

    plan = milp_plan_df.reindex(df.index).ffill().bfill()

    n = len(df)
    t = df.index
    t_np = t.to_numpy()
    delta_t = pd.Timedelta(minutes=delta_t_minutes)
    delta_t_h = float(delta_t_minutes) / 60.0

    Pb        = df["Pb"].to_numpy(dtype=float)
    Pg        = df["Pg"].to_numpy(dtype=float)
    Pb_ch     = _column_or_nan(df, "Pb_charge",    n)
    Pb_dis    = _column_or_nan(df, "Pb_discharge", n)
    b_int     = _column_or_nan(df, "b_int",        n)
    P_imp     = _column_or_nan(df, "P_imp",        n)
    P_exp     = _column_or_nan(df, "P_exp",        n)
    price_imp = _column_or_nan(df, "price_imp",    n)
    price_exp = _column_or_nan(df, "price_exp",    n)

    # Fall back to deriving import/export from Pg if the columns are missing
    # (older CSVs, hand-built DataFrames).
    if np.all(np.isnan(P_imp)):
        P_imp = np.maximum(Pg, 0.0)
    if np.all(np.isnan(P_exp)):
        P_exp = np.maximum(-Pg, 0.0)

    Pb_plan = plan["Pb"].to_numpy(dtype=float) if "Pb" in plan.columns else Pb

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    ax1, ax2, ax3, ax4 = axes

    # ============== Panel 1 — Pb charge/discharge decomposition ==============
    t_ext, Pb_dis_ext = _extend_for_step_post(t_np, np.where(np.isnan(Pb_dis), 0.0, Pb_dis), delta_t)
    _,     Pb_ch_ext  = _extend_for_step_post(t_np, np.where(np.isnan(Pb_ch),  0.0, Pb_ch),  delta_t)
    _,     Pb_ext     = _extend_for_step_post(t_np, Pb, delta_t)
    _,     Pb_p_ext   = _extend_for_step_post(t_np, Pb_plan, delta_t)

    ax1.fill_between(t_ext, 0.0, Pb_dis_ext, step="post",
                     color=_C_BAT_DISCH, alpha=0.55, label="Pb_discharge (>0)")
    ax1.fill_between(t_ext, -Pb_ch_ext, 0.0, step="post",
                     color=_C_BAT_CHARGE, alpha=0.55, label="-Pb_charge (<0)")
    ax1.plot(t_ext, Pb_ext, drawstyle="steps-post", linestyle="--",
             color=_C_ACTUAL, linewidth=1.4, label="Pb (net, realized)")
    ax1.plot(t_ext, Pb_p_ext, drawstyle="steps-post", linestyle=":",
             color=_C_MILP_PLAN, linewidth=1.2, label="Pb (net, MILP plan)")
    ax1.spines["bottom"].set_position("zero")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylabel("Pb (kW)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.set_title("Pb decomposition — charge vs discharge", fontsize=10)
    pb_amp = float(np.nanmax(np.abs(np.concatenate([Pb, Pb_dis, Pb_ch])))) or 1.0
    ax1.set_ylim(-1.15 * pb_amp, 1.15 * pb_amp)

    # ============== Panel 2 — b_int mutual exclusion =========================
    # Visualise b_int as bars (1 = discharge slot, 0 = charge slot). Anything
    # else (HiGHS shouldn't produce it) is flagged as a red scatter so the
    # violation jumps out.
    b_safe = np.where(np.isnan(b_int), 0.0, b_int)
    bar_colors = np.where(b_safe >= 0.5, _C_BAT_DISCH, _C_BAT_CHARGE)
    width = pd.Timedelta(minutes=delta_t_minutes).to_pytimedelta()
    # `align='edge'` so each bar covers [t[k], t[k]+delta_t), matching steps-post.
    ax2.bar(t, b_safe, width=width, align="edge",
            color=bar_colors, edgecolor="none", alpha=0.8,
            label="b_int (green=discharge, blue=charge)")

    eps = 1e-3
    not_binary = (~np.isnan(b_int)) & (np.abs(b_int) > eps) & (np.abs(b_int - 1.0) > eps)
    if np.any(not_binary):
        ax2.scatter(t[not_binary], b_int[not_binary],
                    color=_C_VIOLATION, s=24, zorder=5,
                    label="non-binary (violation)")

    ax2.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    ax2.axhline(1.0, color="grey", linewidth=0.8, alpha=0.5)
    ax2.set_ylim(-0.15, 1.15)
    ax2.set_yticks([0.0, 1.0])
    ax2.set_ylabel("b_int")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("b_int — mutual exclusion flag (1 = discharge)", fontsize=10)

    # ============== Panel 3 — import/export split + import price =============
    P_imp_safe = np.where(np.isnan(P_imp), 0.0, P_imp)
    P_exp_safe = np.where(np.isnan(P_exp), 0.0, P_exp)
    _, P_imp_ext = _extend_for_step_post(t_np, P_imp_safe, delta_t)
    _, P_exp_ext = _extend_for_step_post(t_np, P_exp_safe, delta_t)
    _, Pg_ext    = _extend_for_step_post(t_np, Pg, delta_t)

    ax3.fill_between(t_ext, 0.0, P_imp_ext, step="post",
                     color=_C_GRID_IMP, alpha=0.55, label="P_imp (>0)")
    ax3.fill_between(t_ext, -P_exp_ext, 0.0, step="post",
                     color=_C_GRID_EXP, alpha=0.55, label="-P_exp (<0)")
    ax3.plot(t_ext, Pg_ext, drawstyle="steps-post", linestyle="--",
             color=_C_ACTUAL, linewidth=1.3, label="Pg (cross-check)")
    ax3.spines["bottom"].set_position("zero")
    ax3.spines["top"].set_visible(False)
    ax3.set_ylabel("Power (kW)")
    ax3.grid(True, axis="y", alpha=0.3)

    pg_amp = float(np.nanmax(np.abs(np.concatenate([P_imp_safe, P_exp_safe, Pg])))) or 1.0
    ax3.set_ylim(-1.15 * pg_amp, 1.15 * pg_amp)

    ax3_price = ax3.twinx()
    if not np.all(np.isnan(price_imp)):
        ax3_price.plot(t, price_imp, color=_C_PRICE, linewidth=1.0,
                       alpha=0.9, label="price_imp")
    ax3_price.set_ylabel("import price (€/kWh)", color=_C_PRICE)
    ax3_price.tick_params(axis="y", colors=_C_PRICE)
    ax3_price.spines["top"].set_visible(False)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_price.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, ncol=2)
    ax3.set_title("Grid: import / export split + import price", fontsize=10)

    # ============== Panel 4 — Step economic cost decomposition ===============
    # cost contributions (€) per step. Import cost is positive (we pay),
    # export revenue is negative (we earn). Sum is the net step cost.
    if np.all(np.isnan(price_imp)):
        cost_imp = np.zeros(n)
    else:
        cost_imp = np.where(np.isnan(price_imp), 0.0, price_imp) * P_imp_safe * delta_t_h
    if np.all(np.isnan(price_exp)):
        cost_exp = np.zeros(n)
    else:
        cost_exp = -np.where(np.isnan(price_exp), 0.0, price_exp) * P_exp_safe * delta_t_h

    width = pd.Timedelta(minutes=delta_t_minutes).to_pytimedelta()
    ax4.bar(t, cost_imp, width=width, align="edge",
            color=_C_COST_IMP, edgecolor="none", alpha=0.75,
            label="import cost (+)")
    ax4.bar(t, cost_exp, width=width, align="edge",
            color=_C_COST_EXP, edgecolor="none", alpha=0.75,
            label="export revenue (−)")
    ax4.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    ax4.set_ylabel("Step cost (€)")
    ax4.grid(True, axis="y", alpha=0.3)

    ax4_cum = ax4.twinx()
    net_cost = cost_imp + cost_exp
    cum_net = np.cumsum(net_cost)
    ax4_cum.plot(t, cum_net, color="#1a1a1a", linewidth=1.6,
                 label="cumulative net cost")
    ax4_cum.set_ylabel("Cumulative cost (€)")
    ax4_cum.spines["top"].set_visible(False)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_cum.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, ncol=2)
    ax4.set_title("Step cost / revenue and cumulative net cost", fontsize=10)

    # -------- shared x-axis ----------------------------------------------
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax4.set_xlabel("Time")

    fig.suptitle(
        f"MILP validation — auxiliaries, grid, economics (Δt = {delta_t_minutes:.0f} min)",
        fontsize=12,
    )
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.08, right=0.92, hspace=0.32)

    if show:
        plt.show()
    return fig, (ax1, ax2, ax3, ax4)
