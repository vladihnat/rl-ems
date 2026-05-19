"""4-panel comparison plot — MILP setpoints vs env-replayed actuals.

Python port of MATLAB ``@MPC/plot_splitted.m`` for the perfect-foresight MILP
case. Where the MATLAB version overlays MILP setpoints (stairs) on top of the
monitored trace from a rolling MPC, here we do the same with the actions
produced by a single full-horizon MILP solve replayed through the env.

Panels (top → bottom): Pp, Pb, Pg, SoC. Pp/Pl are inputs to both the MILP and
the env, so the Pp panel should overlap exactly with perfect foresight (it
acts as a sanity check). Divergences on Pb / Pg / SoC reveal where the MILP's
linearised battery + hard SoC bounds differ from the env's nonlinear model.

Colour palette is identical to ``plot_monitoring.py`` so the two comparison
plots feel like one tool.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Shared palette — keep in lockstep with plot_monitoring.py and plot_power.py.
_C_ACTUAL    = "#2c3e50"   # env-replayed trajectory (solid line)
_C_MILP_PLAN = "#9b59b6"   # MILP setpoints (dashed step)
_C_ERROR     = "#e74c3c"   # |plan − actual| fill
_C_BAT_DISCH  = "#2ecc71"
_C_BAT_CHARGE = "#3498db"
_C_GRID_IMP   = "#7f8c8d"
_C_GRID_EXP   = "#34495e"
_C_SOC        = "#16a085"


def _extend_for_step_post(t: np.ndarray, y: np.ndarray, delta_t: pd.Timedelta):
    """Append a sentinel timestamp + repeated last value so the trailing
    interval of a ``drawstyle='steps-post'`` curve is actually drawn.

    Same trick as ``plot_power.py:_extend_for_step_post`` — duplicated locally
    so the two plot modules stay independent.
    """
    t_ext = np.concatenate([t, [t[-1] + delta_t]])
    y_ext = np.concatenate([y, [y[-1]]])
    return t_ext, y_ext


def plot_monitoring_milp(
    monitoring_df: pd.DataFrame,
    milp_plan_df: pd.DataFrame,
    delta_t_minutes: float = 5.0,
    soc_safe_min: Optional[float] = None,
    soc_safe_max: Optional[float] = None,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Plot MILP setpoints vs env-replayed actuals across four stacked panels.

    Args:
        monitoring_df: env-replayed actuals (from `MonitoringTable.to_dataframe()`).
            Required columns: Pp, Pl, SoC, Pb, Pg. DatetimeIndex.
        milp_plan_df: MILP setpoints over the same time window. Required columns:
            Pp, Pl, SoC, Pb, Pg. DatetimeIndex (need not match `monitoring_df`
            exactly — it will be reindexed onto `monitoring_df.index` with a
            forward/back fill, mirroring `plot_monitoring.py:65`).
        delta_t_minutes: timestep size, used in the subtitle and to size the
            stair sentinel on the Pb / Pg panels.
        soc_safe_min, soc_safe_max: optional operational SoC bounds (fractions
            in [0, 1]) to draw as horizontal references on the SoC panel.
            Defaults to 0.20 / 0.90 to match exp01.
        show: when True, call ``plt.show()``. Tests pass False.

    Returns:
        (fig, (ax_pp, ax_pb, ax_pg, ax_soc)).
    """
    df = monitoring_df.copy()
    valid = ~df["Pp"].isna()
    df = df.loc[valid]
    if df.empty:
        raise ValueError("monitoring_df is empty or fully NaN.")

    plan = milp_plan_df.reindex(df.index).ffill().bfill()
    if plan["Pp"].isna().any():
        raise ValueError(
            "milp_plan_df could not be aligned to monitoring_df.index "
            "(all-NaN after reindex). Check the two DataFrames share a time range."
        )

    t = df.index
    Pp     = df["Pp"].to_numpy(dtype=float)
    Pb     = df["Pb"].to_numpy(dtype=float)
    Pg     = df["Pg"].to_numpy(dtype=float)
    SoC    = df["SoC"].to_numpy(dtype=float)
    Pp_p   = plan["Pp"].to_numpy(dtype=float)
    Pb_p   = plan["Pb"].to_numpy(dtype=float)
    Pg_p   = plan["Pg"].to_numpy(dtype=float)
    SoC_p  = plan["SoC"].to_numpy(dtype=float)

    delta_t = pd.Timedelta(minutes=delta_t_minutes)

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    ax_pp, ax_pb, ax_pg, ax_soc = axes

    # -------- Panel 1: Pp (PV production) ---------------------------------
    # With perfect foresight, MILP and env see the same PV trace, so the two
    # curves should overlap exactly — a useful balance / sign sanity check.
    ax_pp.plot(t, Pp, "-", color=_C_ACTUAL, linewidth=1.8, label="Pp realized")
    ax_pp.plot(t, Pp_p, drawstyle="steps-post", linestyle="--",
               color=_C_MILP_PLAN, linewidth=1.5, label="Pp MILP setpoint")
    ax_pp.fill_between(t, Pp, Pp_p, color=_C_ERROR, alpha=0.18,
                       label="|plan − actual|")
    ax_pp.set_ylabel("Pp (kW)")
    ax_pp.grid(True, alpha=0.3)
    ax_pp.legend(loc="upper right", fontsize=9)

    # -------- Panel 2: Pb (battery) — cross-axes --------------------------
    # MATLAB equivalent: plot_splitted.m line 16-20 (stairs(consignes.Pb) vs plot(monito.Pb))
    t_ext, Pb_ext = _extend_for_step_post(t.to_numpy(), Pb, delta_t)
    _,     Pb_p_ext = _extend_for_step_post(t.to_numpy(), Pb_p, delta_t)
    Pb_pos_ext = np.maximum(Pb_ext, 0.0)
    Pb_neg_ext = np.minimum(Pb_ext, 0.0)

    ax_pb.fill_between(t_ext, 0.0, Pb_pos_ext, step="post",
                       color=_C_BAT_DISCH, alpha=0.40, label="Pb > 0 (discharge)")
    ax_pb.fill_between(t_ext, Pb_neg_ext, 0.0, step="post",
                       color=_C_BAT_CHARGE, alpha=0.40, label="Pb < 0 (charge)")
    ax_pb.plot(t_ext, Pb_ext, drawstyle="steps-post",
               color=_C_ACTUAL, linewidth=1.4, label="Pb realized")
    ax_pb.plot(t_ext, Pb_p_ext, drawstyle="steps-post", linestyle="--",
               color=_C_MILP_PLAN, linewidth=1.5, label="Pb MILP setpoint")
    ax_pb.spines["bottom"].set_position("zero")
    ax_pb.spines["top"].set_visible(False)
    ax_pb.spines["right"].set_visible(False)
    ax_pb.set_ylabel("Pb (kW)")
    ax_pb.grid(True, axis="y", alpha=0.3)
    ax_pb.legend(loc="upper right", fontsize=9, ncol=2)
    pb_amp = float(np.nanmax(np.abs(np.concatenate([Pb, Pb_p])))) or 1.0
    ax_pb.set_ylim(-1.15 * pb_amp, 1.15 * pb_amp)

    # -------- Panel 3: Pg (grid) — cross-axes -----------------------------
    _, Pg_ext   = _extend_for_step_post(t.to_numpy(), Pg, delta_t)
    _, Pg_p_ext = _extend_for_step_post(t.to_numpy(), Pg_p, delta_t)
    Pg_pos_ext = np.maximum(Pg_ext, 0.0)
    Pg_neg_ext = np.minimum(Pg_ext, 0.0)

    ax_pg.fill_between(t_ext, 0.0, Pg_pos_ext, step="post",
                       color=_C_GRID_IMP, alpha=0.40, label="Pg > 0 (import)")
    ax_pg.fill_between(t_ext, Pg_neg_ext, 0.0, step="post",
                       color=_C_GRID_EXP, alpha=0.40, label="Pg < 0 (export)")
    ax_pg.plot(t_ext, Pg_ext, drawstyle="steps-post",
               color=_C_ACTUAL, linewidth=1.4, label="Pg realized")
    ax_pg.plot(t_ext, Pg_p_ext, drawstyle="steps-post", linestyle="--",
               color=_C_MILP_PLAN, linewidth=1.5, label="Pg MILP setpoint")
    ax_pg.spines["bottom"].set_position("zero")
    ax_pg.spines["top"].set_visible(False)
    ax_pg.spines["right"].set_visible(False)
    ax_pg.set_ylabel("Pg (kW)")
    ax_pg.grid(True, axis="y", alpha=0.3)
    ax_pg.legend(loc="upper right", fontsize=9, ncol=2)
    pg_amp = float(np.nanmax(np.abs(np.concatenate([Pg, Pg_p])))) or 1.0
    ax_pg.set_ylim(-1.15 * pg_amp, 1.15 * pg_amp)

    # -------- Panel 4: SoC (continuous, not stairs) -----------------------
    # MATLAB plot_splitted.m line 35-39 uses plot() not stairs() for SoC
    # because it's a state trajectory, not a piecewise-constant command.
    ax_soc.plot(t, SoC, "-o", color=_C_SOC, linewidth=1.6, markersize=3.5,
                label="SoC realized")
    ax_soc.plot(t, SoC_p, "--", color=_C_MILP_PLAN, linewidth=1.5,
                label="SoC MILP setpoint")
    soc_lo = 20.0 if soc_safe_min is None else float(soc_safe_min) * 100.0
    soc_hi = 90.0 if soc_safe_max is None else float(soc_safe_max) * 100.0
    ax_soc.axhline(soc_lo, linestyle="--", color="grey", linewidth=1.0, alpha=0.7,
                   label=f"Operational bounds ({soc_lo:.0f}% / {soc_hi:.0f}%)")
    ax_soc.axhline(soc_hi, linestyle="--", color="grey", linewidth=1.0, alpha=0.7)
    ax_soc.set_ylim(0, 100)
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_yticks([0, int(soc_lo), 50, int(soc_hi), 100])
    ax_soc.grid(True, alpha=0.3)
    ax_soc.legend(loc="upper right", fontsize=9)

    # -------- Shared x-axis ----------------------------------------------
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_soc.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax_soc.set_xlabel("Time")

    fig.suptitle(
        f"MILP — planned setpoints vs env-replayed (Δt = {delta_t_minutes:.0f} min)",
        fontsize=12,
    )
    fig.autofmt_xdate(rotation=0, ha="center")
    # subplots_adjust (not tight_layout): the Pb / Pg cross-axes don't play
    # well with tight_layout (same caveat as plot_monitoring.py:148).
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.97, hspace=0.22)

    if show:
        plt.show()
    return fig, (ax_pp, ax_pb, ax_pg, ax_soc)
