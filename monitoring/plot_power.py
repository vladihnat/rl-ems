"""Filled-stairs power plot — Python port of `@EMS/plot.m` (top subplot).

The MATLAB version stacks ``area(stairs, ...)`` calls with custom tricks because
MATLAB has no native filled-stairs primitive. matplotlib does: a single call to
``ax.fill_between(t, 0, y, step='post')`` replaces all of it.

Visual conventions:
- Cross axes: the x-spine sits at y=0; sources (Pp, Pb>0, Pg>0) are stacked
  cumulatively upward, sinks (Pb<0, Pg<0) are stacked cumulatively downward.
- The load Pl is drawn as a step-line on the **upper** side: the top of the
  source stack should reach this line at every instant (Pp + Pb>0 + Pg>0 + Pph ≈ Pl).
- Phantom power Pph (slack à la Duchaud-JL) is the top source layer, drawn in
  violet: energy "conjured from nothing" to close the balance when import +
  battery can't meet the load. It fills exactly the gap that a P_grid clipping
  would otherwise leave between the source stack and the load line
  (see base_microgrid_env.py:153, milp_solver.py:44).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colour palette tuned to roughly match the MATLAB `lines` colormap that the
# supervisor uses in EMS/plot.m (PV ≈ orange/yellow, ESS ≈ green/blue,
# grid ≈ grey, load ≈ red).
_C_PV         = "#f1c40f"  # PV — warm yellow
_C_BAT_DISCH  = "#2ecc71"  # battery discharge (top half)
_C_BAT_CHARGE = "#3498db"  # battery charge    (bottom half)
_C_GRID_IMP   = "#7f8c8d"  # grid import       (top half)
_C_GRID_EXP   = "#34495e"  # grid export       (bottom half)
_C_PHANTOM    = "#9b59b6"  # phantom power     (top half, last layer — slack Duchaud-JL)


def _extend_for_step_post(t: np.ndarray, y: np.ndarray, delta_t: pd.Timedelta):
    """Append a sentinel timestamp and a repeated last value so that ``step='post'``
    actually draws the last interval (matplotlib drops it otherwise).
    """
    t_ext = np.concatenate([t, [t[-1] + delta_t]])
    y_ext = np.concatenate([y, [y[-1]]])
    return t_ext, y_ext


def plot_power(
    monitoring_df: pd.DataFrame,
    delta_t_minutes: float = 10.0,
    cost: Optional[float] = None,
    show: bool = True,
):
    """Display the main power-balance plot from a completed rollout.

    Args:
        monitoring_df: DataFrame from `MonitoringTable.to_dataframe()`, indexed by
            datetime with columns Pp, Pl, SoC, Pb, Pg.
        delta_t_minutes: timestep size, used to extend the last stair and label
            the x-axis correctly.
        cost: optional total optimization cost (€). Shown in the title.
        show: when True, call ``plt.show()`` to open the interactive window.
            Tests pass show=False to keep the call non-blocking.

    Returns:
        (fig, ax) so callers can decorate further if they want.
    """
    df = monitoring_df
    if df.empty:
        raise ValueError("monitoring_df is empty — nothing to plot.")

    # Drop unfilled (NaN) tail rows so the plot doesn't show a flat zero region.
    valid = ~df["Pg"].isna()
    df = df.loc[valid]
    if df.empty:
        raise ValueError("monitoring_df contains only NaN rows — was insert() called?")

    t = df.index.to_numpy()
    Pp = df["Pp"].to_numpy(dtype=float)
    Pl = df["Pl"].to_numpy(dtype=float)
    Pb = df["Pb"].to_numpy(dtype=float)
    Pg = df["Pg"].to_numpy(dtype=float)
    # Phantom power (slack à la Duchaud-JL). Optional column: tolerate old CSVs
    # without it and NaN-filled rows (treated as zero phantom).
    Pph = (np.nan_to_num(df["Pph"].to_numpy(dtype=float), nan=0.0)
           if "Pph" in df.columns else np.zeros_like(Pp))

    # Split battery and grid by sign — each side becomes its own filled stairs.
    Pb_pos = np.maximum(Pb, 0.0)   # discharge, top half
    Pb_neg = np.minimum(Pb, 0.0)   # charge,    bottom half
    Pg_pos = np.maximum(Pg, 0.0)   # import,    top half
    Pg_neg = np.minimum(Pg, 0.0)   # export,    bottom half

    delta_t = pd.Timedelta(minutes=delta_t_minutes)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # ---- cumulative source stack on the upper half --------------------------
    # `step='post'` holds the value of point i from t[i] until t[i+1] — a true
    # zero-order-hold matching MATLAB's `retime(..., 'previous')`.
    # Phantom is the LAST top layer: it tops up the stack to the load line
    # exactly when import+battery fall short (Pph>0). Always added so the
    # legend entry stays consistent across RL and MILP figures (invisible when
    # Pph=0).
    top_layers = [
        ("PV (Pp)",              Pp,     _C_PV),
        ("ESS discharge (Pb>0)", Pb_pos, _C_BAT_DISCH),
        ("Grid import (Pg>0)",   Pg_pos, _C_GRID_IMP),
        ("Phantom (Pph)",        Pph,    _C_PHANTOM),
    ]
    top_cum = np.zeros_like(Pp)
    for label, layer, color in top_layers:
        lower = top_cum.copy()
        top_cum = top_cum + layer
        t_ext, lo_ext = _extend_for_step_post(t, lower,   delta_t)
        _,     hi_ext = _extend_for_step_post(t, top_cum, delta_t)
        ax.fill_between(
            t_ext, lo_ext, hi_ext,
            step="post",
            facecolor=color, edgecolor=color, alpha=0.9, linewidth=0.6,
            label=label,
        )

    # ---- cumulative sink stack on the lower half ----------------------------
    bot_layers = [
        ("ESS charge (Pb<0)",  Pb_neg, _C_BAT_CHARGE),
        ("Grid export (Pg<0)", Pg_neg, _C_GRID_EXP),
    ]
    bot_cum = np.zeros_like(Pp)
    for label, layer, color in bot_layers:
        upper = bot_cum.copy()
        bot_cum = bot_cum + layer  # accumulates downward (≤ 0)
        t_ext, up_ext = _extend_for_step_post(t, upper,   delta_t)
        _,     lo_ext = _extend_for_step_post(t, bot_cum, delta_t)
        ax.fill_between(
            t_ext, lo_ext, up_ext,
            step="post",
            facecolor=color, edgecolor=color, alpha=0.9, linewidth=0.6,
            label=label,
        )

    # ---- load: reference line on the UPPER half -----------------------------
    # The source stack above must reach this line at every instant.
    t_ext, Pl_ext = _extend_for_step_post(t, Pl, delta_t)
    ax.plot(t_ext, Pl_ext, color="#1a1a1a", linewidth=2.5,
            drawstyle="steps-post", label="Load (Pl)")

    # ---- cross-axes layout (the supervisor's signature visual convention)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)

    # Push the x-tick labels below the data area, otherwise they collide with
    # negative-half traces near y=0.
    ax.tick_params(axis="x", direction="out", pad=18)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))

    ax.set_ylabel("Power (kW)")
    ax.grid(True, axis="y", alpha=0.3)

    # Symmetric y-limits so the y=0 cross axis is visually centred.
    # The new stack geometry means top reaches `top_cum.max()` (not max of a
    # single layer) and bottom reaches `bot_cum.min()`.
    top_max  = float(np.nanmax(top_cum)) if top_cum.size else 0.0
    bot_min  = float(np.nanmin(bot_cum)) if bot_cum.size else 0.0
    load_max = float(np.nanmax(Pl))      if Pl.size      else 0.0
    ymax = max(top_max, load_max, abs(bot_min), 1e-6)
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)

    title = "Optimization Result"
    if cost is not None:
        title = f"Optimization Result — Cost: {cost:.2f} €"
    ax.set_title(title)

    ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax
