"""4-panel actual-vs-forecast comparison plot.

Adapts MATLAB ``@MPC/plot_splitted.m`` to the RL deployment context:
where the MPC code overlays MILP set-points (planned stairs) on top of the
realised monitoring trace, we instead overlay forecasted PV / load
(observations the agent saw) on top of what actually happened — so a quick
glance tells the user how forecast error propagated through the policy.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Same palette as plot_power.py so the two plots feel like one tool.
_C_ACTUAL   = "#2c3e50"   # actual trajectory (solid)
_C_FORECAST = "#9b59b6"   # forecast trajectory (dashed)
_C_ERROR    = "#e74c3c"   # forecast-vs-actual error fill
_C_BAT_DISCH  = "#2ecc71"
_C_BAT_CHARGE = "#3498db"
_C_SOC      = "#16a085"


def plot_monitoring(
    monitoring_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    delta_t_minutes: float = 10.0,
    show: bool = True,
):
    """Compare actuals vs forecasts across four stacked panels.

    Args:
        monitoring_df: actual rollout, from `MonitoringTable.to_dataframe()`.
            Required columns: Pp, Pl, SoC, Pb. Pg may be present but is not
            shown here (use plot_power for the full balance).
        forecast_df: optional, same columns and index as `monitoring_df`.
            When provided, Pp and Pl panels overlay forecast (dashed) plus a
            shaded error fill between forecast and actual.
        delta_t_minutes: timestep size, used in the subtitle and to size the
            stair sentinel on the Pb panel.
        show: when True, call ``plt.show()``. Tests pass False.

    Returns:
        (fig, axes): the figure and a 4-tuple of subplot Axes.
    """
    df = monitoring_df.copy()
    valid = ~df["Pp"].isna()
    df = df.loc[valid]
    if df.empty:
        raise ValueError("monitoring_df is empty or fully NaN.")

    t = df.index
    Pp = df["Pp"].to_numpy(dtype=float)
    Pl = df["Pl"].to_numpy(dtype=float)
    Pb = df["Pb"].to_numpy(dtype=float)
    SoC = df["SoC"].to_numpy(dtype=float)

    has_forecast = forecast_df is not None and not forecast_df.empty
    if has_forecast:
        fc = forecast_df.reindex(df.index).ffill().bfill()
        Pp_fc = fc["Pp"].to_numpy(dtype=float)
        Pl_fc = fc["Pl"].to_numpy(dtype=float)
    else:
        Pp_fc = None
        Pl_fc = None

    fig, axes = plt.subplots(
        5, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"hspace": 0.20, "height_ratios": [1, 1, 1, 1, 0.9]},
    )
    ax_pp, ax_pl, ax_pb, ax_soc, ax_eco = axes

    # -------- Panel 1: Pp actual vs forecast -------------------------------
    # MATLAB equivalent: MPC/plot_splitted.m line 9-13 (Pp stairs vs monito plot)
    ax_pp.plot(t, Pp, "-", color=_C_ACTUAL, linewidth=1.8, label="Pp actual")
    if Pp_fc is not None:
        ax_pp.plot(t, Pp_fc, "--", color=_C_FORECAST, linewidth=1.5, label="Pp forecast")
        ax_pp.fill_between(t, Pp, Pp_fc, color=_C_ERROR, alpha=0.18,
                           label="Forecast error")
    ax_pp.set_ylabel("Pp (kW)")
    ax_pp.grid(True, alpha=0.3)
    ax_pp.legend(loc="upper right", fontsize=9)

    # -------- Panel 2: Pl actual vs forecast -------------------------------
    ax_pl.plot(t, Pl, "-", color=_C_ACTUAL, linewidth=1.8, label="Pl actual")
    if Pl_fc is not None:
        ax_pl.plot(t, Pl_fc, "--", color=_C_FORECAST, linewidth=1.5, label="Pl forecast")
        ax_pl.fill_between(t, Pl, Pl_fc, color=_C_ERROR, alpha=0.18,
                           label="Forecast error")
    ax_pl.set_ylabel("Pl (kW)")
    ax_pl.grid(True, alpha=0.3)
    ax_pl.legend(loc="upper right", fontsize=9)

    # -------- Panel 3: Pb actual only, cross-axes --------------------------
    # MATLAB equivalent: MPC/plot_splitted.m line 16-20 (Pb stairs vs monito plot).
    # No forecast: battery action is the agent's decision, never an observation.
    delta_t = pd.Timedelta(minutes=delta_t_minutes)
    t_ext = t.append(pd.DatetimeIndex([t[-1] + delta_t]))
    Pb_ext = np.concatenate([Pb, [Pb[-1]]])
    Pb_pos_ext = np.maximum(Pb_ext, 0.0)
    Pb_neg_ext = np.minimum(Pb_ext, 0.0)

    ax_pb.fill_between(t_ext, 0.0, Pb_pos_ext, step="post",
                       color=_C_BAT_DISCH, alpha=0.55, label="Pb > 0 (discharge)")
    ax_pb.fill_between(t_ext, Pb_neg_ext, 0.0, step="post",
                       color=_C_BAT_CHARGE, alpha=0.55, label="Pb < 0 (charge)")
    ax_pb.plot(t_ext, Pb_ext, drawstyle="steps-post",
               color=_C_ACTUAL, linewidth=1.0, alpha=0.7)
    ax_pb.spines["bottom"].set_position("zero")
    ax_pb.spines["top"].set_visible(False)
    ax_pb.spines["right"].set_visible(False)
    ax_pb.set_ylabel("Pb (kW)")
    ax_pb.grid(True, axis="y", alpha=0.3)
    ax_pb.legend(loc="upper right", fontsize=9)
    pb_max = float(np.nanmax(np.abs(Pb))) if len(Pb) else 1.0
    ax_pb.set_ylim(-1.15 * pb_max, 1.15 * pb_max)

    # -------- Panel 4: SoC with markers + operational bounds ---------------
    # MATLAB equivalent: MPC/plot_splitted.m line 35-39 (SoC plot).
    ax_soc.plot(t, SoC, "-o", color=_C_SOC, linewidth=1.6, markersize=3.5,
                label="SoC actual")
    ax_soc.axhline(20.0, linestyle="--", color="grey", linewidth=1.0, alpha=0.7,
                   label="Operational bounds (20% / 90%)")
    ax_soc.axhline(90.0, linestyle="--", color="grey", linewidth=1.0, alpha=0.7)
    ax_soc.set_ylim(0, 100)
    ax_soc.set_ylabel("SoC (%)")
    ax_soc.set_yticks([0, 20, 50, 90, 100])
    ax_soc.grid(True, alpha=0.3)
    ax_soc.legend(loc="upper right", fontsize=9)

    # -------- Panel 5: economic step reward + cumulative -------------------
    r_eco = df["r_eco"].to_numpy(dtype=float) if "r_eco" in df.columns \
        else np.full(len(df), np.nan, dtype=float)
    r_soc = df["r_soc"].to_numpy(dtype=float) if "r_soc" in df.columns \
        else np.full(len(df), np.nan, dtype=float)
    r_eco_safe = np.where(np.isnan(r_eco), 0.0, r_eco)
    bar_w = pd.Timedelta(minutes=delta_t_minutes).to_pytimedelta()
    bar_colors_eco = np.where(r_eco_safe >= 0.0, "#2ecc71", "#e74c3c")
    ax_eco.bar(t, r_eco_safe, width=bar_w, align="edge",
               color=bar_colors_eco, edgecolor="none", alpha=0.75,
               label="r_eco (step €)")
    ax_eco.axhline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    # SoC penalty markers: orange dot at any step with a non-zero r_soc.
    viol = (~np.isnan(r_soc)) & (np.abs(r_soc) > 1e-12)
    if np.any(viol):
        ax_eco.scatter(t[viol], r_soc[viol], color="#e67e22", s=22,
                       zorder=5, label="r_soc violation")
    ax_eco.set_ylabel("r_eco (€)")
    ax_eco.grid(True, axis="y", alpha=0.3)

    ax_eco_cum = ax_eco.twinx()
    reward_col = df["reward"].to_numpy(dtype=float) if "reward" in df.columns \
        else (r_eco_safe + np.where(np.isnan(r_soc), 0.0, r_soc))
    reward_safe = np.where(np.isnan(reward_col), 0.0, reward_col)
    ax_eco_cum.plot(t, np.cumsum(reward_safe), color="#1a1a1a",
                    linewidth=1.6, label="cumulative reward")
    ax_eco_cum.set_ylabel("cumulative reward")
    ax_eco_cum.spines["top"].set_visible(False)

    lines1, labels1 = ax_eco.get_legend_handles_labels()
    lines2, labels2 = ax_eco_cum.get_legend_handles_labels()
    ax_eco.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=9, ncol=2)

    # -------- x-axis (shared) ----------------------------------------------
    ax_eco.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_eco.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax_eco.set_xlabel("Time")

    fig.suptitle(
        f"RL Agent — step-by-step decisions (Δt = {delta_t_minutes:.0f} min)",
        fontsize=12,
    )
    fig.autofmt_xdate(rotation=0, ha="center")
    # subplots_adjust (not tight_layout) — tight_layout doesn't play well with
    # the cross-axes Pb panel whose bottom spine sits at y=0.
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.97, hspace=0.22)

    if show:
        plt.show()
    return fig, axes
