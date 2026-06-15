"""5-panel actual-vs-forecast comparison plot for an RL rollout.

Panels (top → bottom): Pp, Pb, Pg, SoC, r_eco — the same layout as
``plot_monitoring_milp`` so the RL and MILP monitoring figures read as one tool.

Adapts MATLAB ``@MPC/plot_splitted.m`` to the RL deployment context:
where the MPC code overlays MILP set-points (planned stairs) on top of the
realised monitoring trace, we instead overlay the forecasted PV (the
observation the agent saw) on top of what actually happened — so a quick
glance tells the user how forecast error propagated through the policy. The
Pg panel mirrors the MILP figure: import/export filled stairs with the
import/export prices overlaid on a twin axis. Use ``plot_power`` for the full
power balance including the load Pl.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Same palette as plot_power.py / plot_monitoring_milp.py so the plots feel like one tool.
_C_ACTUAL   = "#2c3e50"   # actual trajectory (solid)
_C_FORECAST = "#9b59b6"   # forecast trajectory (dashed)
_C_ERROR    = "#e74c3c"   # forecast-vs-actual error fill
_C_BAT_DISCH  = "#2ecc71"
_C_BAT_CHARGE = "#3498db"
_C_GRID_IMP   = "#7f8c8d"  # grid import (top half of the Pg panel)
_C_GRID_EXP   = "#34495e"  # grid export (bottom half of the Pg panel)
_C_SOC      = "#16a085"
_C_PHANTOM  = "#9b59b6"   # phantom power (slack Duchaud-JL) — stacked above grid import
_C_PRICE    = "#e67e22"   # import/export price overlay (twin axis on the Pg panel)


def plot_monitoring(
    monitoring_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    delta_t_minutes: float = 10.0,
    init_soc: Optional[float] = None,
    show: bool = True,
):
    """Compare actuals vs forecasts across five stacked panels.

    Panels (top → bottom): Pp, Pb, Pg, SoC, r_eco — the same layout (and the
    same Pg import/export + price overlay) as ``plot_monitoring_milp`` so the RL
    and MILP monitoring figures read as one tool. The load Pl is no longer given
    its own panel here; use ``plot_power`` for the full power balance.

    Args:
        monitoring_df: actual rollout, from `MonitoringTable.to_dataframe()`.
            Required columns: Pp, SoC, Pb, Pg. Optional: Pph, price_imp,
            price_exp (used for the Pg phantom layer and price overlay).
        forecast_df: optional, same columns and index as `monitoring_df`.
            When provided, the Pp panel overlays forecast (dashed) plus a
            shaded error fill between forecast and actual. (Pb / Pg are
            decisions/outcomes, never observations, so they carry no forecast.)
        delta_t_minutes: timestep size, used in the subtitle and to size the
            stair sentinel on the Pb panel.
        init_soc: optional initial SoC *fraction* (0-1) at t0. The monitoring
            table only records post-step SoC (SoC(1)..SoC(n)) at the start of
            each interval; when given, the SoC panel is reconstructed as the
            SoC(0)->SoC(n) staircase — SoC(0) plotted at t0 and the recorded
            post-step states realigned to the end of their interval (t1..tn).
            Power panels are left untouched. When None, the legacy display
            (raw SoC(1)..SoC(n) at t0..t(n-1)) is kept.
        show: when True, call ``plt.show()``. Tests pass False.

    Returns:
        (fig, axes): the figure and a 5-tuple of subplot Axes
            (ax_pp, ax_pb, ax_pg, ax_soc, ax_eco).
    """
    df = monitoring_df.copy()
    valid = ~df["Pp"].isna()
    df = df.loc[valid]
    if df.empty:
        raise ValueError("monitoring_df is empty or fully NaN.")

    t = df.index
    Pp = df["Pp"].to_numpy(dtype=float)
    Pb = df["Pb"].to_numpy(dtype=float)
    Pg = df["Pg"].to_numpy(dtype=float)
    SoC = df["SoC"].to_numpy(dtype=float)
    # Phantom power (slack à la Duchaud-JL). Optional column; NaN → 0.
    Pph = (np.nan_to_num(df["Pph"].to_numpy(dtype=float), nan=0.0)
           if "Pph" in df.columns else np.zeros_like(Pp))
    # Import/export prices for the Pg overlay. Optional columns; all-NaN → no overlay.
    price_imp = (df["price_imp"].to_numpy(dtype=float)
                 if "price_imp" in df.columns else np.full(len(df), np.nan))
    price_exp = (df["price_exp"].to_numpy(dtype=float)
                 if "price_exp" in df.columns else np.full(len(df), np.nan))

    has_forecast = forecast_df is not None and not forecast_df.empty
    if has_forecast:
        fc = forecast_df.reindex(df.index).ffill().bfill()
        Pp_fc = fc["Pp"].to_numpy(dtype=float)
    else:
        Pp_fc = None

    fig, axes = plt.subplots(
        5, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"hspace": 0.20, "height_ratios": [1, 1, 1, 1, 0.9]},
    )
    ax_pp, ax_pb, ax_pg, ax_soc, ax_eco = axes

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

    # -------- Panel 2: Pb actual only, cross-axes --------------------------
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

    # -------- Panel 3: Pg (grid) — cross-axes + price overlay --------------
    # Mirrors plot_monitoring_milp's Pg panel (minus the MILP setpoint line):
    # import/export filled stairs, phantom stacked atop the import, and the
    # import/export prices overlaid on a twin axis. Pg is an outcome of the
    # agent's decision, not an observation, so it carries no forecast.
    Pg_ext  = np.concatenate([Pg, [Pg[-1]]])
    Pph_ext = np.concatenate([Pph, [Pph[-1]]])
    Pg_pos_ext = np.maximum(Pg_ext, 0.0)
    Pg_neg_ext = np.minimum(Pg_ext, 0.0)

    ax_pg.fill_between(t_ext, 0.0, Pg_pos_ext, step="post",
                       color=_C_GRID_IMP, alpha=0.55, label="Pg > 0 (import)")
    ax_pg.fill_between(t_ext, Pg_neg_ext, 0.0, step="post",
                       color=_C_GRID_EXP, alpha=0.55, label="Pg < 0 (export)")
    # Phantom stacked on top of the (capped) grid import: slack conjured when
    # import alone can't serve the load. Invisible when Pph=0 — legend always on.
    ax_pg.fill_between(t_ext, Pg_pos_ext, Pg_pos_ext + Pph_ext, step="post",
                       color=_C_PHANTOM, alpha=0.5, linewidth=0.0,
                       label="Pph (phantom)")
    ax_pg.plot(t_ext, Pg_ext, drawstyle="steps-post",
               color=_C_ACTUAL, linewidth=1.0, alpha=0.7)
    ax_pg.spines["bottom"].set_position("zero")
    ax_pg.spines["top"].set_visible(False)
    ax_pg.set_ylabel("Pg (kW)")
    ax_pg.grid(True, axis="y", alpha=0.3)
    pg_amp = float(np.nanmax(np.concatenate(
        [np.abs(Pg), Pg_pos_ext + Pph_ext]))) or 1.0
    ax_pg.set_ylim(-1.15 * pg_amp, 1.15 * pg_amp)

    # Import/export price overlay on a twin axis (€/kWh). Invisible when the
    # optional price columns are absent — the legend then carries power only.
    ax_pg_price = ax_pg.twinx()
    if not np.all(np.isnan(price_imp)):
        ax_pg_price.plot(t, price_imp, color=_C_PRICE, linewidth=1.0,
                         alpha=0.9, label="price import")
    if not np.all(np.isnan(price_exp)):
        ax_pg_price.plot(t, price_exp, color=_C_PRICE, linewidth=1.0,
                         linestyle="--", alpha=0.9, label="price export")
    ax_pg_price.set_ylabel("price (€/kWh)", color=_C_PRICE)
    ax_pg_price.tick_params(axis="y", colors=_C_PRICE)
    ax_pg_price.spines["top"].set_visible(False)

    lines1, labels1 = ax_pg.get_legend_handles_labels()
    lines2, labels2 = ax_pg_price.get_legend_handles_labels()
    ax_pg.legend(lines1 + lines2, labels1 + labels2,
                 loc="upper right", fontsize=9, ncol=2)

    # -------- Panel 4: SoC with markers + operational bounds ---------------
    # MATLAB equivalent: MPC/plot_splitted.m line 35-39 (SoC plot).
    # SoC(0)->SoC(n) : la table stocke le SoC d'APRÈS chaque pas (SoC(1)..SoC(n)),
    # daté au DÉBUT de l'intervalle. On préfixe l'état initial SoC(0)=init_soc à
    # t0 et on réaligne les états post-pas sur la FIN de leur intervalle (t1..tn).
    # La courbe SoC s'étend donc d'un Δt au-delà des panneaux de puissance, qui
    # eux ne sont pas décalés (puissances = grandeurs d'intervalle, datées au début).
    if init_soc is not None:
        delta_t_soc = pd.Timedelta(minutes=delta_t_minutes)
        t_soc = pd.DatetimeIndex([t[0]]).append(
            pd.DatetimeIndex(t) + delta_t_soc
        )
        SoC_plot = np.concatenate([[init_soc * 100.0], SoC])
    else:
        t_soc = t
        SoC_plot = SoC
    ax_soc.plot(t_soc, SoC_plot, "-o", color=_C_SOC, linewidth=1.6, markersize=3.5,
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
    # NB: the PBRS store-value shaping term (r_store) is intentionally NOT drawn
    # here — its magnitude crushes the economic-reward scale. The cumulative line
    # below still reflects the total reward (separate twin axis).
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
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.92, hspace=0.22)

    if show:
        plt.show()
    return fig, axes
