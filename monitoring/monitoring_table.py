"""Pre-allocated monitoring table for post-training simulation rollouts.

Python port of the MATLAB `MicroGrid.Monitoring` matrix and `monitoTable`
dependent property. One row per RL decision step; the row is written
(indexed) at the moment the agent acts on the env.

Column layout (extended schema, 25 columns):
    SHARED (always populated by RL or MILP rollouts):
        t          unix seconds (float64)
        Pp         PV production [kW], >= 0
        Pl         load demand [kW], >= 0
        SoC        state of charge in PERCENT (0-100)
        Pb         battery power [kW]; >0 discharge, <0 charge
        Pg         grid power [kW]; >0 import, <0 export
        P_imp      grid import [kW, >=0] = max(Pg, 0)
        P_exp      grid export [kW, >=0] = max(-Pg, 0)
        price_imp  import price at this step [€/kWh]
        price_exp  export price at this step [€/kWh]
        r_eco      instantaneous economic reward [€]
        r_soc      SoC penalty (0 when no violation)
        r_bat_power  quadratic battery power penalty = -sigma_bat * (Pb/Pb_max)² (0 when sigma_bat = 0)
        r_spread   spread penalty on useless discharge (0 when spread_penalty=False)
        r_store    PBRS store-value shaping = γ·Φ(s')−Φ(s), Φ=σ·v(t)·E_util (0 when store_value=False)
        r_export_stock  one-sided opportunity-cost penalty on battery-sourced export
                   = -σ_xs·max(0, max_future_import − price_exp)·e_batt_exp·dt
                   (0 when sigma_export_stock=0; 0 on optimal trajectories)
        reward     total step reward = r_eco + r_soc + r_curt + r_bat_power + r_spread + r_phantom + r_store + r_export_stock
        Pcurt      PV curtailment [kW, >=0] — power shed by inverter when
                   surplus exceeds max_export and battery is saturated
        Pph        phantom power [kW, >=0] — unserved load conjured to close the
                   balance when import+battery can't meet demand (Duchaud-JL slack)
        r_phantom  phantom-power penalty [€] = -phantom_penalty * Pph * dt (0 when Pph=0)

    MILP-ONLY (NaN for RL rollouts):
        Pb_charge      auxiliary charging magnitude [kW, >=0]
        Pb_discharge   auxiliary discharging magnitude [kW, >=0]
        b_int          binary mutual-exclusion flag (0 or 1, stored as float)

    RL-ONLY (NaN for MILP rollouts):
        action_raw   raw action in [-1, 1] before kW scaling
        Pb_command   scaled command [kW] before battery clamp

Sign convention (must match `envs/components/battery.py` and `envs/base_microgrid_env.py`):
- Pp: PV production [kW], always >= 0.
- Pl: load demand [kW], magnitude (>= 0). Plotted as -Pl on the consumption side.
- SoC: state of charge in PERCENT (0-100), per the spec for this monitoring layer.
       Note: `battery.py` keeps SoC as a fraction in [0, 1]; multiply by 100 before insert.
- Pb: battery power [kW]. Pb > 0 = DISCHARGING = power injected into the bus.
                         Pb < 0 = CHARGING   = power drawn from the bus.
- Pg: grid power [kW].   Pg > 0 = IMPORT from grid (power into bus).
                         Pg < 0 = EXPORT to grid (power out of bus).
       Defined by Kirchhoff at the bus: Pg = Pl - Pp - Pb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Column indices for the internal numpy array.
# Shared columns
COL_T            = 0
COL_PP           = 1
COL_PL           = 2
COL_SOC          = 3
COL_PB           = 4
COL_PG           = 5
COL_P_IMP        = 6
COL_P_EXP        = 7
COL_PRICE_IMP    = 8
COL_PRICE_EXP    = 9
COL_R_ECO        = 10
COL_R_SOC        = 11
COL_REWARD       = 12
# MILP-only columns
COL_PB_CHARGE    = 13
COL_PB_DISCHARGE = 14
COL_B_INT        = 15
# RL-only columns
COL_ACTION_RAW   = 16
COL_PB_COMMAND   = 17
# Shared curtailment column (appended last to avoid renumbering above)
COL_PCURT        = 18
# New reward components (appended last to avoid renumbering above)
COL_R_BAT_POWER  = 19
COL_R_SPREAD     = 20
# Phantom power (slack à la Duchaud-JL) + its reward penalty — shared RL/MILP,
# appended last to keep all preceding indices stable.
COL_PPH          = 21
COL_R_PHANTOM    = 22
# PBRS store-value shaping reward (RL-only, 0/NaN otherwise) — appended last.
COL_R_STORE      = 23
# One-sided export-stock opportunity-cost penalty (RL-only, 0/NaN otherwise) — appended last.
COL_R_EXPORT_STOCK = 24

COLUMN_NAMES = [
    "t",
    "Pp", "Pl", "SoC", "Pb", "Pg",
    "P_imp", "P_exp",
    "price_imp", "price_exp",
    "r_eco", "r_soc", "reward",
    "Pb_charge", "Pb_discharge", "b_int",
    "action_raw", "Pb_command",
    "Pcurt",
    "r_bat_power", "r_spread",
    "Pph", "r_phantom",
    "r_store",
    "r_export_stock",
]
NUM_COLUMNS = len(COLUMN_NAMES)

# Required keys for backward compatibility. Anything else is optional and
# defaults to NaN.
_REQUIRED_KEYS = ("pp", "pl", "soc", "pb", "pg")

# Mapping lowercase-key → column index for optional columns.
_OPTIONAL_KEYS = {
    "p_imp":        COL_P_IMP,
    "p_exp":        COL_P_EXP,
    "price_imp":    COL_PRICE_IMP,
    "price_exp":    COL_PRICE_EXP,
    "r_eco":        COL_R_ECO,
    "r_soc":        COL_R_SOC,
    "reward":       COL_REWARD,
    "pb_charge":    COL_PB_CHARGE,
    "pb_discharge": COL_PB_DISCHARGE,
    "b_int":        COL_B_INT,
    "action_raw":   COL_ACTION_RAW,
    "pb_command":   COL_PB_COMMAND,
    "pcurt":        COL_PCURT,
    "r_bat_power":  COL_R_BAT_POWER,
    "r_spread":     COL_R_SPREAD,
    "pph":          COL_PPH,
    "r_phantom":    COL_R_PHANTOM,
    "r_store":      COL_R_STORE,
    "r_export_stock": COL_R_EXPORT_STOCK,
}


class MonitoringTable:
    """Indexed-write monitoring buffer accumulated during a deployment-style rollout.

    Usage example:
        mt = MonitoringTable(n_steps=144,
                             start_time=pd.Timestamp("2025-05-13"),
                             delta_t_min=10)
        for k in range(144):
            mt.insert(k, {'pp': ..., 'pl': ..., 'soc': ..., 'pb': ..., 'pg': ...,
                          'r_eco': ..., 'reward': ...})  # optional extras OK
        df = mt.to_dataframe()
        mt.to_csv("runs/monitoring_table.csv")
        cost = mt.get_total_cost(buy_price=0.15, sell_price=0.15)
    """

    def __init__(
        self,
        n_steps: int,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        delta_t_min: float = 10.0,
    ) -> None:
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")
        self.n_steps = int(n_steps)
        self.delta_t_min = float(delta_t_min)
        self.delta_t_h = self.delta_t_min / 60.0

        if start_time is None:
            start_time = pd.Timestamp("1970-01-01")
        self.start_time = pd.Timestamp(start_time)

        self._data: np.ndarray = np.full(
            (self.n_steps, NUM_COLUMNS), np.nan, dtype=np.float64
        )
        # Pre-compute uniformly spaced timestamps (unix seconds).
        self._fill_time_column()

    # ------------------------------------------------------------------ utils

    def _fill_time_column(self) -> None:
        start_s = self.start_time.timestamp()
        dt_s = self.delta_t_min * 60.0
        self._data[:, COL_T] = start_s + np.arange(self.n_steps) * dt_s

    # ------------------------------------------------------------------ API

    def reset(self, n_steps: Optional[int] = None,
              start_time: Optional[Union[str, pd.Timestamp]] = None,
              delta_t_min: Optional[float] = None) -> None:
        """Reinitialize the buffer for a new simulation run.

        All four arguments are optional; passing None keeps the previous value.
        """
        if n_steps is not None:
            self.n_steps = int(n_steps)
        if start_time is not None:
            self.start_time = pd.Timestamp(start_time)
        if delta_t_min is not None:
            self.delta_t_min = float(delta_t_min)
            self.delta_t_h = self.delta_t_min / 60.0

        self._data = np.full((self.n_steps, NUM_COLUMNS), np.nan, dtype=np.float64)
        self._fill_time_column()

    def insert(self, step_idx: int, state_dict: dict) -> None:
        """Write a single row at `step_idx` from a state dict.

        Required keys (case-insensitive): pp, pl, soc, pb, pg. SoC is expected
        in percent (0-100). Optional keys (silently ignored if unknown):
        p_imp, p_exp, price_imp, price_exp, r_eco, r_soc, reward,
        pb_charge, pb_discharge, b_int, action_raw, pb_command. Missing
        optional columns stay at NaN.
        """
        if not 0 <= step_idx < self.n_steps:
            raise IndexError(
                f"step_idx={step_idx} out of range [0, {self.n_steps})"
            )

        sd_lower = {k.lower(): v for k, v in state_dict.items()}
        try:
            self._data[step_idx, COL_PP] = float(sd_lower["pp"])
            self._data[step_idx, COL_PL] = float(sd_lower["pl"])
            self._data[step_idx, COL_SOC] = float(sd_lower["soc"])
            self._data[step_idx, COL_PB] = float(sd_lower["pb"])
            self._data[step_idx, COL_PG] = float(sd_lower["pg"])
        except KeyError as e:
            raise KeyError(
                f"state_dict missing key {e}; required: pp, pl, soc, pb, pg"
            ) from None

        for key, col in _OPTIONAL_KEYS.items():
            if key in sd_lower and sd_lower[key] is not None:
                try:
                    self._data[step_idx, col] = float(sd_lower[key])
                except (TypeError, ValueError):
                    # Silently leave as NaN on unparseable input — keeps the
                    # caller's hot loop free of try/except for optional fields.
                    pass

        # Optional explicit timestamp override (handles non-uniform sampling).
        if "t" in sd_lower or "timestamp" in sd_lower:
            ts_raw = sd_lower.get("t", sd_lower.get("timestamp"))
            self._data[step_idx, COL_T] = pd.Timestamp(ts_raw).timestamp()

    # ------------------------------------------------------------------ views

    @property
    def data(self) -> np.ndarray:
        """Raw (n_steps, NUM_COLUMNS) numpy array."""
        return self._data

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with named columns and a UTC DatetimeIndex."""
        times = pd.to_datetime(self._data[:, COL_T], unit="s", utc=True)
        cols = {name: self._data[:, i] for i, name in enumerate(COLUMN_NAMES) if i != COL_T}
        df = pd.DataFrame(cols, index=times)
        df.index.name = "t"
        return df

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Export the monitoring table to CSV (t column + all data columns).

        Returns the resolved Path written.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out_path)
        return out_path

    # ------------------------------------------------------------------ cost

    def get_total_cost(self, buy_price, sell_price, n_steps: int | None = None) -> float:
        """Total optimization cost over the rollout, in the same currency as the prices.

        cost = sum( max(Pg, 0) * buy_price * dt )  -  sum( max(-Pg, 0) * sell_price * dt )

        Pg > 0 means importing from the grid (we pay `buy_price`).
        Pg < 0 means exporting to the grid (we earn `sell_price`).
        NaN rows (unfilled timesteps) are ignored.

        Args:
            buy_price:  scalar float or array of shape (n_steps,) in EUR/kWh.
            sell_price: scalar float or array of shape (n_steps,) in EUR/kWh.
            n_steps: Score only the first ``n_steps`` decision steps. ``None``
                (default) scores the whole rollout. Used for N-jours / scoring
                sur N−1 : on simule la fenêtre complète mais on n'évalue le coût
                que sur les premiers pas, pour exclure le dump de fin d'horizon.
        """
        pg_full = self._data[:, COL_PG]
        if n_steps is not None:
            pg_full = pg_full[:n_steps]
        mask = ~np.isnan(pg_full)
        if not np.any(mask):
            return 0.0
        pg = pg_full[mask]

        buy = np.asarray(buy_price)
        sell = np.asarray(sell_price)
        if buy.ndim > 0:
            buy = buy[: len(pg_full)][mask]
        if sell.ndim > 0:
            sell = sell[: len(pg_full)][mask]

        import_kw = np.maximum(pg, 0.0)
        export_kw = np.maximum(-pg, 0.0)
        cost = float(np.sum(import_kw * buy - export_kw * sell) * self.delta_t_h)
        return cost

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        n_filled = int(np.sum(~np.isnan(self._data[:, COL_PG])))
        return (
            f"MonitoringTable(n_steps={self.n_steps}, "
            f"filled={n_filled}, delta_t_min={self.delta_t_min})"
        )
