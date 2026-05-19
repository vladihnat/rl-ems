"""Pre-allocated monitoring table for post-training simulation rollouts.

Python port of the MATLAB `MicroGrid.Monitoring` matrix and `monitoTable`
dependent property. One row per RL decision step; the row is written
(indexed) at the moment the agent acts on the env.

Column layout (matches MATLAB exactly):
    [t (unix seconds, float64), Pp, Pl, SoC, Pb, Pg]

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
# MATLAB equivalent: MicroGrid.m:41-46  array2table(..., {'t','Pp','Pl','SoC','Pb','Pg'})
COL_T, COL_PP, COL_PL, COL_SOC, COL_PB, COL_PG = 0, 1, 2, 3, 4, 5
COLUMN_NAMES = ["t", "Pp", "Pl", "SoC", "Pb", "Pg"]
NUM_COLUMNS = 6


class MonitoringTable:
    """Indexed-write monitoring buffer accumulated during a deployment-style rollout.

    Usage:
        mt = MonitoringTable(n_steps=144,
                             start_time=pd.Timestamp("2025-05-13"),
                             delta_t_min=10)
        for k in range(144):
            mt.insert(k, {'pp': ..., 'pl': ..., 'soc': ..., 'pb': ..., 'pg': ...})
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

        # Pre-allocate (n_steps, 6) NaN array.
        # MATLAB equivalent: MPC/simu.m:15  obj.MicroGrid.Monitoring = nan(nPoints, 6);
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

        Mirror of MATLAB `insert_monitoring_data` (MicroGridSimu/insert_monitoring_data.m).
        `state_dict` must contain keys: pp, pl, soc, pb, pg (case-insensitive). SoC is
        expected in percent (0-100).
        """
        if not 0 <= step_idx < self.n_steps:
            raise IndexError(
                f"step_idx={step_idx} out of range [0, {self.n_steps})"
            )

        # Accept any casing so the call site can use either pp/Pp or PV/pv etc.
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

        # Optional explicit timestamp override (handles non-uniform sampling).
        if "t" in sd_lower or "timestamp" in sd_lower:
            ts_raw = sd_lower.get("t", sd_lower.get("timestamp"))
            self._data[step_idx, COL_T] = pd.Timestamp(ts_raw).timestamp()

    # ------------------------------------------------------------------ views

    @property
    def data(self) -> np.ndarray:
        """Raw (n_steps, 6) numpy array. Equivalent to MATLAB `obj.Monitoring`."""
        return self._data

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with named columns and a UTC DatetimeIndex.

        Mirror of MATLAB `MicroGrid.monitoTable` (MicroGrid.m:41-46 + num2dt).
        """
        # MATLAB equivalent: MicroGrid.m:41-46  table2timetable + num2dt(T.t)
        times = pd.to_datetime(self._data[:, COL_T], unit="s", utc=True)
        df = pd.DataFrame(
            {
                "Pp": self._data[:, COL_PP],
                "Pl": self._data[:, COL_PL],
                "SoC": self._data[:, COL_SOC],
                "Pb": self._data[:, COL_PB],
                "Pg": self._data[:, COL_PG],
            },
            index=times,
        )
        df.index.name = "t"
        return df

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Export the monitoring table to CSV with a `t` column (ISO 8601) + Pp/Pl/SoC/Pb/Pg.

        Returns the resolved Path written.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out_path)
        return out_path

    # ------------------------------------------------------------------ cost

    def get_total_cost(self, buy_price: float, sell_price: float) -> float:
        """Total optimization cost over the rollout, in the same currency as the prices.

        cost = sum( max(Pg, 0) * buy_price * dt )  -  sum( max(-Pg, 0) * sell_price * dt )

        Pg > 0 means importing from the grid (we pay `buy_price`).
        Pg < 0 means exporting to the grid (we earn `sell_price`).
        NaN rows (unfilled timesteps) are ignored.
        """
        pg = self._data[:, COL_PG]
        mask = ~np.isnan(pg)
        if not np.any(mask):
            return 0.0
        pg = pg[mask]

        import_kw = np.maximum(pg, 0.0)
        export_kw = np.maximum(-pg, 0.0)
        cost = float(np.sum(import_kw * buy_price - export_kw * sell_price) * self.delta_t_h)
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
