"""PV source model with perfect-foresight forecast for exp01.

Loads irradiance from CSV and converts to PV power:
  P_pv_kw = Global30_kW * surface_m2 * eta_ref
  Global30_kW is the global 30° irradiance in kW/m2 (from CSV Pyranometer data)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PVSource:
    def __init__(self, cfg: dict, data_cfg: dict):
        self.surface_m2 = cfg["surface_m2"]
        self.eta_ref = cfg["eta_ref"]
        self.pv_type = cfg["type"]

        df = pd.read_csv(data_cfg["pv_csv"], parse_dates=["Time"])
        self.irradiance = df[data_cfg["pv_column"]].values.astype(np.float64)
        self.hour_sin = df["hour_sin"].values.astype(np.float64)
        self.hour_cos = df["hour_cos"].values.astype(np.float64)
        self.doy_sin = df["doy_sin"].values.astype(np.float64)
        self.doy_cos = df["doy_cos"].values.astype(np.float64)
        self.dates = pd.to_datetime(df["date"]).values
        self.timestamps = df["Time"].values

        self.pv_power = self.irradiance * self.surface_m2 * self.eta_ref
        self.pv_power = np.maximum(self.pv_power, 0.0)

        max_pv = self.pv_power.max()
        if max_pv > 50.0:
            logger.warning(f"Max PV power = {max_pv:.1f} kW — seems high, check surface/eta")
        if max_pv < 0.01:
            logger.warning(f"Max PV power = {max_pv:.4f} kW — seems very low")

        self.n_steps = len(self.pv_power)

    def set_data_slice(self, indices: np.ndarray):
        """Restrict this source to a subset of timestep indices, used for train/test splits."""
        self.irradiance = self.irradiance[indices]
        self.hour_sin = self.hour_sin[indices]
        self.hour_cos = self.hour_cos[indices]
        self.doy_sin = self.doy_sin[indices]
        self.doy_cos = self.doy_cos[indices]
        self.dates = self.dates[indices]
        self.timestamps = self.timestamps[indices]
        self.pv_power = self.pv_power[indices]
        self.n_steps = len(self.pv_power)

    def get_irradiance(self, step_index: int) -> float:
        """Return the actual PV power output in kW at the given step (magnitude, always >= 0).
           This is considered after the surface and efficiency conversion (kW/m2 * m2 * eta)."""
        return float(self.pv_power[step_index])

    def get_forecast(self, step_index: int, horizon_steps: int) -> np.ndarray:
        """Return future PV power values for the next horizon_steps.
        If the horizon extends beyond the available data, we stop pad with the last known value.

        For exp01 (perfect foresight), returns the actual future values.
        """
        end = min(step_index + 1 + horizon_steps, self.n_steps)
        forecast = self.pv_power[step_index + 1: end]
        if len(forecast) < horizon_steps:
            forecast = np.pad(forecast, (0, horizon_steps - len(forecast)), mode="edge")
        return forecast.astype(np.float32)

    def get_temporal_features(self, step_index: int) -> tuple:
        """Return (hour_sin, hour_cos, doy_sin, doy_cos) at step_index."""
        return (
            self.hour_sin[step_index],
            self.hour_cos[step_index],
            self.doy_sin[step_index],
            self.doy_cos[step_index],
        )
