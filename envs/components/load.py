"""Load model: fixed sinusoidal load matching the Matlab abs(sin(t)) * 1e3 W convention."""

import numpy as np
import pandas as pd


class LoadModel:
    def __init__(self, cfg: dict, n_steps: int, delta_t_min: float, timestamps=None):
        self.load_type = cfg["type"]
        self.base_load_kw = cfg["base_load_kw"]
        self.n_steps = n_steps
        self.delta_t_min = delta_t_min

        if self.load_type == "fixed":
            self.load_profile = self._generate_sinusoidal(n_steps, delta_t_min, timestamps)
        else:
            # TODO : Implementer des charges variables à partir de données réelles (ou de modèles plus complexes)
            raise ValueError(f"Unknown load type: {self.load_type}")

    def _generate_sinusoidal(self, n_steps, delta_t_min, timestamps=None):
        """P_load(t) = base_load_kw * |sin(2*pi*t_hours / 48)|
        Choice of sine to simulate a load profile with a peak at noon"""
        if timestamps is not None:
            ts = pd.DatetimeIndex(timestamps)
            t_hours = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
            t_hours = t_hours.values.astype(np.float64)
        else:
            t_hours = np.arange(n_steps) * delta_t_min / 60.0

        profile = self.base_load_kw * np.abs(np.sin(2.0 * np.pi * t_hours / 48.0))
        return profile

    def set_data_slice(self, indices: np.ndarray):
        """Restrict to a subset of timestep indices.
        Useful for creating train/test splits from a single load profile."""
        self.load_profile = self.load_profile[indices]
        self.n_steps = len(self.load_profile)

    def get_load(self, step_index: int) -> float:
        """Return load in kW (magnitude, always >= 0)."""
        return float(self.load_profile[step_index])

    def get_forecast(self, step_index: int, horizon_steps: int) -> np.ndarray:
        """Return future load values. Deterministic for fixed load with perfect foresight."""
        end = min(step_index + 1 + horizon_steps, self.n_steps)
        forecast = self.load_profile[step_index + 1: end]
        if len(forecast) < horizon_steps:
            forecast = np.pad(forecast, (0, horizon_steps - len(forecast)), mode="edge")
        return forecast.astype(np.float32)
