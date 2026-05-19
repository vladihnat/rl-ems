"""Time-varying electricity price signal for import/export pricing.

Inspired by the MATLAB reference:
    Param.fun_prix_reseau = @(t, E) (5 + sin(t)') .* E' .* (E'>0)
                                   + ones(size(t')) .* E' .* (E'<=0);

which defines a sinusoidal import price and a flat export price.

Supported types (config["type"]):
    "fixed"      — constant import/export prices (exp01 backward-compatible)
    "sinusoidal" — import_price(t) = base_import + amplitude * sin(2π*hour/period_h)
                   export_price(t) = base_export  (constant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PriceSignal:
    """Pre-computed electricity price arrays aligned with PVSource timestamps."""

    def __init__(self, config: dict, timestamps, delta_t_min: float) -> None:
        """
        Args:
            config:      The ``grid`` section of the experiment YAML.
            timestamps:  Iterable of timestamps aligned with PVSource (length = n_steps).
            delta_t_min: Timestep duration in minutes (unused here, kept for symmetry).
        """
        self.price_type: str = config.get("price_type", "fixed")
        self.delta_t_min = delta_t_min
        n = len(timestamps)

        if self.price_type == "fixed":
            p_imp = float(config["price_import"])
            p_exp = float(config["price_export"])
            self.import_prices: np.ndarray = np.full(n, p_imp, dtype=np.float64)
            self.export_prices: np.ndarray = np.full(n, p_exp, dtype=np.float64)
            self.has_forecast: bool = False

        elif self.price_type == "sinusoidal":
            base_imp = float(config["base_import"])
            amplitude = float(config.get("amplitude", 0.05))
            period_h = float(config.get("period_h", 24.0))
            base_exp = float(config["base_export"])

            hours = np.array(
                [pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
                 for ts in timestamps],
                dtype=np.float64,
            )
            self.import_prices = base_imp + amplitude * np.sin(
                2.0 * np.pi * hours / period_h
            )
            self.export_prices = np.full(n, base_exp, dtype=np.float64)
            self.has_forecast = True

        else:
            raise ValueError(f"Unknown price_type: '{self.price_type}'. Use 'fixed' or 'sinusoidal'.")

    # ------------------------------------------------------------------ API

    def get_import_price(self, step_idx: int) -> float:
        return float(self.import_prices[step_idx])

    def get_export_price(self, step_idx: int) -> float:
        return float(self.export_prices[step_idx])

    def get_import_forecast(self, step_idx: int, horizon: int) -> np.ndarray:
        """Return import prices for the next `horizon` steps starting at step_idx.

        If the window exceeds the array length the last known price is repeated.
        """
        end = step_idx + horizon
        arr = self.import_prices
        if end <= len(arr):
            return arr[step_idx:end].astype(np.float32)
        tail = arr[step_idx:].astype(np.float32)
        pad = int(end - len(arr))
        return np.pad(tail, (0, pad), constant_values=tail[-1] if len(tail) else 0.0)

    def __repr__(self) -> str:
        return (
            f"PriceSignal(type={self.price_type!r}, "
            f"import=[{self.import_prices.min():.4f}, {self.import_prices.max():.4f}], "
            f"export={self.export_prices[0]:.4f})"
        )
