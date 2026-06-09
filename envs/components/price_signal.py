"""Time-varying electricity price signal for import/export pricing.

Inspired by the MATLAB reference:
    Param.fun_prix_reseau = @(t, E) (5 + sin(t)') .* E' .* (E'>0)
                                   + ones(size(t')) .* E' .* (E'<=0);

which defines a sinusoidal import price and a flat export price.

Supported types (config["price_type"]):
    "fixed"      — constant import/export prices (exp01 backward-compatible)
    "sinusoidal" — import_price(t) = base_import + amplitude * sin(2π*hour/period_h)
                   export_price(t) = base_export  (constant)
    "step_hp_hc" — import HP/HC (+ créneau midi optionnel via hc_midday),
                   export depuis la grille CoutsProd (heure × saison × semaine/WE)
    "step_pointe"— import HC nuit / HP journée / Pointe soir (3 niveaux),
                   export depuis la grille CoutsProd
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Grille CoutsProd — coûts de production Corse, prix d'export (vente) en €/MWh.
# 24 lignes = heures 0..23 ; prix maintenu constant intra-heure.
# 4 colonnes = [Semaine-HauteSaison, WeekEnd-HauteSaison,
#               Semaine-BasseSaison, WeekEnd-BasseSaison].
_COUTS_PROD = np.array(
    [
        172, 178, 147, 147,
        151, 153, 139, 138,
        144, 145, 136, 134,
        142, 142, 136, 134,
        142, 142, 137, 134,
        145, 143, 140, 135,
        154, 146, 148, 138,
        197, 161, 162, 144,
        231, 179, 165, 147,
        230, 189, 161, 148,
        216, 186, 157, 145,
        201, 181, 152, 143,
        191, 175, 149, 140,
        186, 166, 147, 138,
        181, 164, 146, 137,
        180, 164, 146, 137,
        189, 170, 149, 140,
        230, 200, 157, 146,
        348, 295, 176, 160,
        424, 356, 199, 175,
        427, 366, 193, 173,
        442, 392, 199, 183,
        380, 367, 188, 180,
        225, 217, 168, 165,
    ],
    dtype=np.float64,
).reshape(24, 4)

# Mois de haute saison Corse : été (juillet-août) + hiver (novembre-février).
_HAUTE_SAISON_MONTHS = (1, 2, 7, 8, 11, 12)


def _coutsprod_export_prices(timestamps) -> np.ndarray:
    """Prix d'export €/kWh depuis la grille CoutsProd (heure × saison × semaine/WE).

    Sélection de colonne :
        col = (0 si haute saison, 2 si basse saison) + (1 si week-end, 0 si semaine).
    Les valeurs CoutsProd sont en €/MWh → divisées par 1000 pour obtenir des €/kWh.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    is_haute = np.isin(ts.month.to_numpy(), _HAUTE_SAISON_MONTHS)
    weekend = (ts.dayofweek.to_numpy() >= 5).astype(int)
    col = np.where(is_haute, 0, 2) + weekend
    return (_COUTS_PROD[ts.hour.to_numpy(), col] / 1000.0).astype(np.float64)


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

        elif self.price_type == "step_hp_hc":          # modèle 2 : HP / HC
            p_hp = float(config["price_hp"])           # ex. 0.2475
            p_hc = float(config["price_hc"])           # ex. 0.1895
            hc_midday = bool(config.get("hc_midday", False))

            hours = np.array(
                [pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
                 for ts in timestamps],
                dtype=np.float64,
            )
            is_hc = (hours >= 23) | (hours < 7)
            if hc_midday:
                is_hc |= (hours >= 12) & (hours < 15)

            self.import_prices = np.where(is_hc, p_hc, p_hp)
            self.export_prices = _coutsprod_export_prices(timestamps)
            self.has_forecast = True

        elif self.price_type == "step_pointe":         # modèle 1 : HC / HP / Pointe
            p_hc = float(config["price_hc"])           # ex. 0.19
            p_hp = float(config["price_hp"])           # ex. 0.25
            p_pointe = float(config["price_pointe"])   # ex. 0.35

            hours = np.array(
                [pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
                 for ts in timestamps],
                dtype=np.float64,
            )
            is_hc = (hours >= 23) | (hours < 7)
            is_pointe = (hours >= 18) & (hours < 22)

            imp = np.full(n, p_hp, dtype=np.float64)   # HP par défaut (7-18 et 22-23)
            imp[is_pointe] = p_pointe
            imp[is_hc] = p_hc
            self.import_prices = imp
            self.export_prices = _coutsprod_export_prices(timestamps)
            self.has_forecast = True

        else:
            raise ValueError(
                f"Unknown price_type: '{self.price_type}'. "
                "Use 'fixed', 'sinusoidal', 'step_hp_hc' or 'step_pointe'."
            )

    # ------------------------------------------------------------------ API

    def get_import_price(self, step_idx: int) -> float:
        return float(self.import_prices[step_idx])

    def get_export_price(self, step_idx: int) -> float:
        return float(self.export_prices[step_idx])

    def get_import_forecast(self, step_idx: int, horizon: int) -> np.ndarray:
        """Return import prices for the next `horizon` steps starting at step_idx.

        If the window exceeds the array length the last known price is repeated.
        """
        return self._forecast(self.import_prices, step_idx, horizon)

    def get_export_forecast(self, step_idx: int, horizon: int) -> np.ndarray:
        """Return export prices for the next `horizon` steps starting at step_idx.

        Miroir de ``get_import_forecast`` : le prix d'export (grille CoutsProd) varie
        fortement sur la journée — son forecast est le signal d'arbitrage pour timer les
        exports. Pad avec la dernière valeur connue si l'horizon dépasse les données.
        """
        return self._forecast(self.export_prices, step_idx, horizon)

    @staticmethod
    def _forecast(arr: np.ndarray, step_idx: int, horizon: int) -> np.ndarray:
        end = step_idx + horizon
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
