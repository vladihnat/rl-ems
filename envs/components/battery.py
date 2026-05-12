"""Battery model following the simulate_microgrid.m sign convention.

Sign convention:
  Pb > 0  -> DISCHARGING (battery provides power to the bus)
  Pb < 0  -> CHARGING   (battery absorbs power from the bus)
"""

import numpy as np


class BatteryModel:
    def __init__(self, cfg: dict):
        self.capacity_kwh = cfg["capacity_kwh"]
        self.soc_min = cfg["soc_min"]
        self.soc_max = cfg["soc_max"]
        self.max_charge_kw = cfg["max_charge_kw"]
        self.max_discharge_kw = cfg["max_discharge_kw"]
        self.eta_charge = cfg["efficiency_charge"]
        self.eta_discharge = cfg["efficiency_discharge"]
        self.init_soc = cfg["init_soc"]
        # cost_cycle c'est le prix en euro par kWh de cycle de batterie (charge/décharge).
        # Variable non utilisée dans l'environnement de base, mais peut être intégrée
        # dans la fonction de récompense pour pénaliser l'usure liée aux cycles de batterie 
        self.cost_cycle = cfg["cost_cycle"]
        self.soc = self.init_soc

    def reset(self):
        self.soc = self.init_soc
        return self.soc

    def step(self, action_kw: float, delta_t_h: float):
        """Apply battery command and return effective power and new SoC.

        Args:
            action_kw: Requested power in kW. Positive = discharge, negative = charge.
            delta_t_h: Timestep duration in hours.

        Returns:
            (Pb_effective, new_soc): Effective power delivered/absorbed in kW, new SoC.
        """
        Pb = np.clip(action_kw, -self.max_charge_kw, self.max_discharge_kw)

        if Pb < 0:
            nu = self.eta_charge
        else:
            nu = 1.0 / self.eta_discharge

        soc_old = self.soc
        soc_new = soc_old - Pb * nu * delta_t_h / self.capacity_kwh
        soc_new = np.clip(soc_new, 0.0, 1.0)

        dE = (soc_old - soc_new) * self.capacity_kwh
        Pb_effective = dE / delta_t_h / nu

        self.soc = float(soc_new)
        return float(Pb_effective), self.soc


def _p_grid(load_kw, pv_kw, pb_kw):
    """Compute grid power: P_grid = load - pv - Pb (all magnitudes, signs applied)."""
    return load_kw - pv_kw - pb_kw


if __name__ == "__main__":
    assert _p_grid(5, 3, 0) == 2.0, "Import case failed"
    assert _p_grid(2, 5, 0) == -3.0, "Export case failed"
    print("P_grid sign convention tests passed.")

    cfg = {
        "capacity_kwh": 24.0,
        "soc_min": 0.2,
        "soc_max": 0.9,
        "max_charge_kw": 7.0,
        "max_discharge_kw": 7.0,
        "efficiency_charge": 0.9,
        "efficiency_discharge": 0.9,
        "init_soc": 0.5,
        "cost_cycle": 0.02,
    }
    bat = BatteryModel(cfg)

    # Matlab-style test: request Pb=100kW discharge (way above max), dt=1min
    dt_h = 1.0 / 60.0
    Pb_eff, new_soc = bat.step(100.0, dt_h)
    print(f"Requested Pb=100 kW (clipped to {cfg['max_discharge_kw']} kW)")
    print(f"  dt = {dt_h*60:.1f} min")
    print(f"  SoC: {cfg['init_soc']:.4f} -> {new_soc:.4f}")
    print(f"  Pb_effective = {Pb_eff:.4f} kW")

    # Verify clipping: effective power should not exceed max_discharge
    assert abs(Pb_eff - cfg["max_discharge_kw"]) < 0.01, "Discharge clipping failed"
    print("Battery unit tests passed.")
