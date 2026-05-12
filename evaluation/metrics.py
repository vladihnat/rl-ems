"""Compute episode-level metrics from a completed rollout."""

import numpy as np


def compute_metrics(
    history: dict,
    delta_t_h: float,
    soc_min: float,
    soc_max: float,
    price_import: float,
    price_export: float,
) -> dict:
    """Compute summary metrics from episode history arrays.

    Args:
        history: dict with keys P_grid, soc, load_t, etc. (numpy arrays).
        delta_t_h: timestep duration in hours.
        soc_min, soc_max: safe SoC bounds.
        price_import, price_export: EUR/kWh prices.

    Returns:
        dict of scalar metrics.
    """
    P_grid = np.asarray(history["P_grid"])
    soc = np.asarray(history["soc"])
    load_t = np.asarray(history["load_t"])

    import_power = np.maximum(P_grid, 0.0)
    export_power = np.maximum(-P_grid, 0.0)

    energy_imported = np.sum(import_power) * delta_t_h
    energy_exported = np.sum(export_power) * delta_t_h
    total_load_energy = np.sum(load_t) * delta_t_h

    total_cost = float(np.sum(import_power * price_import * delta_t_h))
    total_revenue = float(np.sum(export_power * price_export * delta_t_h))
    net_cost = total_cost - total_revenue

    soc_violations = int(np.sum((soc < soc_min) | (soc > soc_max)))

    if total_load_energy > 0:
        self_consumption_rate = 1.0 - energy_imported / total_load_energy
    else:
        self_consumption_rate = 1.0

    peak_grid_import = float(np.max(import_power))
    total_reward = float(np.sum(history["reward"]))

    return {
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "net_cost": net_cost,
        "soc_violations": soc_violations,
        "self_consumption_rate": float(self_consumption_rate),
        "peak_grid_import": peak_grid_import,
        "energy_imported_kwh": float(energy_imported),
        "energy_exported_kwh": float(energy_exported),
        "total_reward": total_reward,
    }
