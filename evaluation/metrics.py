"""Compute episode-level metrics from a completed rollout."""

import numpy as np


def compute_metrics(
    history: dict,
    delta_t_h: float,
    soc_min: float,
    soc_max: float,
    price_import: float,
    price_export: float,
    phantom_penalty: float = 1e3,
) -> dict:
    """Compute summary metrics from episode history arrays.

    Args:
        history: dict with keys P_grid, soc, load_t, etc. (numpy arrays).
            If ``P_phantom`` is present, phantom-aware metrics are added.
        delta_t_h: timestep duration in hours.
        soc_min, soc_max: safe SoC bounds.
        price_import, price_export: EUR/kWh prices — scalar or array of shape (T,).
        phantom_penalty: VOLL (€/kWh) charged on unserved load to build
            ``net_cost_adjusted`` — the only cost comparable across policies
            that leave different amounts of load unserved.

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

    # Phantom power (unserved load conjured to close the balance, à la Duchaud-JL).
    # net_cost above only counts the *capped* grid flows, so a policy that leaves
    # more load unserved imports less and looks cheaper. net_cost_adjusted charges
    # that unserved load at `phantom_penalty` (VOLL) to restore comparability.
    if "P_phantom" in history:
        phantom = np.asarray(history["P_phantom"])
        phantom_energy_kwh = float(np.sum(phantom) * delta_t_h)
        phantom_steps = int(np.sum(phantom > 1e-6))
    else:
        phantom_energy_kwh = 0.0
        phantom_steps = 0

    if total_load_energy > 0:
        served_load_ratio = float(1.0 - phantom_energy_kwh / total_load_energy)
    else:
        served_load_ratio = 1.0
    net_cost_adjusted = float(net_cost + phantom_penalty * phantom_energy_kwh)

    return {
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "net_cost": net_cost,
        "net_cost_adjusted": net_cost_adjusted,
        "phantom_energy_kwh": phantom_energy_kwh,
        "phantom_steps": phantom_steps,
        "served_load_ratio": served_load_ratio,
        "soc_violations": soc_violations,
        "self_consumption_rate": float(self_consumption_rate),
        "peak_grid_import": peak_grid_import,
        "energy_imported_kwh": float(energy_imported),
        "energy_exported_kwh": float(energy_exported),
        "total_reward": total_reward,
    }
