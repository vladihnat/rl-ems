"""Compute episode-level metrics from a completed rollout."""

import numpy as np
import pandas as pd

# Mois (1-12) comptés comme "été". Le reste (nov-avril) = "hiver".
# Éditable : c'est la seule définition du découpage saisonnier.
SUMMER_MONTHS = {5, 6, 7, 8, 9, 10}


def season_labels(timestamps) -> np.ndarray:
    """Mappe chaque timestamp vers 'ete' ou 'hiver' selon SUMMER_MONTHS.

    Args:
        timestamps: séquence de datetimes (DatetimeIndex, array datetime64, ...).
    Returns:
        np.ndarray[str] de même longueur, valeurs dans {'ete', 'hiver'}.
    """
    months = pd.DatetimeIndex(timestamps).month.to_numpy()
    return np.where(np.isin(months, list(SUMMER_MONTHS)), "ete", "hiver")


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


def boundary_soc_credit(
    soc_end_frac: float,
    soc_init_frac: float,
    capacity_kwh: float,
    eta_discharge: float,
    store_value: float,
) -> float:
    """Valeur (€) de l'énergie batterie portée au BORD d'une fenêtre scorée.

    Quand on score un préfixe STRICT d'un rollout (``score_days`` < horizon de plan),
    l'énergie restée dans la batterie au bord servira AU-DELÀ de la fenêtre (éviter un
    import futur). L'ignorer biaise la comparaison : une politique qui draine la batterie
    avant le bord paraît moins chère (artefact pur sous prix fixes, cf. exp22 hiver_haute :
    le RL « bat » le MILP de −61 % alors qu'ils sont à égalité sur le plein horizon). On
    crédite donc ce stock, SYMÉTRIQUEMENT pour le RL et le MILP :

        credit = (soc_end − soc_init) · capacity · eta_discharge · store_value
        net_cost_scoré ← net_cost_scoré − credit

    ``eta_discharge`` convertit le SoC stocké en énergie réellement livrable (1 kWh de SoC
    rend ``eta_discharge`` kWh) — NÉCESSAIRE ici (≠ ``MicrogridEnv._export_opportunity_cost``
    où le rendement se simplifie entre deux actions futures). ``store_value`` = valeur
    marginale du stock (€/kWh ; sous prix fixes = prix d'import, cf. ``_store_value``).

    NE PAS appliquer au vrai terminal du rollout : l'énergie restante n'a aucun futur ⇒
    c'est une vraie perte (la sous-décharge terminale doit rester pénalisée).

    Returns:
        Le crédit (€) à SOUSTRAIRE du coût net scoré. >0 si l'agent porte de l'inventaire au
        bord (soc_end > soc_init), <0 s'il l'a bradé. ``soc_init`` (commun RL/MILP) s'annule
        dans le gap ; il garde seulement ``net_cost`` physiquement interprétable.
    """
    return (soc_end_frac - soc_init_frac) * capacity_kwh * eta_discharge * store_value


def metrics_by_season(
    history: dict,
    labels: np.ndarray,
    delta_t_h: float,
    soc_min: float,
    soc_max: float,
    price_import,
    price_export,
    phantom_penalty: float = 1e3,
) -> dict:
    """Découpe le rollout par saison et calcule les métriques sur chaque tranche.

    Réutilise ``compute_metrics`` sur des sous-history masquées par saison, pour
    que les scalaires par saison soient cohérents bit-à-bit avec le calcul global.

    Args:
        history: dict des tableaux par pas (cf. compute_metrics).
        labels: np.ndarray[str] de longueur T (sortie de ``season_labels``).
        delta_t_h, soc_min, soc_max, phantom_penalty: cf. compute_metrics.
        price_import, price_export: scalaire OU array de longueur >= T.

    Returns:
        {saison: {scalaires compute_metrics}} pour les saisons réellement présentes.
    """
    n = len(labels)

    def _slice(v, mask):
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] >= n:
            return arr[:n][mask]
        return v  # scalaire / longueur non alignée → inchangé

    out = {}
    for season in ("hiver", "ete"):
        mask = labels == season
        if not mask.any():
            continue
        sub_hist = {k: _slice(v, mask) for k, v in history.items()}
        out[season] = compute_metrics(
            sub_hist, delta_t_h, soc_min, soc_max,
            _slice(price_import, mask), _slice(price_export, mask),
            phantom_penalty,
        )
        out[season]["n_steps"] = int(mask.sum())
    return out
