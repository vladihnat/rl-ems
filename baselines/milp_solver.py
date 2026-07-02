"""MILP baseline: optimal dispatch with perfect foresight using CVXPY + HiGHS."""

import numpy as np
import cvxpy as cp

from evaluation.metrics import compute_metrics, metrics_by_season, season_labels


def run_milp(env, config: dict) -> dict:
    """Solve the full-horizon optimal dispatch problem.

    Decision variable: Pb[t] (battery power at each timestep).
    Objective: minimize total import cost minus export revenue.
    """
    cfg = config
    delta_t_h = cfg["time"]["delta_t_min"] / 60.0
    T = int(env.max_steps)

    pv_vals = np.array([env.pv.get_irradiance(t) for t in range(T)])
    load_vals = np.array([env.load.get_load(t) for t in range(T)])

    capacity = cfg["battery"]["capacity_kwh"]
    soc_min = cfg["battery"]["soc_min"]
    soc_max = cfg["battery"]["soc_max"]
    max_charge = cfg["battery"]["max_charge_kw"]
    max_discharge = cfg["battery"]["max_discharge_kw"]
    eta_c = cfg["battery"]["efficiency_charge"]
    eta_d = cfg["battery"]["efficiency_discharge"]
    init_soc = cfg["battery"]["init_soc"]

    price_imp = env.price_signal.import_prices[:T]
    price_exp = env.price_signal.export_prices[:T]
    max_imp = cfg["grid"]["max_import_kw"]
    max_exp = cfg["grid"]["max_export_kw"]

    Pb = cp.Variable(T)
    Pg = cp.Variable(T)
    P_imp = cp.Variable(T, nonneg=True)
    P_exp = cp.Variable(T, nonneg=True)
    # Complémentarité import/export : binaire interdisant P_imp et P_exp d'être tous deux > 0
    # au même pas (1 = import, 0 = export). SANS elle, dès que le prix d'export dépasse le prix
    # d'import (pic CoutsProd haute saison soir : export ~0.35-0.44 vs import HP 0.2475 €/kWh),
    # le solveur « pompe » un spread fictif en achetant ET vendant à la borne réseau au même
    # instant : objectif non physique (énergie créée du néant) et dispatch batterie corrompu,
    # rendant la baseline MILP battable par le RL. cf. tests/test_milp_no_money_pump.py.
    g_grid = cp.Variable(T, boolean=True)
    Pcurt = cp.Variable(T, nonneg=True)  # curtailment PV explicite
    # Puissance fantôme (slack à la Duchaud-JL, cf. @EmsLinprog Pph) : production
    # « sortie du néant » ≥ 0, sans borne supérieure, fortement pénalisée. Permet de
    # boucler le bilan quand l'import réseau + la batterie ne peuvent pas servir la charge.
    P_phantom = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    constraints = []

    constraints.append(soc[0] == init_soc)

    for t in range(T):
        # bilan corrigé : Pp_used = pv - Pcurt ; la puissance fantôme s'ajoute à l'offre
        constraints.append(Pg[t] == load_vals[t] - (pv_vals[t] - Pcurt[t]) - Pb[t] - P_phantom[t])
        constraints.append(Pg[t] == P_imp[t] - P_exp[t])
        constraints.append(Pcurt[t] <= pv_vals[t])

    constraints.append(Pb >= -max_charge)
    constraints.append(Pb <= max_discharge)
    # Big-M (= bornes réseau) liées au binaire g_grid : g=1 ⇒ P_exp=0 (import ≤ max_imp) ;
    # g=0 ⇒ P_imp=0 (export ≤ max_exp). Remplace les bornes simples pour tuer le money-pump.
    # Non-contraignant quand price_exp ≤ price_imp (toute basse saison / heures creuses) :
    # l'optimum LP y a déjà P_imp·P_exp=0 ⇒ résultats inchangés sur ces régimes.
    #Decomente ligne 46 pour que cela marche et commente lignes 75 et 76
    constraints.append(P_imp <= max_imp * g_grid)
    constraints.append(P_exp <= max_exp * (1 - g_grid))

    # Contraintes ancienne sans complementarité 
    # constraints.append(P_imp <= max_imp)
    # constraints.append(P_exp <= max_exp )

    # SoC dynamics: linearised (charge/discharge handled via single efficiency approximation)
    # For the MILP we use a simplified model:
    #   charging (Pb<0):  soc[t+1] = soc[t] - Pb[t] * eta_c * dt / capacity
    #   discharging (Pb>0): soc[t+1] = soc[t] - Pb[t] / eta_d * dt / capacity
    # Since this makes the problem nonlinear, we introduce auxiliary variables.
    Pb_charge = cp.Variable(T, nonneg=True)  # magnitude of charging
    Pb_discharge = cp.Variable(T, nonneg=True)  # magnitude of discharging
    b = cp.Variable(T, boolean=True)  # b[t]=1 → discharge, b[t]=0 → charge

    constraints.append(Pb == Pb_discharge - Pb_charge)

    # Contrainte EDF : interdiction de charger la batterie DEPUIS LE RÉSEAU. Deux formulations,
    # pilotées par battery.pv_charge_mode (cf. temp_md/milpIncoherences.md #5). Les deux bornent
    # Pb_charge par des DONNÉES pv/load ⇒ borne constante, donc linéaire, et interdisent tout
    # import réseau d'alimenter la batterie (charge ≤ PV produit) :
    #   - "total" (DÉFAUT, relâché, fidèle à Duchaud-JL @EmsLinprog split-node : Ppv_b ≤ PV,
    #     Pg_l ≤ load) : la batterie peut absorber TOUT le PV (Pb_charge ≤ pv) pendant que le
    #     réseau sert la charge indépendamment. Autorise l'arbitrage été (charge PV le jour →
    #     export du soir). Régime cible de toutes les nouvelles expériences.
    #   - "surplus" (STRICT, historique) : la batterie n'absorbe que le surplus PV au-delà de la
    #     charge instantanée (Pb_charge ≤ max(0, pv−load)). Bloque toute « libération » de PV vers
    #     la batterie pendant que le réseau sert la charge. Les configs antérieures l'épinglent.
    pv_charge_mode = cfg["battery"].get("pv_charge_mode", "total")
    if pv_charge_mode == "total":
        constraints.append(Pb_charge <= pv_vals)
    elif pv_charge_mode == "surplus":
        pv_surplus = np.maximum(0.0, pv_vals - load_vals)
        constraints.append(Pb_charge <= pv_surplus)
    else:
        raise ValueError(
            f"Unknown battery.pv_charge_mode={pv_charge_mode!r}; use 'surplus' or 'total'."
        )

    for t in range(T):
        soc_change = (Pb_charge[t] * eta_c - Pb_discharge[t] / eta_d) * delta_t_h / capacity
        constraints.append(soc[t + 1] == soc[t] + soc_change)

    constraints.append(soc >= soc_min)
    constraints.append(soc <= soc_max)
    # Mutual exclusion: prevent simultaneous charge and discharge (Duchaud-JL strategy)
    constraints.append(Pb_discharge <= max_discharge * b)
    constraints.append(Pb_charge    <= max_charge    * (1 - b))

    # Pénalité fantôme ≫ tout prix d'import ⇒ le solveur épuise d'abord le réseau réel
    # (P_imp = max_imp) avant de recourir à la puissance fantôme. Défaut 1e3 €/kWh (Duchaud-JL).
    phantom_penalty = cfg["grid"].get("phantom_penalty", 1e3)

    # Tie-break lexicographique (opt-in, défaut OFF). Sous prix plats (exp22), l'optimum
    # est dégénéré : import/export figés, timing libre ⇒ toute une face de trajectoires
    # au même coût, et HiGHS en renvoie un sommet arbitraire ⇒ le tracé MILP « clignote »
    # d'un solve à l'autre. Un ε·tilt strictement croissant sur les flux flexibles dans le
    # temps préfère « agir tôt » et rend l'optimum unique → tracé déterministe. ε ≪ coût
    # économique ⇒ ne change pas le coût rapporté (recalculé hors de ce terme par replay).
    milp_cfg = cfg.get("milp", {})
    tie_eps = float(milp_cfg.get("tie_break_eps", 0.0))
    tie_break = 0
    if tie_eps > 0.0:
        tilt = np.linspace(0.0, 1.0, T)
        tie_break = tie_eps * cp.sum(
            cp.multiply(tilt, Pb_discharge + Pb_charge + P_imp + P_exp)
        )

    objective = cp.Minimize(
        (cp.sum(cp.multiply(price_imp, P_imp) - cp.multiply(price_exp, P_exp))
         + phantom_penalty * cp.sum(P_phantom)) * delta_t_h
        + tie_break
    )

    prob = cp.Problem(objective, constraints)
    # Déterminisme du solveur : HiGHS multithread renvoie un sommet arbitraire de la face
    # optimale dégénérée à chaque solve → le tracé MILP « clignote » d'un run à l'autre, même
    # avec le tie-break (la dégénérescence sur les binaires b/g_grid n'est pas figée par un ε
    # sur les flux continus). parallel="off" + random_seed fixe ⇒ HiGHS est une fonction
    # déterministe de l'entrée (même résultat à chaque run). Quand le tie-break est actif, on
    # ferme le gap MIP à 0 pour que l'optimum unique du tilt soit réellement atteint.
    solver_options = {"parallel": "off", "random_seed": 0}
    if tie_eps > 0.0:
        solver_options.update(mip_rel_gap=0.0, mip_abs_gap=0.0)
    solver_options.update(milp_cfg.get("solver_options", {}))  # override explicite gagne
    prob.solve(solver=cp.HIGHS, verbose=False, **solver_options)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MILP solver failed: {prob.status}")

    Pb_sol = Pb.value
    Pg_sol = Pg.value
    soc_sol = soc.value

    P_imp_sol = np.maximum(Pg_sol, 0.0)
    P_exp_sol = np.maximum(-Pg_sol, 0.0)
    r_eco_sol = -(price_imp * P_imp_sol - price_exp * P_exp_sol) * delta_t_h

    # Include the phantom penalty in the reward so total_reward is comparable with
    # the RL side (which already pays r_phantom). net_cost stays grid-only.
    P_phantom_sol = np.asarray(P_phantom.value, dtype=np.float64)
    r_phantom_sol = -phantom_penalty * P_phantom_sol * delta_t_h

    history = {
        "P_grid": Pg_sol,
        "Pb_effective": Pb_sol,
        "soc": soc_sol[1:],
        "pv_t": pv_vals,
        "load_t": load_vals,
        "r_eco": r_eco_sol,
        "r_soc": np.zeros(T),
        "reward": r_eco_sol + r_phantom_sol,
        "Pb_charge": Pb_charge.value,
        "Pb_discharge": Pb_discharge.value,
        "b_int": np.asarray(b.value, dtype=np.float64),
        "P_imp": P_imp_sol,
        "P_exp": P_exp_sol,
        "price_imp": np.asarray(price_imp, dtype=np.float64),
        "price_exp": np.asarray(price_exp, dtype=np.float64),
        "Pcurt": np.asarray(Pcurt.value, dtype=np.float64),
        "P_phantom": P_phantom_sol,
    }

    # compute_metrics derives phantom_energy_kwh / phantom_steps / net_cost_adjusted
    # from history["P_phantom"] (shared with the RL path for an apples-to-apples
    # comparison). net_cost stays the pure grid economics recomputed from Pg_sol.
    metrics = compute_metrics(history, delta_t_h, soc_min, soc_max,
                              price_imp, price_exp, phantom_penalty)
    labels = season_labels(env.pv.timestamps[:T])
    metrics["by_season"] = metrics_by_season(history, labels, delta_t_h, soc_min,
                                             soc_max, price_imp, price_exp,
                                             phantom_penalty)
    metrics["history"] = history
    metrics["solver_status"] = prob.status
    metrics["objective_value"] = prob.value
    return metrics
