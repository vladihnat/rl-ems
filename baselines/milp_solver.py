"""MILP baseline: optimal dispatch with perfect foresight using CVXPY + HiGHS."""

import numpy as np
import cvxpy as cp

from evaluation.metrics import compute_metrics, metrics_by_season, season_labels


def _build_problem(env, config: dict, fix_b=None, fix_g=None) -> dict:
    """Construit le problème de dispatch plein-horizon (variables + contraintes + objectif).

    Factorisation partagée entre ``run_milp`` (solve MIP) et ``milp_soc_duals`` (re-solve LP à
    binaires figés pour extraire les duals) : UNE seule définition du modèle, pas de dérive.

    Args:
        fix_b, fix_g: arrays 0/1 de longueur T. Si fournis, les binaires (b = décharge/charge,
            g_grid = import/export) deviennent des CONSTANTES ⇒ le problème est un pur LP dont
            les duals sont exploitables. None (défaut) = vraies variables booléennes (MIP).

    Returns:
        dict avec ``prob`` et toutes les variables, plus ``con_soc_init`` / ``con_soc_dyn``
        (contraintes de dynamique SoC, dans l'ordre t=0..T-1) pour l'extraction des duals.
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
    g_grid = cp.Constant(np.asarray(fix_g, dtype=np.float64)) if fix_g is not None \
        else cp.Variable(T, boolean=True)
    Pcurt = cp.Variable(T, nonneg=True)  # curtailment PV explicite
    # Puissance fantôme (slack à la Duchaud-JL, cf. @EmsLinprog Pph) : production
    # « sortie du néant » ≥ 0, sans borne supérieure, fortement pénalisée. Permet de
    # boucler le bilan quand l'import réseau + la batterie ne peuvent pas servir la charge.
    P_phantom = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    constraints = []

    con_soc_init = soc[0] == init_soc
    constraints.append(con_soc_init)

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
    constraints.append(P_imp <= max_imp * g_grid)
    constraints.append(P_exp <= max_exp * (1 - g_grid))

    # SoC dynamics: linearised (charge/discharge handled via single efficiency approximation)
    # For the MILP we use a simplified model:
    #   charging (Pb<0):  soc[t+1] = soc[t] - Pb[t] * eta_c * dt / capacity
    #   discharging (Pb>0): soc[t+1] = soc[t] - Pb[t] / eta_d * dt / capacity
    # Since this makes the problem nonlinear, we introduce auxiliary variables.
    Pb_charge = cp.Variable(T, nonneg=True)  # magnitude of charging
    Pb_discharge = cp.Variable(T, nonneg=True)  # magnitude of discharging
    b = cp.Constant(np.asarray(fix_b, dtype=np.float64)) if fix_b is not None \
        else cp.Variable(T, boolean=True)  # b[t]=1 → discharge, b[t]=0 → charge

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

    con_soc_dyn = []
    for t in range(T):
        soc_change = (Pb_charge[t] * eta_c - Pb_discharge[t] / eta_d) * delta_t_h / capacity
        con_soc_dyn.append(soc[t + 1] == soc[t] + soc_change)
    constraints.extend(con_soc_dyn)

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

    return {
        "prob": prob, "T": T, "delta_t_h": delta_t_h,
        "pv_vals": pv_vals, "load_vals": load_vals,
        "price_imp": price_imp, "price_exp": price_exp,
        "phantom_penalty": phantom_penalty, "tie_eps": tie_eps, "milp_cfg": milp_cfg,
        "Pb": Pb, "Pg": Pg, "P_imp": P_imp, "P_exp": P_exp, "g_grid": g_grid,
        "Pcurt": Pcurt, "P_phantom": P_phantom, "soc": soc,
        "Pb_charge": Pb_charge, "Pb_discharge": Pb_discharge, "b": b,
        "con_soc_init": con_soc_init, "con_soc_dyn": con_soc_dyn,
        "soc_min": soc_min, "soc_max": soc_max, "capacity": capacity,
    }


def _solve(built: dict) -> None:
    """Résout le problème construit par ``_build_problem`` (options HiGHS partagées).

    Déterminisme du solveur : HiGHS multithread renvoie un sommet arbitraire de la face
    optimale dégénérée à chaque solve → le tracé MILP « clignote » d'un run à l'autre, même
    avec le tie-break (la dégénérescence sur les binaires b/g_grid n'est pas figée par un ε
    sur les flux continus). parallel="off" + random_seed fixe ⇒ HiGHS est une fonction
    déterministe de l'entrée (même résultat à chaque run). Quand le tie-break est actif, on
    ferme le gap MIP à 0 pour que l'optimum unique du tilt soit réellement atteint.
    """
    prob = built["prob"]
    solver_options = {"parallel": "off", "random_seed": 0}
    if built["tie_eps"] > 0.0:
        solver_options.update(mip_rel_gap=0.0, mip_abs_gap=0.0)
    solver_options.update(built["milp_cfg"].get("solver_options", {}))  # override explicite gagne
    prob.solve(solver=cp.HIGHS, verbose=False, **solver_options)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MILP solver failed: {prob.status}")


def run_milp(env, config: dict) -> dict:
    """Solve the full-horizon optimal dispatch problem.

    Decision variable: Pb[t] (battery power at each timestep).
    Objective: minimize total import cost minus export revenue.
    """
    built = _build_problem(env, config)
    _solve(built)

    T = built["T"]
    delta_t_h = built["delta_t_h"]
    pv_vals, load_vals = built["pv_vals"], built["load_vals"]
    price_imp, price_exp = built["price_imp"], built["price_exp"]
    phantom_penalty = built["phantom_penalty"]
    soc_min, soc_max = built["soc_min"], built["soc_max"]
    prob = built["prob"]
    Pb, Pg, soc = built["Pb"], built["Pg"], built["soc"]
    Pb_charge, Pb_discharge, b = built["Pb_charge"], built["Pb_discharge"], built["b"]
    Pcurt, P_phantom = built["Pcurt"], built["P_phantom"]

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


def milp_soc_duals(env, config: dict) -> np.ndarray:
    """Valeur marginale exacte du stock λ(t) en €/kWh, longueur T+1 (t = 0..T).

    Pour le potentiel PBRS ``reward.store_value_mode: "milp_dual"`` (exp26) : remplace
    l'heuristique ``v(t) = max forecast`` (qui survalorise le stock — elle ignore la limite de
    puissance et la saturation, et ne décroît pas après le pic) par le prix fantôme du kWh
    stocké issu du problème d'optimisation lui-même. PBRS reste invariant quel que soit Φ ⇒
    n'altère ni l'optimum ni la comparaison RL↔MILP ; λ(t) ne fait qu'accélérer le credit
    assignment (le pic du soir est à ~84 pas de t0, cf. audit exp23).

    Méthode (2 solves) :
      1. solve MIP plein-horizon (même modèle que ``run_milp`` via ``_build_problem``) ;
      2. re-solve en LP avec les binaires b / g_grid FIGÉS à l'optimum ⇒ duals exploitables ;
         λ(t) = dual de la contrainte qui « produit » soc[t] : init pour t=0, dynamique t−1
         pour t ≥ 1. Duals en €/(fraction de SoC) → ÷ capacité pour des €/kWh.

    Signe : CVXPY (minimisation) donne un dual ≥ 0 quand ajouter du stock RÉDUIT le coût.
    Les λ peuvent être légèrement négatifs aux pas où la batterie sature (soc_max atteint :
    du stock en plus forcerait du curtailment) — on les garde tels quels, c'est la vraie
    valeur marginale locale.
    """
    built = _build_problem(env, config)
    _solve(built)

    b_opt = np.round(np.asarray(built["b"].value, dtype=np.float64))
    g_opt = np.round(np.asarray(built["g_grid"].value, dtype=np.float64))

    lp = _build_problem(env, config, fix_b=b_opt, fix_g=g_opt)
    _solve(lp)

    # Garde-fou : figer les binaires à leur optimum ne doit pas changer l'objectif
    # (sinon l'arrondi a cassé la faisabilité et les duals ne valent rien).
    if not np.isclose(lp["prob"].value, built["prob"].value, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            f"milp_soc_duals: objectif LP ({lp['prob'].value:.6f}) ≠ MIP "
            f"({built['prob'].value:.6f}) après fixation des binaires."
        )

    duals = np.empty(lp["T"] + 1, dtype=np.float64)
    duals[0] = float(np.asarray(lp["con_soc_init"].dual_value))
    for t, con in enumerate(lp["con_soc_dyn"]):
        duals[t + 1] = float(np.asarray(con.dual_value))

    # Signe vérifié empiriquement (exp23 hiver, CVXPY 1.x/HiGHS) : le dual d'égalité est
    # DIRECTEMENT la valeur marginale du stock (≥ 0), sous forme de co-état constant par
    # morceaux qui saute quand le SoC touche une borne — p.ex. λ(fin) = coût de remplacement
    # 0.216/η_c = 0.24 €/kWh. Garde-fou : si une future version de CVXPY inverse la
    # convention du dual d'égalité, on réaligne sur « valeur du stock ≥ 0 en médiane ».
    lam = duals / float(lp["capacity"])
    if np.median(lam) < 0.0:
        lam = -lam
    return lam
