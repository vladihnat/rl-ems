"""MILP baseline: optimal dispatch with perfect foresight using CVXPY + HiGHS."""

import numpy as np
import cvxpy as cp

from evaluation.metrics import compute_metrics


def run_milp(env, config: dict) -> dict:
    """Solve the full-horizon optimal dispatch problem.

    Decision variable: Pb[t] (battery power at each timestep).
    Objective: minimize total import cost minus export revenue.
    """
    cfg = config
    delta_t_h = cfg["time"]["delta_t_min"] / 60.0
    horizon_steps = int(cfg["time"]["horizon_h"] * 60 / cfg["time"]["delta_t_min"])
    T = env.pv.n_steps - horizon_steps

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

    price_imp = cfg["grid"]["price_import"]
    price_exp = cfg["grid"]["price_export"]
    max_imp = cfg["grid"]["max_import_kw"]
    max_exp = cfg["grid"]["max_export_kw"]

    Pb = cp.Variable(T)
    Pg = cp.Variable(T)
    P_imp = cp.Variable(T, nonneg=True)
    P_exp = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    constraints = []

    constraints.append(soc[0] == init_soc)

    for t in range(T):
        constraints.append(Pg[t] == load_vals[t] - pv_vals[t] - Pb[t])
        constraints.append(Pg[t] == P_imp[t] - P_exp[t])

    constraints.append(Pb >= -max_charge)
    constraints.append(Pb <= max_discharge)
    constraints.append(P_imp <= max_imp)
    constraints.append(P_exp <= max_exp)

    # SoC dynamics: linearised (charge/discharge handled via single efficiency approximation)
    # For the MILP we use a simplified model:
    #   charging (Pb<0):  soc[t+1] = soc[t] - Pb[t] * eta_c * dt / capacity
    #   discharging (Pb>0): soc[t+1] = soc[t] - Pb[t] / eta_d * dt / capacity
    # Since this makes the problem nonlinear, we introduce auxiliary variables.
    Pb_charge = cp.Variable(T, nonneg=True)  # magnitude of charging
    Pb_discharge = cp.Variable(T, nonneg=True)  # magnitude of discharging

    constraints.append(Pb == Pb_discharge - Pb_charge)

    for t in range(T):
        soc_change = (Pb_charge[t] * eta_c - Pb_discharge[t] / eta_d) * delta_t_h / capacity
        constraints.append(soc[t + 1] == soc[t] + soc_change)

    constraints.append(soc >= soc_min)
    constraints.append(soc <= soc_max)
    constraints.append(Pb_charge <= max_charge)
    constraints.append(Pb_discharge <= max_discharge)

    objective = cp.Minimize(
        cp.sum(price_imp * P_imp - price_exp * P_exp) * delta_t_h
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MILP solver failed: {prob.status}")

    Pb_sol = Pb.value
    Pg_sol = Pg.value
    soc_sol = soc.value

    history = {
        "P_grid": Pg_sol,
        "Pb_effective": Pb_sol,
        "soc": soc_sol[1:],
        "pv_t": pv_vals,
        "load_t": load_vals,
        "r_eco": -(price_imp * np.maximum(Pg_sol, 0) - price_exp * np.maximum(-Pg_sol, 0)) * delta_t_h,
        "r_soc": np.zeros(T),
        "reward": -(price_imp * np.maximum(Pg_sol, 0) - price_exp * np.maximum(-Pg_sol, 0)) * delta_t_h,
    }

    metrics = compute_metrics(history, delta_t_h, soc_min, soc_max, price_imp, price_exp)
    metrics["history"] = history
    metrics["solver_status"] = prob.status
    metrics["objective_value"] = prob.value
    return metrics
