"""Compare RL and MILP results."""

import json
import os


def compare_results(rl_metrics: dict, milp_metrics: dict, output_dir: str = None) -> dict:
    """Compute relative gap and print a formatted comparison.

    Returns:
        dict with comparison metrics.
    """
    rl_net = rl_metrics["net_cost"]
    milp_net = milp_metrics["net_cost"]

    if abs(milp_net) > 1e-9:
        relative_gap = (rl_net - milp_net) / abs(milp_net)
    else:
        relative_gap = float("inf") if rl_net != 0 else 0.0

    # Phantom-aware (unserved-load) metrics. net_cost alone is NOT comparable
    # across policies that leave different amounts of load unserved — a policy
    # that serves less load imports less and looks cheaper. net_cost_adjusted
    # charges the unserved load at the phantom penalty (VOLL) to restore it.
    rl_adj = rl_metrics.get("net_cost_adjusted", rl_net)
    milp_adj = milp_metrics.get("net_cost_adjusted", milp_net)
    rl_phantom = rl_metrics.get("phantom_energy_kwh", 0.0)
    milp_phantom = milp_metrics.get("phantom_energy_kwh", 0.0)
    rl_served = rl_metrics.get("served_load_ratio", 1.0)
    milp_served = milp_metrics.get("served_load_ratio", 1.0)

    if abs(milp_adj) > 1e-9:
        relative_gap_adjusted = (rl_adj - milp_adj) / abs(milp_adj)
    else:
        relative_gap_adjusted = float("inf") if rl_adj != 0 else 0.0

    comparison = {
        "rl_net_cost": rl_net,
        "milp_net_cost": milp_net,
        "relative_gap": relative_gap,
        "rl_net_cost_adjusted": rl_adj,
        "milp_net_cost_adjusted": milp_adj,
        "relative_gap_adjusted": relative_gap_adjusted,
        "rl_phantom_energy_kwh": rl_phantom,
        "milp_phantom_energy_kwh": milp_phantom,
        "rl_served_load_ratio": rl_served,
        "milp_served_load_ratio": milp_served,
        "rl_total_cost": rl_metrics["total_cost"],
        "milp_total_cost": milp_metrics["total_cost"],
        "rl_total_revenue": rl_metrics["total_revenue"],
        "milp_total_revenue": milp_metrics["total_revenue"],
        "rl_soc_violations": rl_metrics["soc_violations"],
        "milp_soc_violations": milp_metrics["soc_violations"],
        "rl_self_consumption": rl_metrics["self_consumption_rate"],
        "milp_self_consumption": milp_metrics["self_consumption_rate"],
        "rl_peak_import": rl_metrics["peak_grid_import"],
        "milp_peak_import": milp_metrics["peak_grid_import"],
    }

    header = f"{'Metric':<25} {'RL (SAC)':>12} {'MILP':>12} {'Gap':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    rows = [
        ("Net cost (EUR)", rl_net, milp_net, f"{relative_gap:+.1%}"),
        ("Net cost adj. (EUR)", rl_adj, milp_adj, f"{relative_gap_adjusted:+.1%}"),
        ("Phantom energy (kWh)", rl_phantom, milp_phantom, ""),
        ("Served load ratio", rl_served, milp_served, ""),
        ("Total cost (EUR)", rl_metrics["total_cost"], milp_metrics["total_cost"], ""),
        ("Total revenue (EUR)", rl_metrics["total_revenue"], milp_metrics["total_revenue"], ""),
        ("SoC violations", rl_metrics["soc_violations"], milp_metrics["soc_violations"], ""),
        ("Self-consumption", rl_metrics["self_consumption_rate"], milp_metrics["self_consumption_rate"], ""),
        ("Peak import (kW)", rl_metrics["peak_grid_import"], milp_metrics["peak_grid_import"], ""),
    ]

    for label, rl_val, milp_val, gap_str in rows:
        print(f"{label:<25} {rl_val:>12.4f} {milp_val:>12.4f} {gap_str:>10}")
    print(sep)

    if max(rl_phantom, milp_phantom) > 1e-6:
        print(
            "⚠ Unserved load present (phantom > 0): raw net_cost is NOT comparable "
            "— a policy serving less load imports less and looks cheaper.\n"
            "  Use 'Net cost adj.' (charges unserved load at the phantom penalty) "
            "and 'Served load ratio' for an apples-to-apples comparison."
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "comparison.json")
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to {path}")

    return comparison
