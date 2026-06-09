#!/usr/bin/env python3
"""Agrège les metrics.json d'un/plusieurs sweeps → gap moyen ± σ (best & final model).

Lit ``results/<sweep>/run_*/metrics.json`` (produits par experiments/run_experiment.py)
et résume, par sweep, le gap du modèle sélectionné sur validation (best) et du modèle
final, avec moyenne ± écart-type sur les seeds.

⚠ Le ``relative_gap`` BRUT n'est pas comparable entre politiques qui laissent des quantités
différentes de charge NON SERVIE (phantom) : une politique qui sert moins de charge importe
moins et paraît moins chère (cf. evaluation/compare.py). On affiche donc aussi :
    - ``gap_adj``  = ``comparison.relative_gap_adjusted`` (facture la charge non servie au VOLL),
    - ``served``   = ``rl.served_load_ratio`` (1.0 = toute la charge servie).
Toute run ``served < 1`` (ou phantom > 0) est flaggée ``!`` et EXCLUE de la moyenne ajustée :
c'est un faux « bon gap ». ``gap_adj`` / ``served`` portent sur le modèle BEST (le résultat
reporté) ; ``gap_final`` reste brut (l'ajusté du modèle final n'est pas persisté).

Usage:
    python cluster/sweep/aggregate_results.py exp10_seeds exp11b_seeds
"""

import glob
import json
import os
import sys
from statistics import mean, stdev

import yaml


def _seed_of(run_dir: str):
    cfg = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg):
        return "?"
    with open(cfg) as f:
        return yaml.safe_load(f).get("experiment", {}).get("seed", "?")


def collect(sweep: str):
    rows = []
    for mp in sorted(glob.glob(os.path.join("results", sweep, "run_*", "metrics.json"))):
        with open(mp) as f:
            d = json.load(f)
        sel, comp, rl = d.get("selection", {}), d.get("comparison", {}), d.get("rl", {})
        rows.append({
            "run": os.path.basename(os.path.dirname(mp)),
            "seed": _seed_of(os.path.dirname(mp)),
            "gap_best": sel.get("gap_best") if sel.get("gap_best") is not None else comp.get("relative_gap"),
            "gap_best_adj": comp.get("relative_gap_adjusted"),
            "served": rl.get("served_load_ratio"),
            "phantom": rl.get("phantom_energy_kwh"),
            "gap_final": sel.get("gap_final"),
            "sc": rl.get("self_consumption_rate"),
            "import_kwh": rl.get("energy_imported_kwh"),
        })
    return rows


def _is_phantom(r) -> bool:
    """True si le modèle best laisse de la charge non servie → son gap brut est trompeur."""
    if r["served"] is not None and r["served"] < 1.0 - 1e-6:
        return True
    if r["phantom"] is not None and r["phantom"] > 1e-6:
        return True
    return False


def _fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "   n/a"
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:+.1%} ± {s:.1%}  (n={len(xs)})"


def main():
    sweeps = sys.argv[1:]
    if not sweeps:
        sys.exit("usage: aggregate_results.py <sweep_name> [<sweep_name> ...]")

    for sweep in sweeps:
        rows = collect(sweep)
        print(f"\n=== {sweep} ===")
        if not rows:
            print("  (aucun metrics.json — runs pas encore terminés ?)")
            continue
        print(f"  {'run':<9} {'seed':>5} {'gap_best':>9} {'gap_adj':>9} "
              f"{'served':>8} {'gap_final':>10} {'SC':>7} {'import_kWh':>11}")
        for r in rows:
            flag = "!" if _is_phantom(r) else " "
            gb = f"{r['gap_best']:+.1%}" if r["gap_best"] is not None else "n/a"
            ga = f"{r['gap_best_adj']:+.1%}" if r["gap_best_adj"] is not None else "n/a"
            sv = f"{r['served']:.3%}" if r["served"] is not None else "n/a"
            gf = f"{r['gap_final']:+.1%}" if r["gap_final"] is not None else "n/a"
            sc = f"{r['sc']:.1%}" if r["sc"] is not None else "n/a"
            im = f"{r['import_kwh']:.0f}" if r["import_kwh"] is not None else "n/a"
            print(f" {flag}{r['run']:<9} {str(r['seed']):>5} {gb:>9} {ga:>9} "
                  f"{sv:>8} {gf:>10} {sc:>7} {im:>11}")

        feasible = [r for r in rows if not _is_phantom(r)]
        n_flag = len(rows) - len(feasible)
        print(f"  -> gap_best (brut, toutes)     : {_fmt([r['gap_best'] for r in rows])}")
        print(f"  -> gap_final (brut, toutes)    : {_fmt([r['gap_final'] for r in rows])}")
        suffix = f"   [{n_flag} run(s) '!' phantom exclue(s)]" if n_flag else ""
        print(f"  -> gap_adj (best, faisables)   : {_fmt([r['gap_best_adj'] for r in feasible])}{suffix}")


if __name__ == "__main__":
    main()
