#!/usr/bin/env python3
"""Re-score les runs d'un sweep SANS réentraîner.

Le scoring est un POST-TRAITEMENT (cf. crédit de SoC de bord, evaluation.metrics.
boundary_soc_credit) : les ``metrics.json``/``comparison.json`` écrits PENDANT
l'entraînement sont figés à l'ancien scoring. Ce driver relance, pour chaque run,
``experiments/run_experiment.py --config <run>/config_used.yaml --rescore`` qui recharge le
modèle déjà sauvegardé et régénère metrics.json/comparison.json avec le scoring courant.
Les .zip/.pkl/.npz (donc la sélection best-model) restent intacts.

Après ça, ``python cluster/sweep/aggregate_results.py <sweep>`` montre les gaps corrigés.

Usage:
    python cluster/sweep/rescore_runs.py exp22_hiver_haute exp22_ete_haute
"""

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gap(run_dir: str):
    p = os.path.join(run_dir, "comparison.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get("relative_gap")


def rescore_sweep(sweep: str) -> int:
    cfgs = sorted(glob.glob(os.path.join(ROOT, "results", sweep, "run_*", "config_used.yaml")))
    if not cfgs:
        print(f"  (aucun run dans results/{sweep} — rien à re-scorer)")
        return 0
    n_fail = 0
    for cfg in cfgs:
        run_dir = os.path.dirname(cfg)
        run = os.path.basename(run_dir)
        # Run incomplet (pas de modèle final) → on saute proprement.
        if not glob.glob(os.path.join(run_dir, "*_model.zip")):
            print(f"  {run}: SKIP (pas de modèle sauvegardé)")
            continue
        before = _gap(run_dir)
        proc = subprocess.run(
            [sys.executable, "experiments/run_experiment.py", "--config", cfg, "--rescore"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            n_fail += 1
            print(f"  {run}: ÉCHEC (rc={proc.returncode})")
            print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-6:]))
            continue
        after = _gap(run_dir)
        fb = f"{before:+.1%}" if before is not None else "n/a"
        fa = f"{after:+.1%}" if after is not None else "n/a"
        print(f"  {run}: gap {fb} -> {fa}")
    return n_fail


def main():
    sweeps = sys.argv[1:]
    if not sweeps:
        sys.exit("usage: rescore_runs.py <sweep_name> [<sweep_name> ...]")
    total_fail = 0
    for sweep in sweeps:
        print(f"\n=== re-score {sweep} ===")
        total_fail += rescore_sweep(sweep)
    if total_fail:
        print(f"\n⚠ {total_fail} run(s) en échec.")
        sys.exit(1)
    print("\nOK — relancez : python cluster/sweep/aggregate_results.py " + " ".join(sweeps))


if __name__ == "__main__":
    main()
