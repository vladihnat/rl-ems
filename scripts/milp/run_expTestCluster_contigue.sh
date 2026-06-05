#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/expTestCluster.yaml"
PV_SIMU="data/irradiance_simulation.csv"
LOAD_SIMU="data/load_simulation.csv"
OUT="monitoring/runs/milp_expTestCluster_contigue_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_expTestCluster_contigue_plan.csv"

cd "$ROOT"

# Résout le MILP sur une fenêtre de simulation CONTIGUË (plots lisibles, SoC physiquement continue).
# Générer les CSV simu au besoin :
#   micromamba run -n stageCorse python data/extract_pyrano_simu.py --nbD 7 --month 1 --startDate 1 --usage simu
# Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV_SIMU" \
    --measures "$PV_SIMU" \
    --load-csv "$LOAD_SIMU" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --no-show \
    "$@"
