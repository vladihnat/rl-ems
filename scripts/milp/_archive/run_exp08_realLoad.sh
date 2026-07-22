#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp08_realLoad.yaml"
OUT="monitoring/runs/milp_exp08_realLoad_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp08_realLoad_plan.csv"

cd "$ROOT"

# Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --no-show \
    "$@"
