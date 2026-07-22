#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/expTestCluster.yaml"
OUT="monitoring/runs/milp_expTestCluster_exact_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_expTestCluster_exact_plan.csv"

cd "$ROOT"

# Résout le MILP sur le test split EXACT (mêmes données que results/exp_testCluster/comparison.json).
# Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --split test \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    # --no-show \
    "$@"
