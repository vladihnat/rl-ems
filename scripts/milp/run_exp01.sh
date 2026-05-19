#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp01_perfect_foresight.yaml"
OUT="monitoring/runs/milp_exp01_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp01_plan.csv"

cd "$ROOT"

python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    "$@"
