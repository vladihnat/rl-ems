#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp03_1M_300.yaml"
OUT="monitoring/runs/milp_exp03_1M_300_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp03_1M_300_plan.csv"

cd "$ROOT"

python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast data/pyrano_simu.csv \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    "$@"
