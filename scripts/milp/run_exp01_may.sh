#!/usr/bin/env bash
# To run this script, first run the data extraction with :
# python data/extract_pyrano_simu.py --nbD 1 --month 5

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp01_bis_perfect_foresight.yaml"
OUT="monitoring/runs/milp_exp01_bis_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp01_bis_plan.csv"

cd "$ROOT"

python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast data/pyrano_simu_may.csv \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    "$@"