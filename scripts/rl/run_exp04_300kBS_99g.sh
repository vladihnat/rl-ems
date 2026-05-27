#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp04_load_forecast.yaml"
MODEL="results/exp04_load_forecast_300kBS_g99/sac_model.zip"
OUT="monitoring/runs/rl_exp04_300kBS_g99_monitoring_table.csv"


cd "$ROOT"

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord : python experiments/run_experiment.py --config $CONFIG"
    exit 1
fi

python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --forecast data/pyrano_simu.csv \
    --out "$OUT" \
    "$@"
