#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp03_1M_300.yaml"
MODEL="results/exp03_1M_300/sac_model.zip"
OUT="monitoring/runs/rl_exp03_1M_300_monitoring_table.csv"

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
