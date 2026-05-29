#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp07_randSoC.yaml"
MODEL="results/exp07_randSoC_300Bs_g99/sac_model.zip"
OUT="monitoring/runs/rl_exp07_randSoC_300Bs_g99_monitoring_table.csv"


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
