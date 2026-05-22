#!/usr/bin/env bash

# To run this script, first run the data extraction with :
# python data/extract_pyrano_simu.py --nbD 1 --month 5

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp01_bis_perfect_foresight.yaml"
MODEL="results/exp01_bis_perfect_foresight/sac_model.zip"
OUT="monitoring/runs/rl_exp01_bis_monitoring_table.csv"

cd "$ROOT"

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord : python experiments/run_experiment.py --config $CONFIG"
    exit 1
fi

python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --forecast data/pyrano_simu_may.csv \
    --out "$OUT" \
    "$@"
