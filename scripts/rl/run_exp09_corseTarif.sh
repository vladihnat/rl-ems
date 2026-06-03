#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/exp09_corseTarif.yaml"
MODEL="results/exp05_cc_quadBat_300kBS_g99/sac_model.zip" # not the proper model
OUT="monitoring/runs/rl_exp09_corseTarif_monitoring_table.csv"

cd "$ROOT"

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord : micromamba run -n stageCorse python experiments/run_experiment.py --config $CONFIG"
    exit 1
fi

# Env strictement imposé : micromamba stageCorse.
micromamba run -n stageCorse python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --out "$OUT" \
    "$@"
