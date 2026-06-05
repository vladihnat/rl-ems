#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/expTestCluster.yaml"
MODEL="results/exp_testCluster/sac_model.zip"
OUT="monitoring/runs/rl_expTestCluster_exact_monitoring_table.csv"

cd "$ROOT"

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord : micromamba run -n stageCorse python experiments/run_experiment.py --config $CONFIG"
    exit 1
fi

# Rejoue le RL sur le test split EXACT (mêmes données que results/exp_testCluster/comparison.json).
# Env strictement imposé : micromamba stageCorse.
micromamba run -n stageCorse python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --split test \
    --out "$OUT" \
    "$@"
