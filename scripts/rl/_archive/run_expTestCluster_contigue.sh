#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="configs/expTestCluster.yaml"
MODEL="results/exp_testCluster/sac_model.zip"
PV_SIMU="data/irradiance_simulation.csv"
LOAD_SIMU="data/load_simulation.csv"
OUT="monitoring/runs/rl_expTestCluster_contigue_monitoring_table.csv"

cd "$ROOT"

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord : micromamba run -n stageCorse python experiments/run_experiment.py --config $CONFIG"
    exit 1
fi

# Rejoue le RL sur une fenêtre de simulation CONTIGUË (plots lisibles, SoC physiquement continue).
# Générer les CSV simu au besoin :
#   micromamba run -n stageCorse python data/extract_pyrano_simu.py --nbD 7 --month 1 --startDate 1 --usage simu
# Env strictement imposé : micromamba stageCorse.
micromamba run -n stageCorse python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --forecast "$PV_SIMU" \
    --measures "$PV_SIMU" \
    --load-csv "$LOAD_SIMU" \
    --out "$OUT" \
    "$@"
