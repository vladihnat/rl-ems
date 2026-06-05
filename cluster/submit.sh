#!/usr/bin/env bash
# Soumet une seule expérience sur le cluster (push + sbatch train_single).
#
# Usage : ./cluster/submit.sh <exp_name>
#   ex : ./cluster/submit.sh exp07_randSoC
#
# Variables configurables :
REMOTE_HOST=${REMOTE_HOST:-HERRERA-NATIVI_V@193.48.30.217}
REMOTE_PATH=${REMOTE_PATH:-/scratch/HERRERA-NATIVI_V/microgrid-rl}

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EXP_NAME=${1:-}
if [[ -z "$EXP_NAME" ]]; then
    echo "Usage : $0 <exp_name>  (sans extension .yaml)" >&2
    exit 1
fi

CONFIG="configs/${EXP_NAME}.yaml"
if [[ ! -f "${ROOT}/${CONFIG}" ]]; then
    echo "[submit] Config introuvable : ${ROOT}/${CONFIG}" >&2
    exit 1
fi

echo "[submit] Expérience : ${EXP_NAME}"

echo "[submit] Push code..."
"${ROOT}/cluster/transfer/push.sh" "$REMOTE_HOST" "$REMOTE_PATH"

echo "[submit] Soumission..."
JOB_ID=$(ssh "$REMOTE_HOST" \
    "cd '${REMOTE_PATH}' && \
     mkdir -p logs && \
     CONFIG='${CONFIG}' \
     sbatch cluster/jobs/train_single.slurm" \
    | grep -oP '(?<=Submitted batch job )\d+')

echo "[submit] Submitted batch job ${JOB_ID}"
echo ""
echo "Monitoring : ssh ${REMOTE_HOST} \"squeue -u \$USER\""
echo "Pull       : ./cluster/transfer/pull_results.sh ${EXP_NAME}"
