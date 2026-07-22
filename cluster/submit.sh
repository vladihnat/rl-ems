#!/usr/bin/env bash
# Soumet une seule expérience sur le cluster (push + sbatch train_single).
#
# Usage : ./cluster/submit.sh <exp_name>
#   ex : ./cluster/submit.sh exp31_c1_hiver_haute
#   <exp_name> peut être un nom nu (cherché dans configs/{overfit,holdout,_archive}/)
#   ou un chemin relatif complet (ex : configs/overfit/exp28_hiver_haute).
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

# Résout le config : chemin complet fourni, sinon cherché dans les sous-dossiers famille.
CONFIG=""
for CAND in "configs/${EXP_NAME}.yaml" \
            "configs/overfit/${EXP_NAME}.yaml" \
            "configs/holdout/${EXP_NAME}.yaml" \
            "configs/_archive/${EXP_NAME}.yaml"; do
    if [[ -f "${ROOT}/${CAND}" ]]; then CONFIG="$CAND"; break; fi
done
if [[ -z "$CONFIG" ]]; then
    echo "[submit] Config introuvable pour '${EXP_NAME}' (cherché dans configs/{,overfit/,holdout/,_archive/})." >&2
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
