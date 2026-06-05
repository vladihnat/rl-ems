#!/usr/bin/env bash
# Push le code et soumet un job array SLURM pour un sweep.
#
# Prérequis : avoir d'abord généré le sweep avec generate_sweep.py
#
# Usage :
#   ./cluster/sweep/launch_sweep.sh <sweep_name>
#   ./cluster/sweep/launch_sweep.sh <sweep_name> --no-push
#
# Variables configurables :
REMOTE_HOST=${REMOTE_HOST:-HERRERA-NATIVI_V@193.48.30.217}
REMOTE_PATH=${REMOTE_PATH:-/scratch/HERRERA-NATIVI_V/microgrid-rl}

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SWEEP_NAME=${1:-}
NO_PUSH=false
# Throttle = nb max de tâches d'array concurrentes (sbatch --array=0-N%THROTTLE).
# Défaut 8 : courtois sur un cluster partagé (cf. cluster Orsu, 28 nœuds Intel) sans
# saturer. Mettre 0 pour désactiver le plafond (toutes les tâches en parallèle).
THROTTLE=8

if [[ -z "$SWEEP_NAME" ]]; then
    echo "Usage : $0 <sweep_name> [--no-push] [--throttle K]" >&2
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push)   NO_PUSH=true ;;
        --throttle)  THROTTLE="${2:?--throttle requiert une valeur}"; shift ;;
        --throttle=*) THROTTLE="${1#*=}" ;;
        *) echo "[launch_sweep] option inconnue: $1" >&2; exit 1 ;;
    esac
    shift
done

if ! [[ "$THROTTLE" =~ ^[0-9]+$ ]]; then
    echo "[launch_sweep] --throttle doit être un entier (reçu: '$THROTTLE')" >&2
    exit 1
fi

MANIFEST_LOCAL="${ROOT}/cluster/sweep/manifests/${SWEEP_NAME}.json"
MANIFEST_REMOTE="cluster/sweep/manifests/${SWEEP_NAME}.json"

if [[ ! -f "$MANIFEST_LOCAL" ]]; then
    echo "[launch_sweep] Manifest introuvable : ${MANIFEST_LOCAL}" >&2
    echo "[launch_sweep] Exécutez d'abord :" >&2
    echo "  python cluster/sweep/generate_sweep.py --sweep-name ${SWEEP_NAME} ..." >&2
    exit 1
fi

N=$(python3 -c "import json; print(len(json.load(open('${MANIFEST_LOCAL}'))))")
echo "[launch_sweep] sweep=${SWEEP_NAME}  runs=${N}"

# 1. Uploader le manifest (toujours, même avec --no-push)
echo "[launch_sweep] Upload manifest..."
ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_PATH}/cluster/sweep/manifests'"
rsync -vau --progress "$MANIFEST_LOCAL" \
    "${REMOTE_HOST}:${REMOTE_PATH}/cluster/sweep/manifests/"

# 2. Uploader les configs générées
SWEEP_CFG_LOCAL="${ROOT}/configs/sweeps/${SWEEP_NAME}/"
SWEEP_CFG_REMOTE="${REMOTE_HOST}:${REMOTE_PATH}/configs/sweeps/${SWEEP_NAME}/"
if [[ -d "$SWEEP_CFG_LOCAL" ]]; then
    echo "[launch_sweep] Upload configs sweep..."
    ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_PATH}/configs/sweeps/${SWEEP_NAME}'"
    rsync -vau --progress "$SWEEP_CFG_LOCAL" "$SWEEP_CFG_REMOTE"
fi

# 3. Push code (sauf --no-push)
if [[ "$NO_PUSH" == false ]]; then
    echo "[launch_sweep] Push code..."
    "${ROOT}/cluster/transfer/push.sh" "$REMOTE_HOST" "$REMOTE_PATH"
fi

# 4. Soumettre le job array (avec plafond de concurrence %THROTTLE si > 0)
if [[ "$THROTTLE" -gt 0 ]]; then
    ARRAY_SPEC="0-$((N-1))%${THROTTLE}"
else
    ARRAY_SPEC="0-$((N-1))"
fi
echo "[launch_sweep] Soumission sbatch array ${ARRAY_SPEC}..."
JOB_ID=$(ssh "$REMOTE_HOST" \
    "cd '${REMOTE_PATH}' && \
     mkdir -p logs && \
     MANIFEST='${MANIFEST_REMOTE}' \
     sbatch --array='${ARRAY_SPEC}' \
            --job-name='${SWEEP_NAME}' \
            cluster/jobs/train_array.slurm" \
    | grep -oP '(?<=Submitted batch job )\d+')

echo "[launch_sweep] Submitted batch job ${JOB_ID}  (array ${ARRAY_SPEC})"
echo ""
echo "Monitoring :"
echo "  ssh ${REMOTE_HOST} \"squeue -u \$USER\""
echo ""
echo "Rapatrier les résultats :"
echo "  ./cluster/transfer/pull_results.sh ${SWEEP_NAME}"
