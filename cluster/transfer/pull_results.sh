#!/usr/bin/env bash
# Rapatrie les résultats du cluster vers local.
#
# Usage :
#   ./cluster/transfer/pull_results.sh                    # tout results/
#   ./cluster/transfer/pull_results.sh gamma_lr_sweep     # seulement ce sweep
#   ./cluster/transfer/pull_results.sh --dry-run          # aperçu sans copier
#
# Variables par défaut :
REMOTE_HOST=${REMOTE_HOST:-HERRERA-NATIVI_V@193.48.30.217}
REMOTE_PATH=${REMOTE_PATH:-/scratch/HERRERA-NATIVI_V/microgrid-rl}

set -euo pipefail

DRY_RUN=()
SWEEP_NAME=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=(--dry-run) ;;
        -*) echo "[pull] option inconnue: $arg" >&2; exit 1 ;;
        *) SWEEP_NAME="$arg" ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "$SWEEP_NAME" ]]; then
    REMOTE_SRC="${REMOTE_HOST}:${REMOTE_PATH}/results/${SWEEP_NAME}/"
    LOCAL_DST="${ROOT}/results/${SWEEP_NAME}/"
else
    REMOTE_SRC="${REMOTE_HOST}:${REMOTE_PATH}/results/"
    LOCAL_DST="${ROOT}/results/"
fi

echo "[pull] ${REMOTE_SRC} → ${LOCAL_DST}"
[[ ${#DRY_RUN[@]} -gt 0 ]] && echo "[pull] mode dry-run"

mkdir -p "$LOCAL_DST"

rsync -vau --progress "${DRY_RUN[@]}" "$REMOTE_SRC" "$LOCAL_DST"

echo "[pull] Terminé."
