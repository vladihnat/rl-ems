#!/usr/bin/env bash
# Upload le code local vers le cluster (exclut résultats, données brutes, cache).
#
# Usage : ./cluster/transfer/push.sh [USER@HOST] [REMOTE_PATH]
#
# Variables par défaut (modifier ici ou via l'env) :
REMOTE_HOST=${REMOTE_HOST:-HERRERA-NATIVI_V@193.48.30.217}
REMOTE_PATH=${REMOTE_PATH:-/scratch/HERRERA-NATIVI_V/microgrid-rl}

set -euo pipefail

REMOTE_HOST=${1:-$REMOTE_HOST}
REMOTE_PATH=${2:-$REMOTE_PATH}
REMOTE="${REMOTE_HOST}:${REMOTE_PATH}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[push] ${ROOT} → ${REMOTE}"

# Créer le répertoire distant si inexistant
ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_PATH}'"

rsync -vau --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='.claude/' \
    --exclude='.vscode/' \
    --exclude='.pytest_cache/' \
    --exclude='results/' \
    --exclude='monitoring/runs/' \
    --exclude='data/raw/' \
    --exclude='data/clean/' \
    --exclude='data/old/' \
    --exclude='data/extra_cases/' \
    --exclude='data/simu/irradiance_simulation.csv' \
    --exclude='data/simu/load_simulation.csv' \
    --exclude='data/pyrano_simu.csv' \
    --exclude='data/*.py' \
    --exclude='rapport/' \
    --exclude='scratch/' \
    --exclude='duchaud-JL/' \
    --exclude='scripts/' \
    --exclude='tests/' \
    --exclude='cluster/sweep/manifests/' \
    --exclude='temp_md/'\
    --exclude='RL_communication-flow.md' \
    --exclude='test_compat.py'\
    --exclude='prompts.md'\
    --exclude='README.md'\
    "$ROOT/" "$REMOTE/"

echo "[push] Terminé."
