#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP du sweep exp23, pendant de scripts/rl/run_exp23.sh. exp23 utilise les MÊMES données
# et prix VARIABLES qu'exp21/exp21b (la feature de timing est purement côté obs RL, invisible du
# MILP) → on réutilise configs/exp21_overfit_${CASE}.yaml. Le solve MILP est donc identique à
# exp21b ; seuls les noms de sortie changent (milp_exp23_*).
#
# Usage : ./scripts/milp/run_exp23.sh [CASE] [extra python args...]
#   CASE : hiver_haute (défaut) | ete_haute

CASE="${1:-${CASE:-hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi

WINDOW="${CASE}_4d"
CONFIG="configs/exp21_overfit_${CASE}.yaml"
PV="data/irradiance_simu_${WINDOW}.csv"
LOAD="data/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/milp_exp23_${WINDOW}_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp23_${WINDOW}_plan.csv"

for f in "$CONFIG" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Config attendu : configs/exp21_overfit_${CASE}.yaml (prix variables, partagé avec exp21b)."
        echo "Données attendues : data/*_simu_${WINDOW}.csv"
        exit 1
    fi
done

echo "[sim] exp23 MILP | case=$CASE | PV=$PV | load=$LOAD -> $OUT"
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --score-days 2 \
    --cap-horizon \
    "$@"
