#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP du sweep exp29, pendant de scripts/rl/run_exp29.sh. exp29 n'ajoute qu'un terme de
# reward RL (r_charge_hold, sigma_charge_hold) + 2 features d'obs (timing_v3) — purement côté
# agent, INVISIBLE du MILP. Le solve MILP est donc identique à exp25/exp27/exp28 : mêmes
# données/prix, on réutilise configs/exp21_overfit_${CASE}.yaml. Seuls les noms de sortie changent
# (milp_exp29_*) ; le CSV est byte-équivalent à milp_exp28_* (on peut aussi réutiliser directement
# les sorties exp28 pour la comparaison).
#
# Usage : ./scripts/milp/run_exp29.sh [CASE] [extra python args...]
#   CASE : hiver_haute (défaut) | ete_haute

CASE="${1:-${CASE:-hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi

WINDOW="${CASE}_4d"
CONFIG="configs/exp21_overfit_${CASE}.yaml"
PV="data/irradiance_simu_${WINDOW}.csv"
LOAD="data/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/milp_exp29_${WINDOW}_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp29_${WINDOW}_plan.csv"

for f in "$CONFIG" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Config attendu : configs/exp21_overfit_${CASE}.yaml (prix variables, partagé avec exp25+)."
        echo "Données attendues : data/*_simu_${WINDOW}.csv"
        exit 1
    fi
done

echo "[sim] exp29 MILP | case=$CASE | PV=$PV | load=$LOAD -> $OUT"
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --score-days 2 \
    --cap-horizon \
    "$@"
