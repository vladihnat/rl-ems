#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP du sweep exp22 (prix fixes 0.15/0.04), pendant de scripts/rl/run_exp22.sh.
# Le MILP ne dépend pas des axes du sweep RL (norm_obs/sigma_store/norm_reward) → un seul solve
# par période suffit. On réutilise configs/exp22_${CASE}.yaml (batterie/grille/PV/charge identiques
# à exp22 ; les champs training/reward sont ignorés par run_milp_optimization_example).
# Sous export 0.04 < import 0.15, la complémentarité import/export ne mord jamais (MILP trivialement
# propre, aucun money-pump possible).
#
# Usage : ./scripts/milp/run_exp22.sh [CASE] [extra python args...]
#   CASE : hiver_haute (défaut) | ete_haute
#
# Env var surchargeable : CASE=ete_haute ./scripts/milp/run_exp22.sh

CASE="${1:-${CASE:-hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi

WINDOW="${CASE}_4d"
CONFIG="configs/exp22_${CASE}.yaml"
PV="data/irradiance_simu_${WINDOW}.csv"
LOAD="data/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/milp_exp22_${WINDOW}_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp22_${WINDOW}_plan.csv"

for f in "$CONFIG" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Config attendu : configs/exp22_${CASE}.yaml (partagé avec le sweep exp22)."
        echo "Données attendues : data/*_simu_${WINDOW}.csv"
        exit 1
    fi
done

echo "[sim] exp22 MILP | case=$CASE | PV=$PV | load=$LOAD -> $OUT"
# --cap-horizon : le MILP planifie le même horizon que l'entraînement RL (3 j sur 4) pour que
#   MILP et RL soient comparés sur les mêmes pas de décision.
# --score-days 2 : coût scoré sur les jours 1-2 ; le jour 3 tamponne le dump SoC terminal du MILP.
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --score-days 2 \
    --cap-horizon \
    "$@"
