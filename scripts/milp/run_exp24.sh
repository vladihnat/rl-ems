#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP du check exp24 (relâchement no-grid-charging, milpIncoherences.md #5), pendant de
# scripts/rl/run_exp24.sh. CONTRAIREMENT à exp22, le MILP exp24 DÉPEND du mode de charge :
# on passe --pv-charge-mode au solveur (surplus=strict | total=relâché Duchaud-JL). On réutilise
# configs/exp24_${CASE}.yaml (batterie/grille/PV/charge identiques ; le champ battery.pv_charge_mode
# du config est écrasé par --pv-charge-mode). Sous prix fixes export 0.04 < import 0.15, la
# complémentarité import/export ne mord jamais (MILP propre, aucun money-pump possible).
#
# Usage : ./scripts/milp/run_exp24.sh [CASE] [MODE] [extra python args...]
#   CASE : hiver_haute (défaut) | ete_haute
#   MODE : surplus (défaut) | total
#
# Env vars surchargeables : CASE=ete_haute MODE=total ./scripts/milp/run_exp24.sh

CASE="${1:-${CASE:-hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
MODE="${1:-${MODE:-surplus}}"
if [ "$#" -gt 0 ]; then shift; fi

# Tie-break lexicographique ε : sous prix plats, l'optimum MILP est dégénéré (timing libre) et le
# tracé plot_power clignote d'un solve à l'autre. ε>0 le rend unique → figure déterministe, coût
# rapporté inchangé. Surchargeable : TIE_BREAK_EPS=0 désactive.
TIE_BREAK_EPS="${TIE_BREAK_EPS:-1e-5}"

WINDOW="${CASE}_4d"
CONFIG="configs/exp24_${CASE}.yaml"
PV="data/irradiance_simu_${WINDOW}.csv"
LOAD="data/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/milp_exp24_${WINDOW}_${MODE}_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp24_${WINDOW}_${MODE}_plan.csv"

for f in "$CONFIG" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Config attendu : configs/exp24_${CASE}.yaml (partagé avec le check exp24)."
        echo "Données attendues : data/*_simu_${WINDOW}.csv"
        exit 1
    fi
done

echo "[sim] exp24 MILP | case=$CASE | mode=$MODE | PV=$PV | load=$LOAD -> $OUT"
# --cap-horizon : le MILP planifie le même horizon que l'entraînement RL (3 j sur 4).
# --score-days 2 : coût scoré sur les jours 1-2 ; le jour 3 tamponne le dump SoC terminal du MILP.
# --pv-charge-mode : sélectionne la formulation de la contrainte no-grid-charging (cf. #5).
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --score-days 2 \
    --cap-horizon \
    --pv-charge-mode "$MODE" \
    --tie-break-eps "$TIE_BREAK_EPS" \
    "$@"
