#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Référence MILP pour le sweep de tuning de l'ancre exp18_anchortune. Le MILP ne dépend que du
# config (batterie 100 kW, prix) — identique pour tous les bras ; on prend celui du run RL retenu.
# ⚠ À AJUSTER après le sweep : mettre RUN = meilleur run (gap_best) de exp18_anchortune.
RUN="run_005"   # placeholder — remplacer par le meilleur run RL retenu.
CONFIG="configs/sweeps/exp18_anchortune/${RUN}.yaml"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement.
# 4 fenêtres de 2 jours isolant saison météo × régime de prix d'export :
#   hiver_haute  janv.  haute(cher)  PV faible | hiver_basse  mars  basse  PV faible
#   ete_haute    juil.  haute(cher)  PV fort   | ete_basse    juin  basse  PV fort
# 1er arg : un des 4 cas (défaut hiver_haute) ou 'all' ; le reste est transmis au module Python.
# ----------------------------------------------------------------------------
CASES_ALL=(hiver_haute hiver_basse ete_haute ete_basse)

CASE="${1:-hiver_haute}"
if [ "$#" -gt 0 ]; then shift; fi

if [ "$CASE" = "all" ]; then
    CASES=("${CASES_ALL[@]}")
    SHOW_FLAG="--no-show"   # batch non-interactif (les CSV restent écrits)
else
    CASES=("$CASE")
    SHOW_FLAG=""            # cas unique : on ouvre les fenêtres matplotlib
fi

for c in "${CASES[@]}"; do
    PV="data/irradiance_simu_${c}.csv"
    LOAD="data/load_simu_${c}.csv"
    OUT="monitoring/runs/milp_exp18_anchortune_${c}_monitoring_table.csv"
    PLAN_OUT="monitoring/runs/milp_exp18_anchortune_${c}_plan.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
    micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
        --config "$CONFIG" \
        --forecast "$PV" \
        --load-csv "$LOAD" \
        --out "$OUT" \
        --plan-out "$PLAN_OUT" \
        --score-days 1 \
        $SHOW_FLAG \
        "$@"
done
