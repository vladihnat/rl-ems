#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Référence MILP du run_007 d'exp18 (ancre complète, gap_best = 4.8%).
# Le MILP ne dépend que du config (batterie 100 kW, prix) — identique pour tous
# les bras d'exp18 ; on prend celui du run RL retenu pour une comparaison cohérente.
CONFIG="configs/sweeps/exp18_anchor/run_007.yaml"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement.
#
# Le config pointe sur data/irradiance_training.csv (set d'entraînement, 28 j).
# Sans override, le MILP planifierait donc sur tout le TRAIN. On fournit ici
# 4 fenêtres de 2 jours (semaine) isolant saison météo × régime de prix d'export :
#
#   cas           mois    saison export   PV
#   hiver_haute   janv.   haute  (cher)   faible
#   hiver_basse   mars    basse  (bas)    faible
#   ete_haute     juil.   haute  (cher)   fort
#   ete_basse     juin    basse  (bas)    fort
#
# CHANGER LA PÉRIODE — 3 façons :
#   1. un des 4 cas nommés ci-dessous (1er argument) ;
#   2. période arbitraire : passer --forecast X.csv --load-csv Y.csv en plus
#      (tout argument après le cas est transmis tel quel au module Python) ;
#   3. régénérer un cas : data/extract_pyrano_simu.py --usage simu --nbD 2
#      --month M --startDate D  puis renommer en
#      data/irradiance_simu_<cas>.csv / load_simu_<cas>.csv
# ----------------------------------------------------------------------------
CASES_ALL=(hiver_haute hiver_basse ete_haute ete_basse)

# 1er argument : nom du cas (défaut hiver_haute) ou 'all' pour les 4 en batch.
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
    OUT="monitoring/runs/milp_exp18_anchor_${c}_monitoring_table.csv"
    PLAN_OUT="monitoring/runs/milp_exp18_anchor_${c}_plan.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
    # Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
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
