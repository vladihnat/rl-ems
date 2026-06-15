#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Best run : run_004 (gap_best = 15.71%)
CONFIG="configs/sweeps/exp16_warmstart/run_004.yaml"
MODEL="results/exp16_warmstart/run_004/best_model.zip"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement.
#
# Le config pointe sur data/irradiance_training.csv (set d'entraînement, 28 j).
# Sans override, le monitoring rejouerait donc tout le TRAIN — fuite de données
# et impossibilité d'isoler les régimes. On fournit ici 4 fenêtres de 2 jours
# (semaine) qui isolent saison météo × régime de prix d'export :
#
#   cas           mois    saison export   PV
#   hiver_haute   janv.   haute  (cher)   faible
#   hiver_basse   mars    basse  (bas)    faible
#   ete_haute     juil.   haute  (cher)   fort
#   ete_basse     juin    basse  (bas)    fort
#
# Régénérer : data/extract_pyrano_simu.py --usage simu --nbD 2 --month M --startDate D
#             puis renommer en data/irradiance_simu_<cas>.csv / load_simu_<cas>.csv
# ----------------------------------------------------------------------------
CASES_ALL=(hiver_haute hiver_basse ete_haute ete_basse)

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord le sweep : micromamba run -n stageCorse python experiments/run_sweep.py --sweep configs/sweeps/exp16_warmstart/"
    exit 1
fi

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
    OUT="monitoring/runs/rl_exp16_warmstart_${c}_monitoring_table.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
    # Env strictement imposé : micromamba stageCorse.
    micromamba run -n stageCorse python -m monitoring.run_optimization_example \
        --config "$CONFIG" \
        --model "$MODEL" \
        --forecast "$PV" \
        --load-csv "$LOAD" \
        --out "$OUT" \
        --score-days 1 \
        $SHOW_FLAG \
        "$@"
done
