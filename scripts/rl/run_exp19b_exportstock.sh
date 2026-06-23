#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Best run du sweep exp19b (PBRS + coût d'opportunité d'export de stock). À AJUSTER après
# aggregate_results.py : pointer le run au plus petit sigma_export_stock qui ferme le gap
# ete_basse sans dégrader haute. Placeholder = run_001 (sigma_export_stock=0.0 = contrôle exp19).
RUN="run_000"   #placeholder (pour l'instant 0 et 5, manquent moitie des results)
CONFIG="configs/sweeps/exp19b_exportstock/${RUN}.yaml"
MODEL="results/exp19b_exportstock/${RUN}/best_model.zip"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement. 4 fenêtres de 2 jours
# (semaine) isolant saison météo × régime de prix d'export :
#
#   cas           mois    saison export   PV
#   hiver_haute   janv.   haute  (cher)   faible
#   hiver_basse   mars    basse  (bas)    faible
#   ete_haute     juil.   haute  (cher)   fort
#   ete_basse     juin    basse  (bas)    fort  <- cas cible de la révision exp19b
# ----------------------------------------------------------------------------
CASES_ALL=(hiver_haute hiver_basse ete_haute ete_basse)

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord le sweep exp19b (cf. cluster/sweep/manifests/exp19b_exportstock.json)."
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
    OUT="monitoring/runs/rl_exp19b_exportstock_${c}_monitoring_table.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
    # Env strictement imposé : micromamba stageCorse.
    # --score-days 1 : fenêtres de 2 jours → on n'évalue le coût que sur le jour 1
    # (le jour 2 sert de tampon « lendemain », cf. mémoire sim-score-on-n-minus-1).
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
