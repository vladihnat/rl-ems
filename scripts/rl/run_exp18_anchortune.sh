#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Sweep de tuning de l'ancre exp18 (demo_frac × λ sur le combo gagnant buffer+loss ON).
# ⚠ À AJUSTER après le sweep : mettre RUN = meilleur run (gap_best) de exp18_anchortune.
RUN="run_001"   # placeholder (demo_frac=0.25, λ=1.0, seed 40) — remplacer par le meilleur run.
CONFIG="configs/sweeps/exp18_anchortune/${RUN}.yaml"
MODEL="results/exp18_anchortune/${RUN}/best_model.zip"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement.
# 4 fenêtres de 2 jours isolant saison météo × régime de prix d'export :
#   hiver_haute  janv.  haute(cher)  PV faible | hiver_basse  mars  basse  PV faible
#   ete_haute    juil.  haute(cher)  PV fort   | ete_basse    juin  basse  PV fort
# 1er arg : un des 4 cas (défaut hiver_haute) ou 'all' pour les 4 en batch ; le reste
# des arguments est transmis tel quel au module Python (ex. --forecast/--load-csv).
# ----------------------------------------------------------------------------
CASES_ALL=(hiver_haute hiver_basse ete_haute ete_basse)

if [ ! -f "$MODEL" ]; then
    echo "Modèle introuvable : $ROOT/$MODEL"
    echo "Lancez d'abord le sweep, puis fixez RUN au meilleur run :"
    echo "  ./cluster/sweep/launch_sweep.sh exp18_anchortune"
    exit 1
fi

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
    OUT="monitoring/runs/rl_exp18_anchortune_${c}_monitoring_table.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
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
