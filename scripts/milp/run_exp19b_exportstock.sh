#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Référence MILP du sweep exp19b. Le terme r_export_stock est PURE récompense RL (zéro sur
# l'optimum) → le MILP est identique à exp19 ; on prend le config du run RL retenu pour une
# comparaison cohérente (batterie 100 kW, même scénario). À AJUSTER comme le script RL.
RUN="run_000"   # placeholder — remplacer par le run RL retenu.
CONFIG="configs/sweeps/exp19_store/${RUN}.yaml"

cd "$ROOT"

# ----------------------------------------------------------------------------
# Données de simulation (déploiement) — PAS le set d'entraînement. 4 fenêtres de 2 jours :
#   hiver_haute / hiver_basse / ete_haute / ete_basse (cf. run_exp19b_exportstock.sh RL).
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
    OUT="monitoring/runs/milp_exp19b_exportstock_${c}_monitoring_table.csv"
    PLAN_OUT="monitoring/runs/milp_exp19b_exportstock_${c}_plan.csv"
    if [ ! -f "$PV" ] || [ ! -f "$LOAD" ]; then
        echo "Données de simulation introuvables pour le cas '$c' : $PV / $LOAD"
        echo "Cas valides : ${CASES_ALL[*]} (ou 'all')."
        exit 1
    fi
    echo "[sim] cas=$c | PV=$PV | load=$LOAD -> $OUT"
    # Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
    # --score-days 1 : fenêtres de 2 jours → coût scoré sur le jour 1 seul (cf. mémoire).
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
