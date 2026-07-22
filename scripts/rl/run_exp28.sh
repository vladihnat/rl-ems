#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp28 (PBRS milp_dual @ γ=0.999 + timing_v2 + terme sigma_hold_export)
# sur la fenêtre overfit (train==eval, 4 jours). exp28 ajoute r_hold_export (miroir export de
# r_export_stock) pour corriger le biais « déployer trop tôt » d'exp27 : le RL vidait la batterie
# AVANT le pic export du soir (~21h à 0.442) au lieu de garder le stock et d'importer pour servir la
# charge comme le MILP (petites pauses ⇒ vente au pic, export 0.442 > import HP 0.2475). But : voir
# le RL TENIR le SoC jusqu'au pic et resserrer le gap sous les 3.17 % d'exp27 run_003.
# Pendant MILP : scripts/milp/run_exp28.sh (baseline identique à exp27, réutilise exp21_overfit).
#
# Usage : ./scripts/rl/run_exp28.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp28_hiver_haute (défaut) | exp28_ete_haute  (ou forme courte hiver_haute | ete_haute)
#   RUN   : run_000 (défaut) | run_NNN   (run_006/012 = arm σ_hold=0 = contrôle exp27)

_SWEEP_ARG="${1:-${SWEEP:-exp28_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
case "$_SWEEP_ARG" in
    exp28_*) SWEEP="$_SWEEP_ARG" ;;
    *)       SWEEP="exp28_${_SWEEP_ARG}" ;;
esac

RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

CASE="${SWEEP#exp28_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (sigma_hold_export/sigma_store varient)
MODEL="results/${SWEEP}/${RUN}/best_model.zip"
PV="data/simu/irradiance_simu_${WINDOW}.csv"
LOAD="data/simu/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/rl_${SWEEP}_${RUN}_monitoring_table.csv"

for f in "$CONFIG" "$MODEL" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Générez puis lancez le sweep ${SWEEP} (cf. cluster/sweep/manifests/${SWEEP}.json),"
        echo "et rapatriez results/${SWEEP}/${RUN}/ avant ce script."
        exit 1
    fi
done

VEC="results/${SWEEP}/${RUN}/best_vecnormalize.pkl"
VEC_FLAG=""
[ -f "$VEC" ] && VEC_FLAG="--vec-normalize $VEC"

echo "[sim] exp28 RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
# --cap-horizon : ne rejoue que les pas vus à l'entraînement (3 j sur 4, cf. truncation fix).
# --score-days 2 : coût scoré sur les jours 1-2 ; le jour 3 tamponne le dump SoC terminal du MILP.
micromamba run -n stageCorse python -m monitoring.run_optimization_example \
    --config "$CONFIG" \
    --model "$MODEL" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --score-days 2 \
    --cap-horizon \
    $VEC_FLAG \
    "$@"
