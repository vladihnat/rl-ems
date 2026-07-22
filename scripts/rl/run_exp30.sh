#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp30 (winner exp29 + terme sigma_discharge_hold / fix
# charge_hold_deadline) sur la fenêtre overfit (train==eval, 4 jours). exp30 attaque les
# « pauses » du soir (~25 % du résiduel exp28/29 : le MILP importe 2 pas à HP 0.2475 pour tenir
# le stock jusqu'au pic export 21:45/0.442, le RL arrive au pic à moitié vide — 27 vs 57 kWh
# exportés au pic, audit exp29 run_011) + fiabilise le terme charge (deadline = min faisable
# borné au prochain pic export). But : voir un MÊME run tenir le matin (résiduel ≤ +1.6 €) ET
# le pic export (≥ 50 kWh à 0.442), gap hiver < 2.6 % / été < 0.7 %.
# Pendant MILP : scripts/milp/run_exp30.sh (baseline identique à exp25+, réutilise exp21_overfit).
#
# Usage : ./scripts/rl/run_exp30.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp30_hiver_haute (défaut) | exp30_ete_haute  (ou forme courte hiver_haute | ete_haute)
#   RUN   : run_000 (défaut) | run_NNN   (run_000/006/012 = arm deadline=off σ_serve=0
#           = contrôle = winner exp29 sur seeds frais 43/44/45)

_SWEEP_ARG="${1:-${SWEEP:-exp30_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
case "$_SWEEP_ARG" in
    exp30_*) SWEEP="$_SWEEP_ARG" ;;
    *)       SWEEP="exp30_${_SWEEP_ARG}" ;;
esac

RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

CASE="${SWEEP#exp30_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (sigma_discharge_hold/deadline varient)
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

echo "[sim] exp30 RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
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
