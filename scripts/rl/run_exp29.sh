#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp29 (exp28 best arm + terme sigma_charge_hold / features timing_v3)
# sur la fenêtre overfit (train==eval, 4 jours). exp29 attaque le biais « CHARGER trop tôt » (75 %
# du résiduel hiver exp28 run_008, confirmé été run_010) : le RL charge dès 11h (export 0.201
# sacrifié) à puissance partielle au lieu de concentrer la charge dans la fenêtre d'export min
# (0.181-0.191) comme le MILP. But : voir le SoC rester PLAT le matin (tout le PV exporté) puis
# monter à −100 kW dans la fenêtre pas chère, et resserrer le gap sous les 2.63 % (hiver) /
# 0.78 % (été) d'exp28. Pendant MILP : scripts/milp/run_exp29.sh (baseline identique à exp25+,
# réutilise exp21_overfit).
#
# Usage : ./scripts/rl/run_exp29.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp29_hiver_haute (défaut) | exp29_ete_haute  (ou forme courte hiver_haute | ete_haute)
#   RUN   : run_000 (défaut) | run_NNN   (run_000/006/012 = arm v3=off σ_charge=0 = contrôle exp28)

_SWEEP_ARG="${1:-${SWEEP:-exp29_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
case "$_SWEEP_ARG" in
    exp29_*) SWEEP="$_SWEEP_ARG" ;;
    *)       SWEEP="exp29_${_SWEEP_ARG}" ;;
esac

RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

CASE="${SWEEP#exp29_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (sigma_charge_hold/timing_v3 varient)
MODEL="results/${SWEEP}/${RUN}/best_model.zip"
PV="data/irradiance_simu_${WINDOW}.csv"
LOAD="data/load_simu_${WINDOW}.csv"
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

echo "[sim] exp29 RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
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
