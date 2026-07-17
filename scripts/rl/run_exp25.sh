#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp25 (prix VARIABLES + axes credit-assignment (gamma, timing v1/v2, sigma_store)) sur la
# fenêtre overfit (train==eval, 4 jours). But : valider que gamma=0.999 et/ou timing_features_v2 suppriment le dump t0 et
# capturent l'heure de pointe export 21-22h (cf. audit exp23). Pendant MILP : scripts/milp/run_exp25.sh.
#
# Usage : ./scripts/rl/run_exp25.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp25_hiver_haute (défaut) | exp25_ete_haute  (ou forme courte hiver_haute | ete_haute)
#   RUN   : run_000 (défaut) | run_NNN

_SWEEP_ARG="${1:-${SWEEP:-exp25_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
case "$_SWEEP_ARG" in
    exp25_*) SWEEP="$_SWEEP_ARG" ;;
    *)       SWEEP="exp25_${_SWEEP_ARG}" ;;
esac

RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

CASE="${SWEEP#exp25_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (norm_obs/timing varient → replay fidèle)
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

echo "[sim] exp25 RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
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
