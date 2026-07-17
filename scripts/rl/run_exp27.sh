#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp27 (PBRS milp_dual @ γ=0.999 + timing_v2 : Φ = σ·λ(t)·(SoC−min)·cap
# avec λ(t) = duals du LP MILP à binaires figés) sur la fenêtre overfit (train==eval, 4 jours). But :
# valider que le potentiel dual au bon γ supprime le « dump t0 » (SoC tenu ~0.5 en t0, pic midi
# évanoui) et resserre le gap hiver sous les 4.25 % de exp25 run_022. Pendant MILP : scripts/milp/run_exp27.sh.
#
# Usage : ./scripts/rl/run_exp27.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp27_hiver_haute (défaut) | exp27_ete_haute  (ou forme courte hiver_haute | ete_haute)
#   RUN   : run_000 (défaut) | run_NNN

_SWEEP_ARG="${1:-${SWEEP:-exp27_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
case "$_SWEEP_ARG" in
    exp27_*) SWEEP="$_SWEEP_ARG" ;;
    *)       SWEEP="exp27_${_SWEEP_ARG}" ;;
esac

RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

CASE="${SWEEP#exp27_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (sigma_store varie → replay fidèle)
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

echo "[sim] exp27 RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
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
