#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Rejoue un modèle RL du sweep exp21b (post-fix truncation + cap-horizon) sur la fenêtre overfit
# (train==eval, 4 jours). Pendant MILP : scripts/milp/run_exp21_overfit.sh.
#
# Usage : ./scripts/rl/run_exp21b.sh [SWEEP] [RUN] [extra python args...]
#   SWEEP : exp21b_hiver_haute (défaut) | exp21b_ete_haute
#           ou forme courte : hiver_haute | ete_haute (le préfixe exp21b_ est ajouté auto)
#   RUN   : run_000 (défaut) | run_NNN
#
# Env vars surchargeables : SWEEP=exp21b_ete_haute ./scripts/rl/run_exp21b.sh run_007

# 1er arg : SWEEP (forme longue ou courte) — défaut hiver_haute
_SWEEP_ARG="${1:-${SWEEP:-exp21b_hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi
# Normalisation : accepter la forme courte (hiver_haute ou ete_haute)
case "$_SWEEP_ARG" in
    exp21b_*) SWEEP="$_SWEEP_ARG" ;;
    *)        SWEEP="exp21b_${_SWEEP_ARG}" ;;
esac

# 2e arg : RUN
RUN="${1:-${RUN:-run_000}}"
if [ "$#" -gt 0 ]; then shift; fi

# Dériver le cas de simulation depuis le nom du sweep (exp21b_hiver_haute → hiver_haute)
CASE="${SWEEP#exp21b_}"
WINDOW="${CASE}_4d"
CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"     # config PAR RUN (norm_obs varie → replay fidèle)
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

# Modèle BEST → ses stats VecNormalize sont best_vecnormalize.pkl (l'auto-détection ne trouve que
# vec_normalize.pkl, stats du modèle FINAL). On les passe explicitement pour les runs norm_obs=true ;
# pour norm_obs=false ce fichier n'existe pas et l'obs reste brute (correct).
VEC="results/${SWEEP}/${RUN}/best_vecnormalize.pkl"
VEC_FLAG=""
[ -f "$VEC" ] && VEC_FLAG="--vec-normalize $VEC"

echo "[sim] exp21b RL | sweep=$SWEEP | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
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
