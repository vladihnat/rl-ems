#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# exp31 — évaluation HELD-OUT (hors overfit) d'un modèle entraîné en split train/val/test.
#
# Le modèle a été entraîné sur data/*_train_<regime>.csv (spécialiste) ou data/*_train_combined.csv
# (modèle combiné), avec eval_on_train=false / train_split=0.8 / val_split=0.2 (best-model
# sélectionné sur la tranche de VALIDATION). On le rejoue ici sur une fenêtre de TEST
# DISJOINTE du train (0 timestamp commun, vérifié) : --forecast/--load-csv surchargent
# data.pv_csv/load_csv du config (monitoring/run_optimization_example.py:_build_deployment_env).
#
# Pendant MILP : scripts/milp/run_exp31.sh (MÊMES --score-days/--cap-horizon, sinon gap biaisé).
#
# Usage : ./scripts/rl/run_exp31.sh SWEEP RUN [TEST_CASE] [extra python args...]
#   SWEEP     : exp31_{c1,c2}_{hiver_haute,hiver_basse,ete_haute,ete_basse,combined}
#               c1 = meilleur bras exp28 (sigma_charge_hold=0) ; c2 = meilleur bras exp29 (0.5)
#   RUN       : run_000 | run_001 | run_002   (seeds 40 / 41 / 42)
#   TEST_CASE : défaut = le régime du SWEEP (spécialiste testé sur SON régime).
#               Sinon : un des 4 régimes | combined | all  ('all' = les 4 régimes + combined,
#               utile pour le modèle COMBINÉ testé partout).
#
# Scoring (N-1 + boundary_soc_credit, exclut le dump SoC terminal du MILP) :
#   - fenêtre régime  : 4 j de données -> rollout 3 j (--cap-horizon) -> --score-days 2
#   - fenêtre combinée: 16 j          -> rollout 15 j                 -> --score-days 14

SWEEP="${1:?usage: run_exp31.sh SWEEP RUN [TEST_CASE]}"; shift
RUN="${1:-run_000}"; if [ "$#" -gt 0 ]; then shift; fi

# Régime par défaut = celui du sweep (exp31_c1_hiver_haute -> hiver_haute)
DEFAULT_CASE="${SWEEP#exp31_}"; DEFAULT_CASE="${DEFAULT_CASE#c[12]_}"
TEST_CASE="${1:-$DEFAULT_CASE}"; if [ "$#" -gt 0 ]; then shift; fi

CONFIG="results/${SWEEP}/${RUN}/config_used.yaml"
MODEL="results/${SWEEP}/${RUN}/best_model.zip"
if [ ! -f "$CONFIG" ]; then CONFIG="configs/sweeps/${SWEEP}/${RUN}.yaml"; fi
for f in "$CONFIG" "$MODEL"; do
    [ -f "$f" ] || { echo "Introuvable : $ROOT/$f"; echo "Lancez le sweep ${SWEEP} puis rapatriez results/${SWEEP}/${RUN}/."; exit 1; }
done

VEC="results/${SWEEP}/${RUN}/best_vecnormalize.pkl"
[ -f "$VEC" ] || VEC="results/${SWEEP}/${RUN}/vec_normalize.pkl"
VEC_FLAG=""; [ -f "$VEC" ] && VEC_FLAG="--vec-normalize $VEC"

if [ "$TEST_CASE" = "all" ]; then
    CASES=(hiver_haute hiver_basse ete_haute ete_basse combined)
else
    CASES=("$TEST_CASE")
fi

for CASE in "${CASES[@]}"; do
    if [ "$CASE" = "combined" ]; then
        PV="data/simu/irradiance_simu_combined.csv"; LOAD="data/simu/load_simu_combined.csv"; SCORE_DAYS=14
    else
        PV="data/simu/irradiance_simu_${CASE}_4d.csv"; LOAD="data/simu/load_simu_${CASE}_4d.csv"; SCORE_DAYS=2
    fi
    for f in "$PV" "$LOAD"; do
        [ -f "$f" ] || { echo "Données de test introuvables : $ROOT/$f"; exit 1; }
    done

    OUT="monitoring/runs/rl_${SWEEP}_${RUN}_on_${CASE}_monitoring_table.csv"
    echo "[sim] exp31 RL | model=${SWEEP}/${RUN} | TEST=${CASE} | score_days=${SCORE_DAYS} -> $OUT"
    micromamba run -n stageCorse python -m monitoring.run_optimization_example \
        --config "$CONFIG" \
        --model "$MODEL" \
        --forecast "$PV" \
        --load-csv "$LOAD" \
        --out "$OUT" \
        --score-days "$SCORE_DAYS" \
        --cap-horizon \
        $VEC_FLAG \
        "$@"
done
