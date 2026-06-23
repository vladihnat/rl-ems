#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Comparaison visuelle du sanity check « overfit » (exp21) : rejoue un modèle RL ENTRAÎNÉ sur la
# fenêtre hiver_haute (4 j) et l'évalue sur cette MÊME fenêtre (overfit, pas de held-out). Pendant
# MILP : scripts/milp/run_exp21_overfit.sh. But — voir si le RL reproduit le dispatch MILP sur des
# données qu'il a vues (sinon le pb est la formulation, pas la généralisation).
#
# 1er arg : run du sweep exp21_overfit_hiver_haute (défaut run_000). Axe = la VARIANTE à inspecter :
#   run_000 = (norm_obs=false, sigma_store=1.0, ent_coef=auto)  ← contrôle identique à exp19b
#   run_007 = (norm_obs=true,  sigma_store=0.3, ent_coef=0.05)  ← coin « full fix »
# Lancer p.ex. ./scripts/rl/run_exp21_overfit.sh run_000  PUIS le meilleur run (cf. aggregate_results).
RUN="${1:-run_000}"
if [ "$#" -gt 0 ]; then shift; fi

# WINDOW surchargeable : WINDOW=ete_haute_4d ./scripts/rl/run_exp21_overfit.sh <run>
WINDOW="${WINDOW:-hiver_haute_4d}"          # fenêtre overfit (ré-extractible pour ete_haute)
# SWEEP surchargeable : SWEEP=exp21b_hiver_haute pour le re-sweep POST-FIX (cf. plan, étape 5) ;
# défaut = sweep pré-fix dérivé de WINDOW.
SWEEP="${SWEEP:-exp21_overfit_${WINDOW%_4d}}"
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

echo "[sim] overfit RL | run=$RUN | PV=$PV | load=$LOAD -> $OUT"
# Env strictement imposé : micromamba stageCorse.
# --cap-horizon : ne rejoue QUE les pas de décision vus à l'entraînement (max_steps = n_steps - horizon
#   = 3 j) — sinon le 4e jour, jamais joué à l'entraînement, fausse le replay (cf. plan, étape 2).
# --score-days 2 : fenêtre 4 j → 3 j déroulés ; coût scoré sur les jours 1-2 (le jour 3 tamponne le
#   dump SoC terminal du MILP, cf. mémoire sim-score-on-n-minus-1).
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
