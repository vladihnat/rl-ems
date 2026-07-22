#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP du sanity check « overfit » (exp21), pendant de scripts/rl/run_exp21_overfit.sh.
# Le MILP ne dépend PAS des axes du sweep (norm_obs/sigma_store/ent_coef sont de la pure récompense
# RL) → un seul solve sur la fenêtre suffit, identique pour tous les runs. On prend le config de
# BASE (battery/grid/time/pv/load identiques à tous les run_NNN). À overlayer sur la sortie RL.
# WINDOW surchargeable : WINDOW=ete_haute_4d ./scripts/milp/run_exp21_overfit.sh
WINDOW="${WINDOW:-hiver_haute_4d}"
CONFIG="configs/overfit/exp21_overfit_${WINDOW%_4d}.yaml"   # ..._hiver_haute.yaml / ..._ete_haute.yaml
PV="data/simu/irradiance_simu_${WINDOW}.csv"
LOAD="data/simu/load_simu_${WINDOW}.csv"
OUT="monitoring/runs/milp_exp21_overfit_${WINDOW}_monitoring_table.csv"
PLAN_OUT="monitoring/runs/milp_exp21_overfit_${WINDOW}_plan.csv"

for f in "$CONFIG" "$PV" "$LOAD"; do
    if [ ! -f "$f" ]; then
        echo "Introuvable : $ROOT/$f"
        echo "Extrayez la fenêtre (extract_pyrano_simu.py --nbD 4 --month 1 --startDate 1 --usage simu,"
        echo "puis renommez en data/*_simu_${WINDOW}.csv) et créez le config exp21."
        exit 1
    fi
done

echo "[sim] overfit MILP | PV=$PV | load=$LOAD -> $OUT"
# Env strictement imposé : micromamba stageCorse (seul env avec numpy/cvxpy).
# --cap-horizon : le MILP planifie le MÊME horizon que l'entraînement RL (max_steps = n_steps - horizon
#   = 3 j ici), pour que MILP et RL soient comparés sur les mêmes pas de décision (cf. plan, étape 2).
# --score-days 2 : fenêtre 4 j → 3 j déroulés ; coût scoré sur les jours 1-2 (le jour 3 tamponne le
#   dump SoC terminal du MILP, cf. mémoire sim-score-on-n-minus-1).
micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
    --config "$CONFIG" \
    --forecast "$PV" \
    --load-csv "$LOAD" \
    --out "$OUT" \
    --plan-out "$PLAN_OUT" \
    --score-days 2 \
    --cap-horizon \
    "$@"
