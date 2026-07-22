#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Référence MILP des fenêtres de TEST exp31, pendant de scripts/rl/run_exp31.sh.
#
# Le MILP ne voit NI le reward shaping NI le split train/test : il ne dépend que des données de
# la fenêtre + des paramètres physiques (batterie 400 kWh / pv_charge_mode surplus / step_hp_hc /
# 100 kW / 500 kW). On réutilise donc configs/overfit/exp21_overfit_hiver_haute.yaml comme baseline
# partagée (params physiques identiques à exp28..exp31, vérifié) et on surcharge les données via
# --forecast/--load-csv. Le régime de prix haute/basse est déduit AUTOMATIQUEMENT du mois des
# timestamps (envs/components/price_signal.py:_HAUTE_SAISON_MONTHS), donc un seul config suffit
# pour les 4 régimes.
#
# => Le résultat ne dépend ni de la famille (c1/c2) ni du run : UNE exécution par fenêtre de test
#    suffit, réutilisable pour tous les modèles.
#
# Usage : ./scripts/milp/run_exp31.sh [TEST_CASE] [extra python args...]
#   TEST_CASE : hiver_haute (défaut) | hiver_basse | ete_haute | ete_basse | combined | all
#
# Scoring — DOIT être identique au RL (sinon le gap est biaisé) :
#   - fenêtre régime  : 4 j -> rollout 3 j (--cap-horizon) -> --score-days 2
#   - fenêtre combinée: 16 j -> rollout 15 j               -> --score-days 14

TEST_CASE="${1:-${CASE:-hiver_haute}}"
if [ "$#" -gt 0 ]; then shift; fi

CONFIG="configs/overfit/exp21_overfit_hiver_haute.yaml"   # baseline physique partagée (données surchargées)
[ -f "$CONFIG" ] || { echo "Introuvable : $ROOT/$CONFIG"; exit 1; }

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

    OUT="monitoring/runs/milp_exp31_${CASE}_monitoring_table.csv"
    PLAN_OUT="monitoring/runs/milp_exp31_${CASE}_plan.csv"
    echo "[sim] exp31 MILP | TEST=${CASE} | score_days=${SCORE_DAYS} -> $OUT"
    micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example \
        --config "$CONFIG" \
        --forecast "$PV" \
        --load-csv "$LOAD" \
        --out "$OUT" \
        --plan-out "$PLAN_OUT" \
        --score-days "$SCORE_DAYS" \
        --cap-horizon \
        "$@"
done
