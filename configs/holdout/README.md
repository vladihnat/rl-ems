# Famille B — Évaluation propre *train/test* (held-out)

`exp31` : le modèle est **entraîné** sur une fenêtre longue (`train_split: 0.8`, best-model
sélectionné sur la **validation**) puis **rejoué sur une fenêtre de test DISJOINTE** (0 timestamp
commun, vérifié). C'est le seul protocole hors-overfit du dépôt.

**Axes des configs :**
- **Famille de reward** : `c1` = `sigma_charge_hold: 0.0`, `c2` = `sigma_charge_hold: 0.5`
  (base commune : `store_value_mode: milp_dual`, `sigma_hold_export: 0.5`, `sigma_store: 1.0`,
  `timing_features_v2: true`, `pv_charge_mode: surplus`, 400 kWh, `gamma: 0.999`, `norm_obs: true`).
- **Régime** : `hiver_haute`, `hiver_basse`, `ete_haute`, `ete_basse` (spécialistes, entraînés sur
  leur propre régime) + `combined` (modèle multi-saison, rejoué partout et en continu sur 16 j).
- `exp31_smoke.yaml` : run court de fumée (validation du pipeline).

**Données** : `data.pv_csv`/`load_csv` pointent la fenêtre de TRAIN (`data/train/…_<regime>.csv`) ;
au replay, `scripts/{rl,milp}/run_exp31.sh` surchargent avec la fenêtre de TEST
(`data/simu/…_<regime>_4d.csv` ou `…_combined.csv`) via `--forecast`/`--load-csv`.

**Scoring** : identique RL & MILP — rollout borné (`--cap-horizon`), coût sur N−1 jours
(`--score-days`) avec `boundary_soc_credit` (exclut le dump de SoC terminal du MILP).

Réf. : `README.md` §4-B · reproduction complète : [SIMULATION.md](../../SIMULATION.md).
