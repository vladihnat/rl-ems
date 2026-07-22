# Famille A — Validation *overfit*

Protocole commun : `eval_on_train: true`, `train == val == test` sur une **fenêtre fixe** de 4 jours
d'un régime (`make_env_overfit`). On ne teste **pas** la généralisation mais la **capacité de la
formulation RL à atteindre l'optimum MILP**. Chaque expérience isole *une implémentation* :

| Config | Vérifie |
|--------|---------|
| `exp21_overfit_{hiver,ete}_haute.yaml` | Sanity overfit + **fix truncation terminale** (`truncated` ≠ `terminated` → bootstrap `V(s')`). |
| `exp22_{hiver,ete}_haute.yaml` | **Prix fixes** (isole auto-conso vs timing) + **`boundary_soc_credit`** (corrige l'artefact de fenêtre de score). |
| `exp26_{hiver,ete}_haute.yaml` | PBRS **`milp_dual`** : Φ = λ(t), dual du LP à binaires figés (invariant sur l'optimum). |
| `exp28_{hiver,ete}_haute.yaml` | Miroir **EXPORT** `r_hold_export` (anti « exporter le stock trop tôt »). |
| `exp29_{hiver,ete}_haute.yaml` | Miroir **CHARGE** `r_charge_hold` (anti « charger trop tôt » le matin). |
| `exp30_{hiver,ete}_haute.yaml` | Miroir **SERVE** `r_discharge_hold` + `charge_hold_deadline` (pauses du soir). |

Base physique partagée (héritée par la baseline MILP de `scripts/milp/run_exp*.sh`) : batterie
400 kWh, `pv_charge_mode: surplus`, tarif HP/HC, régime déduit du mois. Le MILP réutilise
`exp21_overfit_*.yaml` comme baseline (mêmes paramètres physiques, seul le reward shaping — invisible
au MILP — change entre exp26→30).

Réf. : `README.md` §4-A · replay : `scripts/{rl,milp}/run_exp{21_overfit,22,26,28,29,30}.sh`
(cf. [SIMULATION.md](../../SIMULATION.md)).
