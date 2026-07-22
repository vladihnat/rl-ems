# EMS-RL — Energy Management System par Reinforcement Learning

> Remplacement d'un optimiseur MILP/LP par un agent RL (SAC) pour la gestion de batterie
> dans un microréseau PV (contexte EDF / réseau Corse : contrainte *no-grid-charging*,
> tarif HP/HC avec export saisonnier).

Le fil rouge du projet est la **réduction de l'écart (« gap ») entre l'agent RL et l'optimum
MILP**. Les expériences sont organisées en **deux familles** (cf. §4) : la validation *overfit*
de la formulation, puis l'évaluation propre *train/test* (held-out).

---

## 1. Structure du projet

```
.
├── README.md                     ← ce fichier (structure + synthèse)
├── SIMULATION.md                 ← guide : données → scripts .sh → aggregate results
├── RL_communication-flow.md      ← flux interne env ↔ agent (détail modules)
├── prompts.md                    ← journal des prompts (cf. §8 « VIBE CODING »)
├── requirements.txt
│
├── envs/                         # environnement Gymnasium custom (remplace pymgrid)
│   ├── base_microgrid_env.py
│   ├── registry.py               #   make_env (split) / make_env_overfit
│   └── components/               #   battery.py, load.py, pv_source.py, price_signal.py
├── agents/                       # algos RL + Behavior Cloning
│   ├── sac_agent.py              #   SAC (principal) ; ddpg/td3/ppo aussi
│   ├── bc_sac.py                 #   Behavior Cloning + ancre MILP / AWAC
│   └── common.py                 #   callbacks d'éval, sélection best-model
├── baselines/
│   ├── milp_solver.py            #   optimum MILP (CVXPY + HiGHS), binaire complémentarité import/export
│   └── milp_dispatch.py          #   shim env pour rejouer le plan MILP
├── evaluation/
│   ├── metrics.py                #   métriques + gap saisonnier (gap_abs_eur) + boundary_soc_credit
│   └── compare.py
├── experiments/
│   └── run_experiment.py         # POINT D'ENTRÉE unique (entraînement / --rescore)
├── monitoring/                   # replay + plots (sorties → monitoring/runs/, gitignoré)
│   ├── run_optimization_example.py      #   replay RL
│   └── run_milp_optimization_example.py #   replay MILP
│
├── configs/                      # configs YAML, RANGÉES PAR FAMILLE
│   ├── overfit/                  #   validation overfit (exp21, 22, 26, 28, 29, 30) — cf. configs/overfit/README.md
│   ├── holdout/                  #   train/test propre (exp31 : c1/c2 × 4 régimes + combiné + smoke)
│   ├── _archive/                 #   anciennes expériences (exp01→exp20, exp23/24/25/27, …)
│   └── sweeps/                   #   configs générés par run de sweep (gitignoré)
├── scripts/                      # scripts bash de replay/simulation (cf. SIMULATION.md)
│   ├── rl/    run_exp{21_overfit,22,26,28,29,30,31}.sh   + _archive/
│   └── milp/  run_exp{21_overfit,22,26,28,29,30,31}.sh   + _archive/
├── cluster/                      # infra de sweep HPC SLURM (cf. cluster/README.md)
│   └── sweep/manifests/          #   manifests des familles gardées (à plat) + _archive/
├── data/                         # données (cf. SIMULATION.md §1)
│   ├── raw/                      #   pyranomètre brut (gitignoré)
│   ├── clean/                    #   Pyrano1{w,M,Y}_clean.csv + load_profile_* — SOURCE d'extraction
│   ├── train/                    #   fenêtres d'entraînement (suivies)
│   ├── simu/                     #   fenêtres de test/replay (suivies)
│   └── extract_pyrano_simu.py    #   + clean_meteo.py, clean_load*.py (pipeline)
└── results/                      # artefacts par expérience (gitignoré, local)
```

---

## 2. Environnement & installation

Python via **micromamba**, environnement `stageCorse` (le seul autorisé à exécuter le code) :

```bash
micromamba run -n stageCorse python -c "import stable_baselines3, cvxpy, highspy; print('ok')"
# dépendances : cf. requirements.txt (stable-baselines3, gymnasium, cvxpy+HiGHS, pandas, matplotlib…)
```

---

## 3. Utilisation

Le point d'entrée unique est **`experiments/run_experiment.py`**. Tout le paramétrage (algo RL,
hyperparamètres, composants physiques, données, horizon, prix, contraintes SoC, reward shaping)
est piloté par les **fichiers YAML** de `configs/`.

```bash
# Entraîner (dépose modèle + métriques + courbes dans results/<nom>/)
micromamba run -n stageCorse python experiments/run_experiment.py --config configs/holdout/exp31_c1_hiver_haute.yaml

# Re-scorer un run SANS réentraîner (recharge le modèle, régénère metrics.json/comparison.json)
micromamba run -n stageCorse python experiments/run_experiment.py --config results/<nom>/config_used.yaml --rescore
```

Pour **rejouer** un modèle entraîné ou l'optimum MILP sur une fenêtre concrète, et pour la
construction des données, voir **[SIMULATION.md](./SIMULATION.md)**. Pour les **sweeps
d'hyperparamètres sur cluster HPC**, voir **[cluster/README.md](./cluster/README.md)**.

---

## 4. Les deux familles d'expériences

Le tri de fin de stage ne conserve que **deux familles** ; l'historique exploratoire (exp01→exp20
et intermédiaires) est déplacé dans les dossiers `_archive/`.

### Famille A — Validation *overfit* du modèle  (`configs/overfit/`)

Protocole : `eval_on_train:true`, `train == val == test` sur une **fenêtre fixe** (4 jours d'un
régime). Question posée : **la formulation RL est-elle capable d'atteindre l'optimum MILP** quand la
généralisation n'est plus en jeu ? Chaque expérience valide *une implémentation précise* :

| Exp | Ce qu'elle vérifie |
|-----|--------------------|
| **exp21** | Sanity-check overfit (`make_env_overfit`) **+ fix de truncation terminale** : renvoyer `truncated` (et non `terminated`) en bord de fenêtre pour que le critic bootstrappe `V(s')` → supprime la myopie de fin d'horizon. |
| **exp22** | **Prix fixes** pour isoler l'auto-consommation du timing **+ `boundary_soc_credit`** : crédit de SoC de bord symétrique RL/MILP corrigeant l'artefact de fenêtre de score (gap « négatif » où le RL « battait » le MILP). |
| **exp26** | Reward shaping **PBRS `milp_dual`** : potentiel Φ = λ(t), le dual du LP à binaires figés (invariant sur l'optimum). |
| **exp28** | Terme **miroir EXPORT `r_hold_export`** : pénalise l'export du stock *avant* le pic d'export futur (anti « déployer trop tôt »). |
| **exp29** | Terme **miroir CHARGE `r_charge_hold`** : pénalise la charge trop précoce le matin (hors fenêtre d'export minimal). |
| **exp30** | Terme **miroir SERVE `r_discharge_hold` + `charge_hold_deadline`** : gère les pauses du soir (borne le refill au prochain pic export). |

Les termes exp26→exp30 sont des **potentiels invariants** (PBRS) : ils ne déplacent pas l'optimum,
seulement la vitesse/qualité de convergence. Résultat : sur fenêtre fixe le RL approche l'optimum
MILP, le *dump* de batterie à t0 disparaît, et le résiduel « déployer trop tôt » se décompose
proprement (matin : charger trop tôt ; soir : décharger trop tôt).

### Famille B — Évaluation propre *train/test* (held-out)  (`configs/holdout/`)

`exp31` sort de l'overfit : un modèle est **entraîné** sur une fenêtre longue (`train_split:0.8`,
best-model sélectionné sur la tranche de **validation**) puis **rejoué sur une fenêtre de test
DISJOINTE** (0 timestamp commun, vérifié). Deux familles de reward (**C1** = `sigma_charge_hold:0`,
**C2** = `sigma_charge_hold:0.5`), 4 régimes (été/hiver × haute/basse saison) + un modèle **combiné**
multi-saison rejoué en continu sur 16 jours.

Résultats held-out (gap € = coût_RL − coût_MILP sur les jours scorés, N−1) :

| Régime de test | Gap réel | Note |
|----------------|----------|------|
| été (haute/basse) | **1 – 5 %** | régime le mieux résolu (PV ≫ charge, timing d'ordre 2) |
| hiver_haute | **~18 – 24 %** | **vrai** gap (fenêtre net-**revenu**, dénominateur sain) — rate le pic d'export du soir |
| combiné continu 16 j | **~9 – 10 %** | SoC porté à travers les transitions, MILP `optimal` sur 1440 pas |

Une seule cause, quatre visages : le RL **charge/décharge 1–4 h trop tôt** et sous-valorise le
maintien du stock tard dans la journée. Le shaping exp28→30 réduit clairement ce biais (été
quasi-optimal), mais un résiduel du même signe survit hors-overfit, maximal en hiver_haute où une
heure du soir mal placée coûte le plus. *(Les chiffres détaillés et l'analyse comportementale
complète figurent dans le rapport final ; la reproduction est décrite dans [SIMULATION.md](./SIMULATION.md).)*

⚠️ **Ne jamais citer** les gaps internes de `metrics.json` en basse saison (dénominateur ≈ 0 → 49–237 %
artefactuels) : utiliser le **gap absolu €** ou les chiffres held-out ci-dessus.

---

## 5. Architecture RL & baseline MILP

L'agent est un **SAC (Soft Actor-Critic)** (Stable-Baselines3) sur un environnement Gymnasium custom
(`MicrogridEnv`) encapsulant `PVSource`, `LoadModel`, `BatteryModel`, `PriceSignal`. À chaque pas :
- **observation** : SoC, charge, irradiance, features temporelles (sin/cos), prévisions PV **et
  charge** sur l'horizon, prévision de prix d'export, et une feature optionnelle de timing « gap au
  pic » (`observation.timing_feature`) ;
- **action** continue `[-1, 1]` : commande de puissance batterie normalisée (charge / décharge) ;
- **récompense** : coût réseau négatif + revenu d'export, pénalité de violation SoC, plus les termes
  optionnels de **reward shaping** (PBRS `store_value`, miroirs hold/charge/discharge — cf. §4).

**Contrainte EDF *no-grid-charging*** : la batterie ne se charge que du **surplus PV**
(`Pb_charge ≤ max(0, PV − charge)`), câblée à l'identique côté MILP (`baselines/milp_solver.py`) et RL.
Un flag `battery.pv_charge_mode` bascule `surplus` (défaut) ↔ `total` (variante Duchaud-JL relâchée).

**Baseline MILP** (CVXPY + HiGHS) : intègre un **binaire de complémentarité import/export**
garantissant une formulation propre (pas d'aller-retour import↔export fictif). Le plan est rejoué
dans l'env via `baselines/milp_dispatch.py` pour une comparaison à physique identique.

**Env custom plutôt que pymgrid** : `envs/base_microgrid_env.py` réimplémente les fonctionnalités
nécessaires de pymgrid tout en restant **totalement modifiable** (équations batterie, pertes, prix,
observation, récompense). Le flux interne complet est documenté dans
**[RL_communication-flow.md](./RL_communication-flow.md)**.

---

## 6. Données

Pipeline : `data/raw/` (pyranomètre brut) → `data/clean/` (via `data/clean_meteo.py` +
`data/clean_load*.py`, `Pyrano1Y_clean.csv` = année complète) → fenêtres découpées par
`data/extract_pyrano_simu.py` vers `data/train/` (entraînement) et `data/simu/` (test/replay).
La démarche complète (recettes de fenêtres par régime, régimes haute/basse) est dans
**[SIMULATION.md](./SIMULATION.md)**.

### ⚠️ Problème ouvert — encodage cyclique sin/cos
Le nettoyage ajoute des paires (cos, sin) pour l'heure et le jour de l'année (casse les
discontinuités du temps cyclique). En déploiement **LIVE**, rien ne garantit l'accès à ces paires
pré-calculées. À tester expérimentalement dans les 4 configurations (avec/sans heures brutes
normalisées × avec/sans sin/cos) pour mesurer la sensibilité réelle de l'agent.

---

## 7. État actuel & perspectives

### ✅ Implémenté
- Agents **SAC** (principal), **DDPG / TD3 / PPO** (SB3) ; env **Gymnasium custom** modifiable
- Baseline **MILP** (CVXPY + HiGHS) + replay, contrainte EDF *no-grid-charging* câblée MILP+RL
- **Behavior Cloning** + ancre MILP / AWAC ; **reward shaping PBRS** (store-value + miroirs hold/charge/discharge)
- Métriques de **gap stratifié par saison** (`gap_abs_eur`), `boundary_soc_credit`, **re-scoring** sans réentraînement
- Pipeline de données + jeux de simulation par régime (été/hiver × haute/basse) + fenêtre combinée
- **Validation overfit** (famille A) et **évaluation held-out train/test** (famille B, exp31)
- **Infra de sweep HPC** (`cluster/`) + scripts de replay centralisés (`scripts/`)

### 🔜 À venir
- Réduire le résiduel « déployer trop tôt » hors-overfit (surtout hiver_haute)
- **Rendements variables** (dépendant du SoC/de la puissance) et **coût par cycle** (vieillissement)
- **Étude de robustesse** de l'env custom vs pymgrid
- Test de la **feature sin/cos** en conditions LIVE (cf. §6)
- Génération des **figures held-out** pour le rapport (`scripts/fig/`, à adapter d'exp22 vers exp31)

---

## 8. VIBE CODING ⚠️

Le code a été **généré par Claude Code (Anthropic)** puis **relu et vérifié à la main** : aucune
ligne acceptée sans relecture. Les prompts (itérations, contraintes architecturales, vérifications
croisées) sont dans **[prompts.md](./prompts.md)** pour reproductibilité et inspection.
