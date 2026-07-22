# SIMULATION.md — construire les données, écrire/lancer les `.sh`, lire les résultats

Guide **fonctionnel** (pas d'analyse) pour reproduire les simulations : de la donnée brute au gap
RL↔MILP agrégé. Trois étapes :

1. [Construire les données](#1-construire-les-données) (`raw/ → clean/ → train/ + simu/`)
2. [Écrire & lancer les scripts `.sh`](#2-écrire--lancer-les-scripts-de-replay-sh) de replay RL / MILP
3. [Agréger & re-scorer les résultats](#3-agréger--re-scorer-les-résultats)

Rappel : **tout Python passe par l'env micromamba `stageCorse`** (`micromamba run -n stageCorse …`).

---

## 1. Construire les données

### 1.1 Pipeline `raw/ → clean/`

Les fichiers pyranomètre bruts (`data/raw/Pyrano1{w,M,Y}.csv`) sont nettoyés vers `data/clean/` :

```bash
micromamba run -n stageCorse python data/clean_meteo.py        # → data/clean/Pyrano1{w,M,Y}_clean.csv
micromamba run -n stageCorse python data/clean_load_interpol.py # → data/clean/load_profile_interpol.csv (+ zoh)
```

Le nettoyage ajoute les paires **(cos, sin)** de l'heure et du jour de l'année (casse les
discontinuités du temps cyclique). `data/clean/Pyrano1Y_clean.csv` (**année complète**) est la
**source d'extraction** de toutes les fenêtres.

### 1.2 Découper des fenêtres — `extract_pyrano_simu.py`

`data/extract_pyrano_simu.py` extrait un bloc de jours contigus de `Pyrano1Y_clean.csv`, joint au
profil de charge au pas de 15 min, et écrit **deux fichiers alignés** (irradiance + charge).

| Flag | Rôle |
|------|------|
| `--usage {train,simu}` | destination : `train/` → entraînement, `simu/` → test/replay |
| `--window M S N` | fenêtre : mois `M`, jour de départ `S`, `N` jours. **Répétable** (concaténées, non-chevauchantes) |
| `--nbD N --month M --startDate S` | mode fenêtre unique (équivaut à un seul `--window`) |

**Sorties à noms FIXES** (le dossier parent est créé automatiquement) — **à renommer après chaque appel** :
- `--usage train` → `data/train/irradiance_training.csv` + `data/train/load_training.csv`
- `--usage simu`  → `data/simu/irradiance_simulation.csv` + `data/simu/load_simulation.csv`

### 1.3 Régimes (été/hiver × haute/basse saison)

Le régime de **prix** est déduit du **mois** des timestamps
(`envs/components/price_signal.py:_HAUTE_SAISON_MONTHS = (1,2,7,8,11,12)`, export cher). Les
métriques « saison » utilisent `SUMMER_MONTHS = {5..10}`. D'où les 4 régimes :

| Régime | Mois | Prix export |
|--------|------|-------------|
| `ete_haute`   | Juil / Août            | haut |
| `ete_basse`   | Mai / Juin / Sep       | bas  |
| `hiver_haute` | Nov / Déc / Jan / Fév  | haut |
| `hiver_basse` | Mars / Avril           | bas  |

### 1.4 Recettes exactes des jeux exp31 (held-out)

**Fenêtres de TEST — 4 jours** (`--usage simu`, 288 pas ; N−1 = 3 j scorables) → renommer en
`data/simu/{irradiance,load}_simu_<regime>_4d.csv` :

```bash
micromamba run -n stageCorse python data/extract_pyrano_simu.py --usage simu --window 7 1 4  # ete_haute  (Jul 1-4)
micromamba run -n stageCorse python data/extract_pyrano_simu.py --usage simu --window 1 1 4  # hiver_haute(Jan 1-4)
micromamba run -n stageCorse python data/extract_pyrano_simu.py --usage simu --window 6 5 4  # ete_basse  (Jun 5-8)
micromamba run -n stageCorse python data/extract_pyrano_simu.py --usage simu --window 3 6 4  # hiver_basse(Mar 6-9)
# après CHAQUE appel :
#   mv data/simu/irradiance_simulation.csv data/simu/irradiance_simu_<regime>_4d.csv
#   mv data/simu/load_simulation.csv       data/simu/load_simu_<regime>_4d.csv
```

**Fenêtres de TRAIN par régime** (longues, **DISJOINTES** du test du même régime) → renommer en
`data/train/{irradiance,load}_train_<regime>.csv` :

```bash
# ete_haute   : Jul 6-27 + Août
python data/extract_pyrano_simu.py --usage train --window 7 6 22 --window 8 1 28
# ete_basse   : Mai + Jun 10-28 + Sep
python data/extract_pyrano_simu.py --usage train --window 5 1 28 --window 6 10 19 --window 9 1 28
# hiver_haute : Jan 6-27 + Fév
python data/extract_pyrano_simu.py --usage train --window 1 6 22 --window 2 1 28
# hiver_basse : Mar 11-28 + Avr
python data/extract_pyrano_simu.py --usage train --window 3 11 18 --window 4 1 28
```

**Modèle combiné multi-saison :**

```bash
# TRAIN combiné (28 j, disjoint des 4 tests) → data/train/{irr,load}_train_combined.csv
python data/extract_pyrano_simu.py --usage train --window 1 6 7 --window 3 11 7 --window 6 10 7 --window 7 6 7
# TEST combiné continu (16 j, les 4 fenêtres test en chronologie) → data/simu/{irr,load}_simu_combined.csv
python data/extract_pyrano_simu.py --usage simu  --window 1 1 4 --window 3 6 4 --window 6 5 4 --window 7 1 4
```

> ⚠️ **Disjonction train/test** : les fenêtres train ci-dessus ne partagent aucun timestamp avec la
> fenêtre test du même régime (vérifié). C'est ce qui rend l'évaluation exp31 réellement held-out.

Les CSV `data/train/` et `data/simu/` sont **suivis par git** (contrairement au reste de `data/`,
gitignoré) : les simulations sont donc reproductibles depuis le dépôt sans re-extraire.

---

## 2. Écrire & lancer les scripts de replay (`.sh`)

Rejouer un modèle entraîné (ou l'optimum MILP) sur une fenêtre concrète se fait via
`scripts/rl/run_exp*.sh` (RL) et `scripts/milp/run_exp*.sh` (MILP), qui appellent respectivement
`monitoring/run_optimization_example.py` et `monitoring/run_milp_optimization_example.py`.

### 2.1 Anatomie d'un script (`scripts/rl/run_exp31.sh`)

```bash
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"   # racine du repo (scripts/rl/ → 2 niveaux)
CONFIG="results/${SWEEP}/${RUN}/config_used.yaml"          # config figé du run (fallback: configs/sweeps/)
MODEL="results/${SWEEP}/${RUN}/best_model.zip"             # best-model (sélectionné sur validation)
# données de TEST surchargées (le config pointe le TRAIN ; --forecast/--load-csv imposent le TEST) :
PV="data/simu/irradiance_simu_${CASE}_4d.csv"; LOAD="data/simu/load_simu_${CASE}_4d.csv"
micromamba run -n stageCorse python -m monitoring.run_optimization_example \
    --config "$CONFIG" --model "$MODEL" \
    --forecast "$PV" --load-csv "$LOAD" \
    --out "monitoring/runs/rl_${SWEEP}_${RUN}_on_${CASE}_monitoring_table.csv" \
    --score-days "$SCORE_DAYS" --cap-horizon \
    --vec-normalize "results/${SWEEP}/${RUN}/best_vecnormalize.pkl"
```

Flags clés de `run_optimization_example.py` :

| Flag | Rôle |
|------|------|
| `--forecast` / `--load-csv` | **surchargent** `data.pv_csv`/`load_csv` du config (impose la fenêtre de test) |
| `--measures` | vérité terrain vue pas-à-pas (défaut = `--forecast`, foresight parfait) |
| `--cap-horizon` | garde le cap d'horizon d'entraînement (le dernier jour, jamais joué, ne fausse pas le replay) |
| `--score-days N` | ne score le coût que sur les **N premiers jours** (N−1 exclut le dump de bord) |
| `--vec-normalize` | recharge la normalisation d'obs (`best_vecnormalize.pkl`) — obligatoire si `norm_obs:true` |
| `--split {full,test,train}` | rejoue tout le CSV (`full`, défaut ici) ou la tranche registry |
| `--no-show` | pas de fenêtres matplotlib (batch) |

### 2.2 ⚠️ Le MILP DOIT partager le même scoring

Le gap n'a de sens que si RL et MILP sont scorés **identiquement**. `scripts/milp/run_exp31.sh` est
le pendant exact : mêmes `--score-days` / `--cap-horizon`, mêmes fenêtres. Le MILP ne voit **ni le
reward shaping ni le split** — seulement les données + la physique — donc il réutilise
`configs/overfit/exp21_overfit_hiver_haute.yaml` comme **baseline physique partagée** (400 kWh /
`pv_charge_mode: surplus` / tarif HP-HC), le régime de prix étant déduit du mois. **Une** exécution
MILP par fenêtre suffit, réutilisable pour tous les modèles :

```bash
bash scripts/milp/run_exp31.sh hiver_haute          # → monitoring/runs/milp_exp31_hiver_haute_*.csv
bash scripts/rl/run_exp31.sh   exp31_c1_hiver_haute run_000   # même fenêtre, modèle RL
```

Scoring exp31 : fenêtre régime = 4 j → rollout 3 j (`--cap-horizon`) → `--score-days 2` ;
fenêtre combinée = 16 j → `--score-days 14`.

### 2.3 Créer un script pour une nouvelle fenêtre

1. Copier le plus proche (`cp scripts/rl/run_exp31.sh scripts/rl/run_exp32.sh`).
2. Ajuster `SWEEP`/`CASE` et les chemins `PV`/`LOAD` (→ `data/simu/…`).
3. Créer le **pendant MILP** avec les **mêmes** `--score-days`/`--cap-horizon`.
4. `bash -n scripts/rl/run_exp32.sh` (vérif syntaxe) puis lancer.

Sorties : `monitoring/runs/{rl,milp}_*_monitoring_table.csv` (dispatch pas-à-pas) et `*_plan.csv`
(plan MILP). `monitoring/runs/` est **gitignoré**.

---

## 3. Agréger & re-scorer les résultats

Les gaps d'un **sweep** (grille de seeds/hyperparamètres, cf. [cluster/README.md](./cluster/README.md))
sont dans `results/<sweep>/run_*/metrics.json`.

### 3.1 Agréger — `aggregate_results.py`

```bash
micromamba run -n stageCorse python cluster/sweep/aggregate_results.py exp31_c1_hiver_haute exp31_c2_hiver_haute
```

Résume par sweep le gap du **best-model** (sélectionné sur validation) et du modèle **final**,
moyenne ± σ sur les seeds. Colonnes importantes :
- `gap_adj` = `relative_gap_adjusted` : facture la **charge non servie** (phantom) au VOLL — le
  `relative_gap` brut n'est pas comparable entre politiques qui servent des quantités de charge
  différentes (une politique qui sert moins importe moins et paraît « moins chère »).
- `served` = ratio de charge servie (1.0 = tout servi). Toute run `served < 1` est **flaggée `!` et
  exclue** de la moyenne ajustée (faux « bon gap »).

### 3.2 Re-scorer sans réentraîner — `rescore_runs.py`

Le scoring est un **post-traitement** (`boundary_soc_credit`, fenêtre `score_days`). Les
`metrics.json` écrits pendant l'entraînement sont figés à l'ancien scoring. `rescore_runs.py`
relance, par run, `run_experiment.py --config <run>/config_used.yaml --rescore` : recharge le
modèle sauvegardé, régénère `metrics.json`/`comparison.json` **sans toucher aux `.zip`/`.pkl`/`.npz`**
(la sélection best-model reste intacte).

```bash
micromamba run -n stageCorse python cluster/sweep/rescore_runs.py exp31_c1_hiver_haute
micromamba run -n stageCorse python cluster/sweep/aggregate_results.py exp31_c1_hiver_haute  # gaps corrigés
```

### 3.3 Quel gap regarder ?

- **`gap_abs_eur` (coût_RL − coût_MILP, €)** = métrique **primaire**, robuste.
- **`relative_gap` (%)** = secondaire, **non fiable en basse saison** où `|coût_MILP| ≈ 0`
  (dénominateur → gaps 49–237 % artefactuels). En held-out, préférer les chiffres du rapport /
  du replay, jamais les `metrics.json` internes en basse saison.
- `boundary_soc_credit` : crédite/débite symétriquement RL et MILP de la valeur du SoC de bord, pour
  que le scoring sur N−1 jours n'avantage personne (sinon le dump de SoC terminal du MILP biaise le gap).

---

## Référence rapide

| Besoin | Commande |
|--------|----------|
| Extraire une fenêtre test 4 j | `python data/extract_pyrano_simu.py --usage simu --window <M> <S> 4` |
| Replay RL sur un régime | `bash scripts/rl/run_exp31.sh exp31_c1_<regime> run_000` |
| Replay MILP (baseline) sur un régime | `bash scripts/milp/run_exp31.sh <regime>` |
| Tous les régimes d'un coup | `bash scripts/{rl,milp}/run_exp31.sh … all` |
| Agréger les gaps d'un sweep | `python cluster/sweep/aggregate_results.py <sweep>` |
| Re-scorer un sweep | `python cluster/sweep/rescore_runs.py <sweep>` |

Voir aussi : **[README.md](./README.md)** (structure & familles) · **[cluster/README.md](./cluster/README.md)**
(sweeps HPC) · **[RL_communication-flow.md](./RL_communication-flow.md)** (flux interne env ↔ agent).
