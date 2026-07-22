# Cluster HPC — Orsu (UDCPP)

## Connexion & commandes SLURM

| Action | Commande |
|--------|----------|
| Connexion | `ssh HERRERA-NATIVI_V@193.48.30.217` |
| État des nœuds | `sinfo` |
| Mes jobs | `squeue -u $USER` |
| Annuler un job | `scancel <JOB_ID>` |
| Annuler un array entier | `scancel <JOB_ID>_*` |
| Shell interactif | `srun -t 30:00 -p intel --nodes=1 --ntasks=4 --pty bash -i` |
| Modules | `module avail` / `module load anaconda` / `module purge` |

**Partitions disponibles :** `intel` (26c×2), `amd` (48c×2), `gpu`, `brando_cpu`  
**Stockage rapide :** `/scratch/HERRERA-NATIVI_V/` (BeeGFS, 6 GB/s) — utiliser ici pour les runs

---

## Configuration (à modifier une fois)

En tête de `cluster/transfer/push.sh`, `pull_results.sh`, `submit.sh` et `sweep/launch_sweep.sh` :

```bash
REMOTE_HOST=HERRERA-NATIVI_V@193.48.30.217
REMOTE_PATH=/scratch/HERRERA-NATIVI_V/microgrid-rl
```

Ou exporter via l'env avant chaque commande :

```bash
export REMOTE_HOST=HERRERA-NATIVI_V@193.48.30.217
export REMOTE_PATH=/scratch/HERRERA-NATIVI_V/microgrid-rl
```

---

## Axe 2 — Sweep d'hyperparamètres (usage prioritaire)

Le cluster lance N expériences SAC **en parallèle** via un job array SLURM.  
Chaque tâche de l'array correspond à une combinaison (hyperparamètres × seed).

### Workflow complet

```bash
# 1. Générer la grille (produit cartésien params × seeds)
python cluster/sweep/generate_sweep.py \
    --base-config configs/overfit/exp28_hiver_haute.yaml \
    --sweep-name batchS_lr_sweep \
    --params batch_size=256,512 lr=0.0003,0.001 \
    --seeds 42,43,44
# → 12 configs dans configs/sweeps/gamma_lr_sweep/run_*.yaml
# → manifest dans cluster/sweep/manifests/gamma_lr_sweep.json

# 2. Push + soumettre 12 jobs en parallèle (au plus 8 simultanés via le throttle)
./cluster/sweep/launch_sweep.sh gamma_lr_sweep --throttle 8
# → Submitted batch job 12345  (array 0-11%8)

# 3. Monitorer depuis local
ssh HERRERA-NATIVI_V@193.48.30.217 "squeue -u $USER"

# 4. Rapatrier quand terminé
./cluster/transfer/pull_results.sh gamma_lr_sweep
# → résultats dans results/gamma_lr_sweep/run_000/ ... run_011/
```

### Paramètres disponibles pour le sweep

Alignés sur `PARAM_MAP` dans `cluster/sweep/generate_sweep.py` (clé CLI → chemin YAML). Regroupés par thème ci-dessous ; toute combinaison est mélangeable dans un même sweep.

**Cœur HP & normalisation**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `gamma` | `training.gamma` | `0.99,0.997,0.999` |
| `lr` | `training.learning_rate` | `0.0003,0.0001` |
| `batch_size` | `training.batch_size` | `256,512` |
| `buffer_size` | `training.buffer_size` | `300000,750000` |
| `sigma_soc` | `reward.sigma_soc` | `5.0,10.0,20.0` |
| `net_arch` | `training.net_arch` | `256x256,400x400x300` |
| `ent_coef` | `training.ent_coef` | `auto,0.05` |
| `tau` | `training.tau` | `0.005,0.01` |
| `train_freq` | `training.train_freq` | `1,4` |
| `gradient_steps` | `training.gradient_steps` | `1,2` |
| `total_timesteps` | `training.total_timesteps` | `1000000,2000000` |
| `norm_obs` | `training.norm_obs` | `true,false` |
| `norm_reward` | `training.norm_reward` | `true,false` |
| `seed` | `experiment.seed` | via `--seeds` |

**Leviers comportementaux (exp14+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `learning_starts` | `training.learning_starts` | `10000` |
| `random_soc` | `training.random_soc` | `true,false` |
| `warm_start_milp` | `training.warm_start_milp` | `true,false` |

**Behavior Cloning depuis le MILP (exp17+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `bc_pretrain_epochs` | `training.bc_pretrain_epochs` | `0,20` |
| `bc_pretrain_lr` | `training.bc_pretrain_lr` | `0.0003` |
| `milp_window_days` | `training.milp_window_days` | `1,2` |

**Ancre MILP persistante — WS-1c (exp18+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `bc_anchor_demo_buffer` | `training.bc_anchor_demo_buffer` | `true,false` |
| `bc_anchor_demo_frac` | `training.bc_anchor_demo_frac` | `0.25` |
| `bc_anchor_loss` | `training.bc_anchor_loss` | `true,false` |
| `bc_anchor_lambda` | `training.bc_anchor_lambda` | `1.0` |
| `bc_anchor_lambda_final` | `training.bc_anchor_lambda_final` | `0.1` |
| `bc_anchor_loss_batch` | `training.bc_anchor_loss_batch` | `256` |

**AWAC — BC pondéré par l'avantage critique (exp20+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `bc_anchor_awac` | `training.bc_anchor_awac` | `true,false` |
| `bc_anchor_awac_beta` | `training.bc_anchor_awac_beta` | `1.0` |
| `bc_anchor_awac_wmax` | `training.bc_anchor_awac_wmax` | `20.0` |

**Reward shaping PBRS (exp19+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `store_value` | `reward.store_value` | `true,false` |
| `sigma_store` | `reward.sigma_store` | `0.5,1.0` |
| `sigma_export_stock` | `reward.sigma_export_stock` | `0.0,1.0` |

**Overfit / scoring (exp21+) & feature de timing (exp23+)**

| Param CLI | Clé YAML | Exemple |
|-----------|----------|---------|
| `eval_on_train` | `training.eval_on_train` | `true,false` |
| `score_days` | `training.score_days` | `1,2` |
| `timing_feature` | `observation.timing_feature` | `true,false` |

**Encodage des valeurs :**
- `net_arch` : couches séparées par `x` (pas de virgule interne) → `256x256` → `[256, 256]`, `400x400x300` → `[400, 400, 300]`.
- Booléens : `true` / `false` (ex. `norm_obs=true,false`).
- Une valeur unique (ex. `total_timesteps=1000000`) **fixe** le paramètre sans multiplier la grille.

### Throttle — plafond de concurrence (`%K`)

`launch_sweep.sh` accepte `--throttle K` (défaut **8**), qui soumet `sbatch --array=0-N%K`.

**Mécanisme.** Sans plafond, SLURM met en exécution toutes les N tâches éligibles dès que des ressources
se libèrent. `%K` borne à **au plus K tâches en RUN simultanément** ; les autres restent PENDING et
démarrent au fil de l'eau. `--throttle 0` désactive le plafond.

**Pourquoi l'utiliser.** Courtoisie sur un cluster partagé : empreinte cœurs/RAM bornée, pas d'inondation
de la file, meilleure fair-share. *(Rappel Orsu : 28 nœuds Intel = 1456 cœurs ; un sweep 16×(4 c) = 64 c
tient même sans cap, mais le throttle reste la bonne pratique — et devient indispensable pour de plus gros sweeps.)*

**Effet de K** — temps mur ≈ ⌈N/K⌉ × durée_run (ex. 16 runs × ~3,4 h à 1M steps) :

| `--throttle` | Tâches simultanées | Vagues (16 runs) | Temps mur | Quand |
|--------------|--------------------|------------------|-----------|-------|
| `8` (défaut) | 8 | ~2 | ~7 h | bon compromis débit / courtoisie |
| `6` | 6 | ~3 | ~10 h | cluster moyennement chargé |
| `4` | 4 | ~4 | ~14 h | empreinte minimale, cluster très chargé |
| `0` | toutes | 1 | ~3,4 h | petits sweeps / cluster vide |

### Re-soumettre sans re-uploader le code

```bash
./cluster/sweep/launch_sweep.sh gamma_lr_sweep --no-push
# --no-push et --throttle K sont combinables :
./cluster/sweep/launch_sweep.sh gamma_lr_sweep --no-push --throttle 6
```

### Ajouter un nouvel experiment au sweep

1. Créer (ou copier) un config dans la famille adéquate (`configs/overfit/` ou `configs/holdout/`) :
   ```bash
   cp configs/overfit/exp28_hiver_haute.yaml configs/overfit/exp32_new.yaml
   # éditer exp32_new.yaml
   ```
2. Relancer `generate_sweep.py` avec `--base-config configs/overfit/exp32_new.yaml`
3. `./cluster/sweep/launch_sweep.sh <sweep_name>`

### Re-scoring sans réentraîner

Le scoring est un **post-traitement** (cf. crédit de SoC de bord, `evaluation.metrics.boundary_soc_credit`) : les `metrics.json`/`comparison.json` écrits pendant l'entraînement sont figés à l'ancien scoring. `rescore_runs.py` relance, pour chaque run, `run_experiment.py --config <run>/config_used.yaml --rescore` qui recharge le modèle déjà sauvegardé et régénère ces JSON avec le scoring courant — **les `.zip`/`.pkl`/`.npz` (donc la sélection best-model) restent intacts**.

```bash
# Régénère metrics.json/comparison.json depuis les modèles sauvegardés (sans réentraîner)
python cluster/sweep/rescore_runs.py exp22_hiver_haute exp22_ete_haute
# Agrège les gaps (corrigés) par sweep
python cluster/sweep/aggregate_results.py exp22_hiver_haute
```

---

## Expérience unique (fallback)

```bash
./cluster/submit.sh exp07_randSoC
# → push + sbatch train_single.slurm
```

---

## Structure des fichiers cluster/

```
cluster/
├── README.md                           (ce fichier)
├── jobs/
│   ├── train_array.slurm              # job array — sweep parallèle
│   └── train_single.slurm             # job unique — test/fallback
├── sweep/
│   ├── generate_sweep.py              # générateur de grille
│   ├── launch_sweep.sh                # push + sbatch array
│   ├── aggregate_results.py           # agrégation des gaps par sweep
│   ├── rescore_runs.py                # re-scoring post-traitement (sans réentraîner)
│   └── manifests/                     # JSONs des familles gardées (à plat, résolus par launch_sweep.sh)
│       └── _archive/                  #   manifests des anciennes expériences
├── transfer/
│   ├── push.sh                        # local → cluster
│   └── pull_results.sh               # cluster → local
└── submit.sh                          # one-liner single exp
```
