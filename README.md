# EMS-RL — Energy Management System par Reinforcement Learning

> Remplacement d'un optimiseur MILP/LP par un agent RL pour la gestion de batterie dans un microréseau PV.

---

## 1. Structure du projet

```
.
├── RL_communication-flow.md
├── README.md
├── prompts.md
├── requirements.txt
├── agents/
│   └── sac_agent.py
├── baselines/
│   └── milp_solver.py
├── configs/
├── data/
│   ├── clean_meteo.py
│   └── Pyranometer data* # Raw and cleaned 
├── envs/
│   ├── base_microgrid_env.py
│   ├── registry.py
│   └── components/
│       ├── battery.py
│       ├── load.py
│       └── pv_source.py
├── monitoring/
│   ├── monitoring_table.py
│   ├── plot_monitoring_milp.py
│   ├── plot_monitoring.py
│   ├── plot_power.py
│   ├── run_milp_optimization_example.py
│   ├── run_optimization_example.py
│   └── runs/ # CSV of plannings 
├── evaluation/
│   ├── compare.py
│   └── metrics.py
├── experiments/
│   └── run_experiment.py
├── scripts/ # bash scripts for optimizations
│   ├── rl/
│   └── milp/
└── results/ # Only 1st experiment 
```

---

## 2. Utilisation

Le point d'entrée principal est **`experiments/run_experiment.py`**. Il a été conçu pour être facilement généralisable : la totalité du paramétrage (algorithme RL, hyperparamètres, composants physiques, données, horizon, prix, contraintes SoC, etc.) est pilotée par les fichiers de configuration YAML du dossier `configs/`.

Exemple d'appel typique :

```bash
python experiments/run_experiment.py --config configs/exp01_perfect_foresight.yaml
```

Pour lancer une nouvelle expérience, il suffit de dupliquer un fichier YAML existant, d'ajuster les champs souhaités, et de relancer le script avec le nouveau chemin de configuration. Les artefacts (modèle entraîné, métriques, courbes d'apprentissage, copie de la config utilisée) sont déposés automatiquement dans `results/<nom_experience>/`.

### Scripts d'optimisation (`scripts/`)

Pour simplifier l'utilisation des exemples d'optimisation (MILP et RL) et éviter d'avoir à retenir la combinaison config / modèle / chemin de sortie pour chaque expérience, des scripts bash dédiés ont été centralisés dans `scripts/rl/` et `scripts/milp/`. Chaque script fixe les bons chemins pour une expérience donnée et délègue à `monitoring/run_optimization_example.py` (RL) ou `monitoring/run_milp_optimization_example.py` (MILP).

Exemples d'utilisation :

```bash
# Évaluation de l'agent RL entraîné pour exp02 (prix variables)
bash scripts/rl/run_exp02.sh

# Optimisation MILP de référence sur exp02
bash scripts/milp/run_exp02.sh
```

Les arguments supplémentaires sont transmis tels quels au script Python sous-jacent (`"$@"`), ce qui permet de surcharger ponctuellement un paramètre sans modifier le script.

---

## 3. Données

L'implémentation actuelle utilise les données **`Pyrano1w_clean.csv`**. Le nettoyage et le prétraitement sont centralisés dans le dossier `data/` (voir notamment `data/clean_meteo.py`), à partir des fichiers bruts `Pyrano1w.csv`, `Pyrano1M.csv` et `Pyrano1Y.csv`.

### ⚠️ Problème ouvert d'implémentation — à tester

Lors du nettoyage, des **paires (cos, sin)** ont été ajoutées pour les heures de la journée et le jour de l'année, afin de casser les discontinuités artificielles du temps cyclique (par exemple, l'heure 23 et l'heure 0 sont numériquement éloignées alors qu'elles sont temporellement adjacentes).

Cependant, lors d'un déploiement en conditions réelles (**LIVE**), il n'est pas garanti que le modèle ait accès à ces paires sin/cos pré-calculées : seules les variables temporelles brutes seront probablement disponibles. Il faudra donc tester expérimentalement le comportement de l'agent dans les **4 configurations** :

|                      | Avec heures brutes normalisées | Sans heures brutes normalisées |
|----------------------|:------------------------------:|:------------------------------:|
| **Avec sin/cos**     | config A                       | config B                       |
| **Sans sin/cos**     | config C                       | config D                       |

L'objectif est d'identifier si l'agent est réellement sensible à l'encodage cyclique, et si oui de prévoir un pipeline temps-réel capable de fournir ces features.

---

## 4. Architecture RL

L'agent est un **SAC (Soft Actor-Critic)** implémenté via Stable-Baselines3. Il interagit avec un environnement Gymnasium custom (`MicrogridEnv`) qui encapsule trois composants physiques — `PVSource`, `LoadModel`, `BatteryModel` — et expose à chaque pas de temps :
- une **observation** : SoC courant, charge, irradiance, features temporelles, prévisions PV sur l'horizon ;
- une **action** continue dans `[-1, 1]` : la commande de puissance batterie normalisée (signe : charge / décharge) ;
- une **récompense** : coût négatif d'achat réseau + revenu de vente du surplus, pénalisée si les bornes SoC sont violées.

Le détail complet de la hiérarchie des modules, des équations internes et du flux de communication est documenté dans **[RL_communication-flow.md](./RL_communication-flow.md)**.

### Environnement custom à la place de pymgrid

Initialement, le projet prévoyait de s'appuyer sur **pymgrid** pour la simulation du microréseau. Cette piste a finalement été abandonnée : à la place, `envs/base_microgrid_env.py` implémente un **environnement Gymnasium entièrement custom**, qui reproduit les fonctionnalités nécessaires de pymgrid tout en étant **totalement modifiable** (équations batterie, modèle de pertes, gestion des prix, observation, récompense, etc.). Cela évite la dépendance à une lib externe peu maintenue et donne un contrôle complet sur la physique simulée.

En contrepartie, il faudra dans les travaux à venir mener des **tests et études de robustesse comparés à pymgrid** afin de vérifier que ce module custom est aussi fiable que l'environnement de référence.

---

## 5. Expériences

### exp01 — Perfect foresight, prix fixes

Premier scénario de référence : prévisions PV/charge parfaites et **prix d'achat / vente fixes et égaux**. Permet de valider que la chaîne (env custom, agent SAC, MILP) fonctionne dans le cadre le plus simple.

**Problématique identifiée :** avec des prix fixes et égaux, l'agent RL produit des **cycles de charge/décharge excessifs**. En optimisant pas à pas, il ne perçoit pas que décharger à l'instant *t* (même bénéfique localement) **augmente le coût des charges futures** nécessaires pour ramener la batterie à un SoC exploitable. Le critère immédiat est neutre, mais l'usure et les pertes par cycle ne sont pas internalisées.

### exp02 — Prix variables

Pour observer le changement de comportement du RL face à un signal économique non trivial, `configs/exp02_variable_price.yaml` introduit des **prix variables** (non constants dans le temps).

**Observation :** avec des prix variables, l'agent ne fait plus de cycles parasites — mais il **ignore complètement la batterie**, comme si l'arbitrage temporel n'était plus rentable.

**Hypothèse de travail :** ce comportement vient probablement de la combinaison de :
- **rendements (charge/décharge) fixes** au lieu de variables, qui ne capturent pas la dépendance réelle des pertes à la puissance et au SoC ;
- **absence de coût par cycle** (vieillissement, dégradation), qui rend toute utilisation de la batterie « gratuite » ou « inutile » selon la configuration.

**À faire :** explorer dans les articles déjà collectés dans **Zotero** une **expression plus intelligente des rendements** (dépendants du SoC et de la puissance) et un coût explicite par cycle, afin que l'agent retrouve un usage non trivial de la batterie sous prix variables.

---

## 6. État actuel & perspectives

### ✅ Implémenté
- Agent **SAC** via Stable-Baselines3
- Environnement **Gymnasium custom** (`envs/base_microgrid_env.py`) en remplacement de pymgrid, totalement modifiable
- Pipeline de données basé sur `Pyrano1w_clean.csv`
- Configuration des expériences par fichiers **YAML** (exp01 prix fixes, exp02 prix variables)
- Scripts d'optimisation centralisés dans `scripts/rl/` et `scripts/milp/`

### 🔜 À venir
- Comparaison avec d'autres algorithmes : **DDPG**, **PPO**
- **Rendements variables** (dépendant du SoC / de la puissance) et **coût par cycle** — pour corriger le comportement observé sur exp01 (cycles excessifs) et exp02 (batterie ignorée). Recherche bibliographique à mener à partir de Zotero.
- **Étude de robustesse de l'environnement custom** vs pymgrid (cas-tests comparatifs)
- Ajout de **plots de visualisation** comparables aux sorties MATLAB du EMS original
- Tests de **robustesse des paires sin/cos** (voir section Données)

---

## 7. VIBE CODING ALERT ⚠️

Le code de ce projet a été **généré par Claude Code (Anthropic)**, mais il a été **exhaustivement vérifié à la main** : aucune ligne n'a été acceptée sans relecture. Les prompts utilisés pour la génération ont été construits minutieusement (itérations, contraintes architecturales, vérifications croisées) et sont disponibles dans **[prompts.md](./prompts.md)** pour reproductibilité et inspection.
