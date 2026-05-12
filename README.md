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
├── test_compat.py
├── explore_pymgrid25.py
├── agents/
│   ├── __init__.py
│   └── sac_agent.py
├── baselines/
│   ├── __init__.py
│   └── milp_solver.py
├── configs/
│   └── exp01_perfect_foresight.yaml
├── data/
│   ├── clean_meteo.py
│   ├── Pyrano1w.csv
│   ├── Pyrano1w_clean.csv
│   ├── Pyrano1M.csv
│   ├── Pyrano1M_clean.csv
│   ├── Pyrano1Y.csv
│   └── Pyrano1Y_clean.csv
├── envs/
│   ├── __init__.py
│   ├── base_microgrid_env.py
│   ├── registry.py
│   └── components/
│       ├── __init__.py
│       ├── battery.py
│       ├── load.py
│       └── pv_source.py
├── evaluation/
│   ├── __init__.py
│   ├── compare.py
│   └── metrics.py
├── experiments/
│   └── run_experiment.py
└── results/
    └── exp01_perfect_foresight/
        ├── comparison.json
        ├── config_used.yaml
        ├── metrics.json
        ├── sac_model.zip
        └── training_curves.png
```

---

## 2. Utilisation

Le point d'entrée principal est **`experiments/run_experiment.py`**. Il a été conçu pour être facilement généralisable : la totalité du paramétrage (algorithme RL, hyperparamètres, composants physiques, données, horizon, prix, contraintes SoC, etc.) est pilotée par les fichiers de configuration YAML du dossier `configs/`.

Exemple d'appel typique :

```bash
python experiments/run_experiment.py --config configs/exp01_perfect_foresight.yaml
```

Pour lancer une nouvelle expérience, il suffit de dupliquer un fichier YAML existant, d'ajuster les champs souhaités, et de relancer le script avec le nouveau chemin de configuration. Les artefacts (modèle entraîné, métriques, courbes d'apprentissage, copie de la config utilisée) sont déposés automatiquement dans `results/<nom_experience>/`.

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

---

## 5. État actuel & perspectives

### ✅ Implémenté
- Agent **SAC** via Stable-Baselines3
- Environnement **Gymnasium custom** pour microréseau PV + batterie
- Pipeline de données basé sur `Pyrano1w_clean.csv`
- Configuration des expériences par fichiers **YAML**

### 🔜 À venir
- Comparaison avec d'autres algorithmes : **DDPG**, **PPO**
- **Charges et rendements variables** (modélisation non-linéaire)
- Ajout de **plots de visualisation** comparables aux sorties MATLAB du EMS original
- Tests de **robustesse des paires sin/cos** (voir section Données)

---

## 6. VIBE CODING ALERT ⚠️

Le code de ce projet a été **généré par Claude Code (Anthropic)**, mais il a été **exhaustivement vérifié à la main** : aucune ligne n'a été acceptée sans relecture. Les prompts utilisés pour la génération ont été construits minutieusement (itérations, contraintes architecturales, vérifications croisées) et sont disponibles dans **[prompts.md](./prompts.md)** pour reproductibilité et inspection.
