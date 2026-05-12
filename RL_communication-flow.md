 Module Hierarchy & Communication Flow
 =====================================

### Points clés du fonctionement interne :

Voici la lecture rapide de ce que l'agent calcule et de ce qu'il optimise.

- **Bilan de puissance réseau.** À chaque pas de temps, l'environnement calcule le résidu échangé avec le réseau :

  ```
  P_grid = P_load − P_pv − P_bat
  ```

  où `P_bat` est l'action de l'agent (puissance batterie, **signée** : `< 0` = charge, `> 0` = décharge). Si `P_grid > 0` le microréseau **achète** ; si `P_grid < 0` il **revend** son surplus.

- **Coût instantané → signal de récompense.** Le coût économique sur le pas est :

  ```
  coût  = price_import * max( P_grid, 0.0)     # achat réseau (P_grid > 0)
  revenu = price_export * max(−P_grid, 0.0)    # vente surplus (P_grid < 0)
  ```

  Un terme de pénalité **soft** est ajouté lorsque le SoC sort de la bande de sécurité `[SoC_min, SoC_max]` :

  ```
  r_soc = − sigma_soc * (
            max(0.0, soc_safe_min − new_soc)
          + max(0.0, new_soc − soc_safe_max)
        )
  ```

  La récompense renvoyée à SAC (i.e. l'agent) est l'opposé du coût net, plus `r_soc` ; minimiser le coût ≡ maximiser la récompense.

- **Objectif global.** L'agent cherche à **minimiser les coûts cumulés** sur l'horizon — ce qui revient économiquement à **maximiser l'auto-consommation** et **minimiser les achats réseau**, exactement comme la fonction objectif du MILP, mais appris par essais/erreurs plutôt que résolu analytiquement.

- **Dynamique du SoC.** L'état de charge évolue selon :

  ```
  SoC(t+1) = SoC(t) + ( P_charge * η_charge  −  P_décharge / η_décharge ) * Δt / Capacité
  ```

  avec les rendements de charge/décharge appliqués de manière dissymétrique (pertes dans les deux sens).

- **Où c'est dans le code.** Ces équations sont entièrement encapsulées dans l'environnement Gymnasium custom. Leur implémentation exacte se trouve dans **[envs/base_microgrid_env.py et envs/components/battery.py]** : `base_microgrid_env.py` réalise le bilan de puissance et compose la récompense, `battery.py` applique le clamp d'action et la mise à jour du SoC.

---

```
  experiments/run_experiment.py          ← Entry point (CLI)
  │
  │  sys.path.insert(0, project_root)   ← makes all packages importable
  │
  ├── envs/registry.py :: make_env(config_path)
  │   │   reads YAML config
  │   │
  │   ├── envs/components/pv_source.py :: PVSource(cfg["pv"], cfg["data"])
  │   │       loads CSV → builds pv_power[], timestamps[], temporal features[]
  │   │       .set_data_slice(indices)  → restricts to train or test window
  │   │
  │   ├── envs/components/load.py :: LoadModel(cfg["load"], n_steps, ...)
  │   │       generates sinusoidal load profile aligned to PVSource timestamps
  │   │
  │   ├── envs/components/battery.py :: BatteryModel(cfg["battery"])
  │   │       stateful: holds self.soc, clamps commands, applies efficiency
  │   │
  │   └── envs/base_microgrid_env.py :: MicrogridEnv(pv, load, battery, cfg)
  │           wraps all three components as a gym.Env
  │           observation_space: Box(9 + horizon_steps,)
  │           action_space:      Box(-1, 1, shape=(1,))
  │
  ├── agents/sac_agent.py :: train_sac(env, cfg)
  │   │   receives a MicrogridEnv — SAC never imports envs/ directly
  │   │   uses SB3's SAC("MlpPolicy", env, ...)   ← reads obs/action spaces
  │   │   model.learn() calls env.step() / env.reset() in a loop
  │   │   RewardLoggerCallback hooks into SB3's on_step event
  │   │
  │   └── :: evaluate_sac(model, env)
  │           model.predict(obs) → action → env.step(action) → info dict
  │           reads env.delta_t_h, env.cfg, env.price_import, env.price_export
  │           calls evaluation/metrics.py :: compute_metrics(history, ...)
  │
  ├── baselines/milp_solver.py :: run_milp(env, cfg)
  │       receives same MicrogridEnv for physics params
  │
  └── evaluation/
      ├── metrics.py :: compute_metrics(history, delta_t_h, ...)
      │       pure function — no imports from envs/ or agents/
      └── compare.py :: compare_results(rl_metrics, milp_metrics, output_dir)
```
---

# Data / Call Flow Per Step
```
env.step(action)
└─ BatteryModel.step(Pb_command, delta_t_h)   → Pb_effective, new_soc
└─ PVSource.get_irradiance(step_index)         → pv_t  [kW]
└─ LoadModel.get_load(step_index)              → load_t [kW]
└─ P_grid = load_t - pv_t - Pb_effective
└─ r_eco  = -(price_import*max(P_grid,0) - price_export*max(-P_grid,0)) * dt
└─ r_soc  = -sigma_soc * SoC violation magnitude
└─ returns (obs, reward, terminated, truncated, info)

env._get_obs()  ← called every step
└─ PVSource.get_temporal_features(i)  → [h_sin, h_cos, d_sin, d_cos]
└─ BatteryModel.soc                   → scalar
└─ LoadModel.get_load(i)              → scalar
└─ PVSource.get_irradiance(i)         → scalar
└─ PVSource.get_forecast(i, horizon)  → float32 array [horizon_steps]
└─ concatenated → obs vector (9 + horizon_steps,)
```

---
# Key Modularity Points

* Boundary: run_experiment ⇾ envs/
Mechanism: make_env(config_path) returns two fully-built MicrogridEnv objects; the runner never touches components directly

* Boundary: envs/registry ⇾ components
Mechanism: Constructs PVSource, LoadModel, BatteryModel from config dict, then passes them into MicrogridEnv.__init__ by dependency injection

* Boundary: agents/sac_agent ⇾ env
Mechanism: Receives a gym.Env duck type — communicates only via .step(), .reset(), .observation_space, .action_space, and a handful of named attributes (delta_t_h, cfg, price_*)

* Boundary: agents/sac_agent ⇾ metrics
Mechanism: Imports evaluation.metrics.compute_metrics directly; the metric function is stateless

* Boundary: Config propagation
Mechanism: Raw YAML dict (cfg) is passed from make_env → MicrogridEnv → stored as self.cfg; sac_agent reads env.cfg["reward"] for SoC bounds after evaluation

The RL agent never imports envs/. It only receives an opaque env object and communicates through the Gymnasium interface (step/reset/predict). All physics lives inside MicrogridEnv and
its three components.