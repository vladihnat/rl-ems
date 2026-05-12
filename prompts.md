Prompt pour 1e implementation (prévision parfaite, charges fixes + sinusoidale, rendement fixes, prix fixes) : 
============================================================================================

Implement the initial experiment (exp01: perfect foresight, fixed load, fixed efficiency, fixed prices) for a microgrid RL project. Read the python-microgrid docs at https://python-microgrid.readthedocs.io/en/latest/ and the repo at https://github.com/ahalev/python-microgrid before writing any code, to understand the module APIs (BatteryModule, RenewableModule, LoadModule, GridModule) and ContinuousMicrogridEnv.

## Project structure

Create the following tree under `stage/`:

```
stage/
├── configs/
│   └── exp01_perfect_foresight.yaml
├── envs/
│   ├── __init__.py
│   ├── base_microgrid_env.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── pv_source.py
│   │   ├── load.py
│   │   └── battery.py
│   └── registry.py
├── agents/
│   ├── __init__.py
│   └── sac_agent.py
├── baselines/
│   ├── __init__.py
│   └── milp_solver.py
├── experiments/
│   └── run_experiment.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── compare.py
├── results/
│   └── .gitkeep
├── data/
│   └── .gitkeep              # user places Pyrano1w_clean.csv here
└── requirements.txt
```

## Config file: configs/exp01_perfect_foresight.yaml

```yaml
experiment:
  name: exp01_perfect_foresight
  seed: 42

time:
  delta_t_min: 5
  horizon_h: 6
  # derived: horizon_steps = (horizon_h * 60) / delta_t_min = 72

data:
  pv_csv: "data/Pyrano1w_clean.csv"
  pv_column: "Global30_kW"
  # Columns used from the CSV: hour_sin, hour_cos, doy_sin, doy_cos, Global30_kW

battery:
  capacity_kwh: 24.0           # 30kWh * 0.8 usable
  soc_min: 0.2
  soc_max: 0.9
  max_charge_kw: 7.0
  max_discharge_kw: 7.0
  efficiency_charge: 0.9
  efficiency_discharge: 0.9
  init_soc: 0.5
  cost_cycle: 0.02

grid:
  max_import_kw: 17.0          # limited by inverter Pmax
  max_export_kw: 17.0
  price_import: 0.15           # €/kWh — fixed for exp01
  price_export: 0.15           # €/kWh — equal to import for exp01

load:
  type: fixed                  # fixed sinusoidal load from Matlab: abs(sin(t)) * 1e3 W
  base_load_kw: 1.0            # peak of sinusoidal load in kW

pv:
  type: perfect                # perfect foresight, no noise
  surface_m2: 91.32
  eta_ref: 0.24               # rendement STC du module
  # P_pv_kw = Global30_kW * surface_m2 * eta_ref

reward:
  sigma_soc: 10.0              # penalty weight for SoC violations
  soc_safe_min: 0.2
  soc_safe_max: 0.9

training:
  algorithm: SAC
  total_timesteps: 200000
  learning_rate: 0.0003
  batch_size: 256
  buffer_size: 100000
  train_split: 0.8             # fraction of DATES (not rows) for training
  split_method: temporal       # chronological, never random
```
## Train/test split — TEMPORAL ONLY, NO SHUFFLING

The split MUST be chronological to avoid data leakage:
- Sort all data by Time (already sorted in CSV)
- Find the unique dates in the dataset
- First 80% of dates → training set
- Last 20% of dates → test set
- The cut happens at midnight of the split date (no partial days)

## Component implementations

### envs/components/battery.py

Implement a `BatteryModel` class that mirrors the Matlab simulation logic.
Critical: follow the supervisor's sign convention from simulate_microgrid.m:
- Pb > 0 means DISCHARGING (battery provides power)
- Pb < 0 means CHARGING (battery absorbs power)
- Efficiency applied as:
  - charging:    nu = eta_charge        (when Pb < 0)
  - discharging: nu = 1 / eta_discharge (when Pb >= 0)
- SoC update: SoC_new = SoC_old - Pb * nu * dt / Capacity
- Clip SoC to [0, 1]
- Recompute effective Pb from actual delta_SoC:
    dE = (SoC_old - SoC_new) * Capacity
    Pb_effective = dE / dt / nu

This class wraps the python-microgrid BatteryModule but enforces these conventions.
Expose a `step(action_kw) -> (Pb_effective, new_soc)` method.

### envs/components/pv_source.py

Implement a `PVSource` class:
- Loads the CSV, extracts the PV column and temporal features
- `get_irradiance(step_index) -> float` reads irradiance from CSV (kW/m²),
  multiplies by surface_m2 * eta_ref, and returns PV power output in kW.
  The name follows supervisor's Matlab convention.
- `get_forecast(step_index, horizon_steps) -> np.ndarray` returns the next
  `horizon_steps` values. For exp01 (perfect foresight), this is simply
  the actual future values from the CSV.
- Later experiments will add noise to this forecast.

### envs/components/load.py

Implement a `LoadModel` class:
- For `type: fixed`, generates a sinusoidal load: `P_load(t) = base_load_kw * |sin(2π * t_hours / 48)|`
  where t_hours is the fractional hour of day.
- `get_load(step_index) -> float` returns load in kW
- `get_forecast(step_index, horizon_steps) -> np.ndarray` returns future load values.
  For fixed load with perfect foresight, these are deterministic.

### envs/base_microgrid_env.py

Implement `MicrogridEnv(gymnasium.Env)`:

**Action space**: `Box(low=-1, high=1, shape=(1,))` — normalized battery command.
  - Map [-1, 0) → charging at |action| * max_charge_kw
  - Map [0, 1]  → discharging at action * max_discharge_kw

**Observation space**: `Box` with the following vector (dimension = 9 + HORIZON_STEPS):
```python
observation = np.concatenate([
    # Temporal context (4)
    [hour_sin, hour_cos, doy_sin, doy_cos],
    # System state (2)
    [soc, load_t],
    # Economic signals (2)
    [price_import, price_export],
    # PV: current + forecast (1 + HORIZON_STEPS)
    [pv_t],
    pv_forecast_t1_to_tH,       # shape (HORIZON_STEPS,)
])
```

**Reward function** — computed at each step:
```
# Grid power balance (supervisor convention: Pg = -(Pl + Pp + Pb))
P_grid = -(load_t + pv_t + Pb_effective)
# where pv_t is POSITIVE (generation), load_t is NEGATIVE (consumption)
# so: P_grid > 0 means importing, P_grid < 0 means exporting

# Economic reward
delta_t_h = delta_t_min / 60
r_eco = -(price_import * max(P_grid, 0) - price_export * max(-P_grid, 0)) * delta_t_h

# SoC penalty
r_soc = -sigma * (max(0, soc_min - soc) + max(0, soc - soc_max))

reward = r_eco + r_soc
```

**Sign convention — grid-oriented, matching supervisor's Matlab**:
- `Pl` (load):     NEGATIVE (consumption, power flowing OUT of bus)
- `Pp` (PV):       POSITIVE (generation, power flowing INTO bus)
- `Pb` (battery):  POSITIVE = discharge, NEGATIVE = charge
- `Pg` (grid):     POSITIVE = import, NEGATIVE = export
- Balance: Pg + Pp + Pl + Pb = 0  →  Pg = -(Pl + Pp + Pb)

Component methods always return positive magnitudes:
  pv_t   = PVSource.get_irradiance(step_index)  # kW, magnitude
  load_t = LoadModel.get_load(step_index)         # kW, magnitude

Signs are applied only when assembling P_grid:
  P_grid = -((-load_t) + (+pv_t) + Pb_effective)
         = load_t - pv_t - Pb_effective

Unit test assertions (add in battery.py __main__):
  assert P_grid(load=5, pv=3, Pb=0) ==  2.0  # import
  assert P_grid(load=2, pv=5, Pb=0) == -3.0  # export


**reset()**: returns initial observation, resets SoC to init_soc, step_index to 0.
**step(action)**: returns (obs, reward, terminated, truncated, info).
  `terminated = True` when step_index reaches end of data minus HORIZON_STEPS.
  `info` dict contains: Pb_effective, P_grid, soc, r_eco, r_soc, pv_t, load_t.

### envs/registry.py

Implement `make_env(config_path: str) -> MicrogridEnv` that:
- Loads the YAML config
- Instantiates components (PVSource, LoadModel, BatteryModel)
- Returns a configured MicrogridEnv
- Also returns the config dict for saving with results

In registry.py, `make_env` should return two envs:
  `train_env` (data from first dates) and `test_env` (data from last dates).
  Each env only sees its own date range. The test env must NOT be used
  during training in any way (no normalization stats, no replay buffer).

## agents/sac_agent.py

Implement `train_sac(env, config) -> model` and `evaluate_sac(model, env) -> dict`:
- Uses `stable_baselines3.SAC` with `MlpPolicy`
- Hyperparameters from config
- Returns trained model
- evaluate runs the policy on the test portion and returns metrics dict

## baselines/milp_solver.py

Implement `run_milp(env, config) -> dict`:
- Uses CVXPY with HiGHS solver
- Solves the full horizon optimal dispatch with perfect information
- Decision variable: Pb[t] for each timestep
- Objective: minimize total import cost minus export revenue
- Constraints:
    - Power balance: Pg[t] = -(Pl[t] + Pp[t] + Pb[t])
    - SoC dynamics matching battery model
    - SoC bounds [soc_min, soc_max]
    - Pb bounds [-max_charge, max_discharge]
    - Pg bounds [-max_export, max_import]
- Returns dict with same metrics structure as RL evaluation

## evaluation/metrics.py

Implement functions to compute from a completed episode:
- `total_cost`: sum of import costs
- `total_revenue`: sum of export revenue
- `net_cost`: total_cost - total_revenue
- `soc_violations`: count of timesteps where SoC left [soc_min, soc_max]
- `self_consumption_rate`: 1 - (energy_imported / total_load)
- `peak_grid_import`: max P_grid over episode

## evaluation/compare.py

Implement `compare_results(rl_metrics, milp_metrics) -> dict`:
- Computes relative gap: (rl_net_cost - milp_net_cost) / milp_net_cost
- Prints a formatted comparison table
- Saves comparison to JSON

## experiments/run_experiment.py

Single entry point:
```
python experiments/run_experiment.py --config configs/exp01_perfect_foresight.yaml
```

Workflow:
1. Load config, set random seed
2. Create env via registry
3. Split data: 80% train, 20% test  # chronological, never random !
4. Train SAC on train portion
5. Evaluate SAC on test portion → rl_metrics
6. Run MILP on test portion → milp_metrics
7. Compare and save results to `results/{experiment_name}/`:
   - `config_used.yaml` (copy of config)
   - `metrics.json` (RL + MILP + comparison)
   - `training_curves.png` (reward over episodes)
   - `sac_model.zip` (saved SB3 model)

## requirements.txt

```
python-microgrid>=1.4.0
stable-baselines3>=2.1.0
gymnasium
numpy
pandas
cvxpy
pyyaml
matplotlib
```

## Implementation notes

- All kW, all hours. Never mix W and kW or seconds and hours.
- Use `np.float32` for observation and action spaces (SB3 convention).
- Every file must have a module docstring explaining its role.
- The battery model is the critical piece — unit test it against the Matlab
  simulate_microgrid.m logic before proceeding. Include a simple test at the
  bottom of battery.py under `if __name__ == "__main__"` that reproduces
  the Matlab example: SoC=0.5, Pb=100kW (way above max, should clip),
  dt=1min, and prints the resulting SoC and effective Pb.
- Global30_kW in the CSV is irradiance in kW/m², NOT power. To get PV power:
  P_pv_kw = Global30_kW * surface_m2 * eta_ref (0.24). Handle this in PVSource.
  If resulting values seem unreasonable, add a warning log.

  ----------------



  