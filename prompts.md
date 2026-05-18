**Prompt pour 1e implementation (prévision parfaite, charges fixes + sinusoidale, rendement fixes, prix fixes) :** 
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

**Prompt pour plots :**
=================

You are tasked with implementing a Python monitoring and visualization system for a 
Reinforcement Learning Energy Management System (RL-EMS). This system replicates and 
adapts the existing MATLAB monitoring infrastructure built by my supervisor.

CRITICAL CONTEXT: These plots are NOT used during RL training. They are used AFTER 
training, when evaluating the trained agent on a concrete optimization example 
(e.g. a full day). The trained model is loaded, meteorological data is fed in as 
CSV input, and the agent runs step-by-step through a simulation that mimics real 
deployment conditions. Once the simulation is complete, the two plots are displayed 
interactively via matplotlib's plt.show(). All data inputs and outputs are CSV files.

## STEP 1 — READ AND UNDERSTAND (mandatory before writing any code)

Read the following MATLAB files carefully and in order. For each file, also read any 
file it references or calls if relevant to understanding data structures or plot logic.

Primary files to read:
1. duchaud-JL/model-predictive-control/Classes/EnergyManagementSystem/@EMS/plot.m
   → Main visualization: area-stairs plot of power variables on cross axes.
   NOTE: my supervisor implemented complicated workarounds to compensate for 
   MATLAB's lack of native filled-stair support. In Python with matplotlib we can 
   achieve this much more simply with fill_between(..., step='post'). Read the file 
   to understand WHAT is displayed and the visual conventions, not to replicate 
   HOW it's implemented.
2. duchaud-JL/model-predictive-control/Classes/MicroGrid/@MicroGrid/MicroGrid.m
   → Lines 41-46: definition of the `monitoTable` dependent property and the 
     raw `Monitoring` matrix (nPoints × 6), columns: [t, Pp, Pl, SoC, Pb, Pg]
3. duchaud-JL/model-predictive-control/Classes/MicroGrid/subclasses/@MicroGridSimu/insert_monitoring_data.m
   → How monitoring data is written at each timestep (pre-allocated indexed write)
4. duchaud-JL/model-predictive-control/Classes/PowerManagementSystem/@PMS/follow.m
   → Line 3: the call to insert_monitoring_data(state) at each timestep
5. duchaud-JL/model-predictive-control/Classes/ModelPredictiveControl/@MPC/plot_splitted.m
   → Second plot: Pp, Pb, Pg, SoC columns compared against MILP setpoints 
     from concat_optim_history()
6. duchaud-JL/model-predictive-control/Classes/ModelPredictiveControl/@MPC/simu.m
   → Line 15: initialization of the Monitoring matrix at simulation start

Also read any helper referenced (e.g. num2dt(), concat_optim_history()) if needed 
to understand data shapes or time handling.

After reading, write a brief internal summary of:
- The exact column semantics of the monitoring matrix
- The visual structure of both plots (what goes where, colors, axes orientation)
- The "cross axes" trick in plot.m (positive = discharge/production above, 
  negative = charge/consumption below)
- How plot_splitted.m compares setpoints vs actuals

Finally read the current RL implementations to understand the signs conventions and logic.

## STEP 2 — IMPLEMENT monitoring_table.py

Create `monitoring/monitoring_table.py`:

- Class `MonitoringTable` that accumulates state/action data at each RL 
  decision step during a post-training optimization simulation.
- Internal storage: pre-allocated numpy array of shape (n_steps, 6), columns:
  [timestamp, Pp, Pl, SoC, Pb, Pg]
  matching exactly the MATLAB Monitoring matrix semantics.
- Method `insert(step_idx, state_dict)` to write a row at each timestep.
  `state_dict` keys: `pp` (PV power), `pl` (load), `soc` (battery SoC, 0-100%), 
  `pb` (battery power, positive=discharge), `pg` (grid power).
  Mirror of MATLAB's insert_monitoring_data — indexed write into pre-allocated array.
- Method `to_dataframe()` returning a pandas DataFrame with named columns and 
  a proper datetime index (equivalent of MATLAB's timetable + num2dt()).
- Method `reset(n_steps)` to reinitialize for a new simulation run.
- Method `to_csv(path)` to export the full monitoring table as CSV.
- Method `get_total_cost(buy_price, sell_price)` that computes the total 
  optimization cost from the Pg column (grid import/export), to be displayed 
  as the plot title later.
- Add docstrings explaining sign convention: Pb positive = battery discharging 
  = power injected into the microgrid bus. Pg positive = export to grid, 
  negative = import from grid.

## STEP 3 — IMPLEMENT plot_power.py (filled-stairs power plot)

Create `monitoring/plot_power.py` replicating the VISUAL RESULT of 
@EMS/plot.m using matplotlib. This is the main output visualization of an 
optimization run.

Context for understanding the plot:
- In the MILP case, this plot shows the planned actions for a 6h optimization 
  horizon, all known in advance.
- In the RL case, the agent decides one action per timestep Δt. To generate 
  the equivalent plot, we run the trained agent step-by-step through the 
  simulation, accumulate all decisions in MonitoringTable, and THEN build 
  this plot from the complete table. The result looks identical — stacked 
  filled stairs — but was built incrementally rather than from a single plan.

Requirements:
- Function `plot_power(monitoring_df, delta_t_minutes=10, cost=None)` 
  → displays the plot via plt.show(), no file saved
- "Filled stairs" style: use `ax.fill_between(..., step='post')` to create 
  filled step curves. This replaces the complicated MATLAB workarounds simply.
- Cross-axes layout: y=0 line clearly visible as the dividing axis between:
  - Upper half (y > 0): production / discharge / import
  - Lower half (y < 0): consumption / charge / export
  Implement via `ax.spines['bottom'].set_position('zero')`, hide top/right 
  spines, keep left spine.
- Variables to plot as filled stairs:
  - Pp: PV production (positive, yellow/orange fill)
  - Pl: Load demand (negative, red/coral fill); for the fixed case not stairs needed but only the superposition of the load curve self.base_load_kw * np.abs(np.sin(2.0 * np.pi * t_hours / 48.0)) from envs/components/load.py
  - Pb: Battery power (green if discharging >0, blue if charging <0; 
    split into two traces by sign)
  - Pg: Grid power (grey fill, positive=import, negative=export)
- The layout must make it visually obvious whether at any instant t the 
  battery is charging AND discharging simultaneously (which would be 
  physically impossible — this is a key validation check).
- Title: include the total optimization cost if `cost` is provided 
  (e.g. "Optimization Result — Cost: 12.34 €")
- X-axis: time labels formatted as HH:MM
- Y-axis label: Power (kW)
- Use matplotlib's interactive backend: the user will zoom and pan using 
  the toolbar (built-in with plt.show()). This is sufficient for analysis.

## STEP 4 — IMPLEMENT plot_monitoring.py (comparison plot with forecast errors)

Create `monitoring/plot_monitoring.py` adapting @MPC/plot_splitted.m 
to the RL context. This plot compares what actually happened against what 
was forecast.

Requirements:
- Function `plot_monitoring(monitoring_df, forecast_df=None, delta_t_minutes=10)` 
  → displays the plot via plt.show(), no file saved
- `forecast_df`: optional DataFrame with the same columns as monitoring_df 
  but containing the FORECAST values (predicted PV, predicted load, etc.) 
  that the agent received as observations. When provided, forecast is 
  overlaid on top of actuals to visualize prediction errors.
- 4-panel subplot figure (4 rows × 1 col, shared x-axis):
  Panel 1: Pp — actual PV (solid line) + forecast PV (dashed line, if provided)
  Panel 2: Pl — actual load (solid) + forecast load (dashed, if provided)
  Panel 3: Pb — battery power with cross-axes (y=0 spine), actual only 
    (battery action is decided by the agent, not forecast)
  Panel 4: SoC — actual trajectory with markers at each decision step (dots 
    at each Δt to show the step-by-step nature of RL decisions). 
    Add horizontal dashed lines at 20% and 90% (operational bounds). 
    Y-axis 0-100%.
- When forecast_df is provided, shade the area between forecast and actual 
  on Pp and Pl panels to highlight forecast errors visually 
  (use fill_between with a semi-transparent color).
- Each panel is independently zoomable via matplotlib toolbar.
- Subtitle: "RL Agent — step-by-step decisions (Δt = X min)"

## STEP 5 — IMPLEMENT run_optimization_example.py

Create `monitoring/run_optimization_example.py` — a standalone script 
that simulates real-world deployment conditions for the trained RL agent.

This script mimics what happens in live deployment: the agent receives 
meteorological data (PV irradiance forecasts, load forecasts) and makes 
decisions step-by-step, exactly as it would on the real microgrid.

Input data (all CSV files):
- Meteorological / forecast CSV: contains the forecasted PV irradiance and 
  load for the simulation horizon (e.g. columns: timestamp, pv_forecast, 
  load_forecast). This is what the agent "sees" as observation.
- Optionally, a ground-truth CSV with actual PV and load values 
  (to compute forecast errors for plot_monitoring).

Pipeline:
1. Load the trained RL model from a saved checkpoint
2. Load the meteorological forecast CSV via pandas
3. Initialize MonitoringTable with the number of steps for the simulation 
   horizon (e.g. 144 steps for a 24h day at Δt=10min)
4. Run the simulation loop:
    obs = env.reset()
    for step in range(n_steps):
      action = model.predict(obs)
      obs, reward, done, info = env.step(action)
## MonitoringTable.insert() is called automatically inside env.step()
5. Export MonitoringTable as CSV: `monitoring_table.to_csv('monitoring/runs/monitoring_table.csv')`
6. Compute total cost via `monitoring_table.get_total_cost()`
7. Display both plots interactively :
   - `plot_power(monitoring_df, cost=total_cost)`
   - `plot_monitoring(monitoring_df, forecast_df=forecast_data)`
   The user inspects, zooms, closes the windows.

Output files (all CSV):
- `monitoring/runs/monitoring_table.csv` — full monitoring table (the key deliverable 
  for my supervisor)
- The meteorological input CSV is kept as-is for traceability

NO HTML output, NO PNG output. Plots are displayed live via plt.show() 
for interactive analysis. CSVs are the permanent record.

Find the existing RL codebase to understand:
- How the trained model is saved and loaded
- What the observation/action format looks like
- How to extract Pp, Pl, SoC, Pb, Pg from the env's state/obs/info
Create a mapping dict if variable names differ from the MATLAB conventions.

## STEP 6 — WIRE MonitoringTable INTO THE ENVIRONMENT

Find the existing environment or simulation code. Add MonitoringTable as an 
attribute of the environment (or a wrapper) so that `insert()` is called 
automatically at each `env.step()`. This way any script running the env 
(evaluation, optimization example) automatically accumulates monitoring data 
without manual insertion code scattered everywhere.

The MonitoringTable should be accessible via `env.monitoring_table` and 
auto-reset on `env.reset()`.

## STEP 7 — TESTS

Write `tests/test_monitoring.py` with at least:
- A test that inserts 144 synthetic timesteps (24h with 10 minutes time steps) and checks `to_dataframe()` 
  has correct shape, column names, and datetime index
- A test that `get_total_cost()` returns correct value for known inputs
- A test that `to_csv()` produces a valid CSV readable by pandas
- A test that calls `plot_power()` on synthetic data without error 
  (use matplotlib's Agg backend to avoid display: 
  `matplotlib.use('Agg')` before import pyplot)
- A test that calls `plot_monitoring()` with and without forecast_df 
  without error
- A test that verifies sign conventions: inserting Pb > 0 appears in the 
  upper half, Pb < 0 in the lower half

## CONVENTIONS AND CONSTRAINTS

- All new files go under `monitoring/` with an `__init__.py` 
  that exports MonitoringTable, plot_power, plot_monitoring
- Dependencies: numpy, pandas, matplotlib (all already in the project). 
  No plotly, no HTML output.
- All persistent data is CSV. Plots are transient (plt.show() only).
- Sign convention for Pb must match current implementation and MATLAB exactly: 
  positive = battery discharging = power into the bus
- The cross-axes layout is REQUIRED — it is my supervisor's visual convention 
  and must be preserved for compatibility
- Add `# MATLAB equivalent: <filename> line X` comments near any logic 
  that directly mirrors a MATLAB implementation detail
- If the existing Python codebase uses different variable names for 
  Pp/Pl/Pb/Pg/SoC, create a mapping dict and document the correspondence
