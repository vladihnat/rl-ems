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

----

**exp02 — Variable Grid Pricing (MATLAB-inspired sinusoidal tariff)**
======================================================================

     Context

     The current implementation uses fixed import/export prices (0.15 EUR/kWh for both) defined as scalars
     in the YAML config. The reference MATLAB code (run_Ems.m) uses a time-varying price function:

     Param.fun_prix_reseau = @(t, E) (5 + sin(t)') .* E' .* (E'>0) + ones(size(t')) .* E' .* (E'<=0);

     This encodes:
     - Import price (E > 0): 5 + sin(t) — time-varying (sinusoidal pattern)
     - Export price (E ≤ 0): 1 — constant

     The goal is to add a new experiment exp02_variable_price that introduces this kind of time-dependent
     pricing. Because the price is now informative (it changes in a predictable pattern), the agent must see
      a price forecast in its observation to learn to shift energy use toward cheap periods — making this a
     meaningfully harder and more realistic experiment than exp01.

     ---
     What changes

     1. New: envs/components/price_signal.py

     New PriceSignal component, parallel to PVSource/LoadModel.

     Supports two types (via config["type"]):
     - "fixed" → wraps the current scalar prices, no forecast needed (backward-compatible with exp01)
     - "sinusoidal" → computes base_import + amplitude * sin(2π * hour / period_h) for import price,
     constant export price

     Pre-computes the full import/export price arrays from timestamps at construction.

     Public API:
     price_signal.get_import_price(step_idx: int) -> float
     price_signal.get_export_price(step_idx: int) -> float
     price_signal.get_import_forecast(step_idx: int, horizon: int) -> np.ndarray  # shape (horizon,)
     price_signal.has_forecast: bool  # True for sinusoidal
     price_signal.import_prices: np.ndarray  # full array (for MILP)
     price_signal.export_prices: np.ndarray  # full array (for MILP)

     2. New: configs/exp02_variable_price.yaml

     Clone of exp01 with:
     experiment:
       name: exp02_variable_price

     grid:
       max_import_kw: 17.0
       max_export_kw: 17.0
       price_type: sinusoidal
       base_import: 0.15        # EUR/kWh — mean import price
       amplitude: 0.05          # ±0.05 EUR/kWh variation (10–20 c€/kWh range)
       period_h: 24.0           # daily sinusoidal cycle
       base_export: 0.08        # constant export price (lower than import)

     The daily amplitude 0.05 on a base of 0.15 mirrors the MATLAB spirit (sin fluctuates around 5, export
     is flat at 1) but uses realistic EUR/kWh magnitudes. Peak import at noon (sin(π/2)=1) → 0.20 EUR/kWh;
     trough at midnight → 0.10 EUR/kWh.

     3. Modify: envs/registry.py

     In _build_env(), after constructing PVSource, also build PriceSignal from cfg["grid"] + pv.timestamps:
     from envs.components.price_signal import PriceSignal
     price_signal = PriceSignal(cfg_dict["grid"], pv.timestamps, cfg_dict["time"]["delta_t_min"])
     Pass it to MicrogridEnv(pv, load, battery, price_signal, cfg_dict).

     The cfg["grid"] for exp01 will not have price_type, so PriceSignal.__init__ defaults to "fixed" using
     the existing price_import/price_export keys — fully backward-compatible.

     4. Modify: envs/base_microgrid_env.py

     - Constructor: accept price_signal: PriceSignal parameter; store as self.price_signal; remove
     self.price_import/self.price_export scalar attributes
     - obs_dim computation:
     price_forecast_dim = self.horizon_steps if self.price_signal.has_forecast else 0
     obs_dim = 9 + self.horizon_steps + price_forecast_dim
     - _get_obs(): replace fixed price scalars with price_signal.get_import_price(step_index) /
     get_export_price(step_index); append price_signal.get_import_forecast(step_index, horizon_steps) if
     has_forecast
     - step(): use price_signal.get_import_price(step_index) / get_export_price(step_index) for reward;
     expose them in info dict

     5. Modify: baselines/milp_solver.py

     Replace scalar price_imp/price_exp with full arrays from the env's price signal:
     price_imp_vec = env.price_signal.import_prices[:T]   # shape (T,)
     price_exp_vec = env.price_signal.export_prices[:T]
     CVXPY objective becomes element-wise (still linear — valid MILP):
     objective = cp.Minimize(
         cp.sum(cp.multiply(price_imp_vec, P_imp) - cp.multiply(price_exp_vec, P_exp)) * delta_t_h
     )
     The history dict and compute_metrics call pass arrays instead of scalars.

     6. Modify: evaluation/metrics.py

     No code change needed — np.sum(import_power * price_import * delta_t_h) already works when price_import
      is a numpy array (broadcasting). Update the docstring type hint from float to float | np.ndarray.

     7. Modify: monitoring/monitoring_table.py

     Update get_total_cost(buy_price, sell_price) signature to accept float | np.ndarray. The internal
     computation already handles arrays (all numpy ops). Update docstring only — no logic change.

---

Battery Efficiency Degradation — Literature Extraction Task
===========================================================

    ## Context

    The current `BatteryModel` uses fixed, symmetric efficiencies
    (`eta_charge = 0.9`, `eta_discharge = 0.9`) that are constant regardless of SoC, cycle count,
    or thermal state. The goal of this task is to extract physically grounded ideas from the
    literature to inform a more realistic degradation model that the RL agent can learn to account
    for.

    The relevant source files are:
    - `envs/components/battery.py`   ← where the model lives today
    - `configs/` ← current battery parameters

    ---

    ## Input Articles

    Read the following three PDFs in this priority order:

    1. **[PRIMARY]** `rapport/articles/Shamim et al. - 2021 - Evaluating ZEBRA Battery Module under the Peak-Shaving Duty Cycles.pdf`
    2. `rapport/articles/Benato et al. - 2015 - Sodium nickel chloride battery technology for large-scale stationary storage in the high voltage net.pdf`
    3. `rapport/articles/Galloway and Haslam - 1999 - The ZEBRA electric vehicle battery power and energy improvements.pdf`

    Shamim et al. (2021) is the most directly relevant: it characterises ZEBRA efficiency under
    peak-shaving duty cycles, which closely matches our EMS use case.

    ---

    ## Extraction Targets

    For each article, extract and structure the following — **cite the source** (author + year +
    figure/table/section reference) for every extracted value or claim:

    ### 1. Efficiency as a function of operating point
    - Does charge efficiency η_c vary with C-rate, SoC, or power level?
    - Does discharge efficiency η_d vary similarly?
    - Are there empirical curves, lookup tables, or fitted equations? Copy the functional form.

    ### 2. Capacity fade / end-of-life indicators
    - How does usable capacity degrade with cycle count or calendar ageing?
    - Is there a cycle-number threshold or a capacity-fade curve (e.g., % remaining vs. number
      of cycles)?
    - Any distinction between shallow cycling and deep cycling in terms of degradation rate?

    ### 3. Coulombic vs. energy efficiency
    - Do the articles separate round-trip energy efficiency from Coulombic efficiency?
    - If so, which one is more relevant for our SoC dynamics equation?

    ### 4. Thermal dependency
    - Is efficiency or degradation reported as a function of temperature?
    - If yes, at what operating temperature were the nominal values measured?
      (Our system is in Corsica; thermal context matters.)

    ### 5. Self-discharge
    - Is any self-discharge rate reported (% SoC lost per hour or per day at rest)?

    ### 6. Direct implementation hints
    - Do the authors propose any simplified model (e.g., Peukert-style, empirical polynomial,
      piecewise linear) that could be implemented in Python with reasonable parameter count?
    - Are there ready-to-use parameter values (capacity in Ah or kWh, efficiency at 1C, 2C, etc.)?

    ---

    ## Output Format

    **Produce a single Markdown document saved to `rapport/battery_degradation_notes.md`.**

    Structure it as follows:

    ## Battery Degradation — Literature Notes
    1. Summary table


      |Topic | Value/Equation|  Source   |  Confidence      |
      |------|---------------|-----------|------------------|
      |...   | ...           | ...       |  ...             |


    2. Per-article findings

    Shamim et al. (2021)
    ...
    Benato et al. (2015)
    ...
    Galloway & Haslam (1999)
    ...

    3. Implementation candidates

    Rank 2–4 concrete ideas by implementation complexity (low / medium / high) and expected
    fidelity gain. For each, describe:
    * What changes in battery.py
    * What new parameters enter configs/*.yaml
    * What new observation features (if any) the RL agent would need

    4. Open questions

    List physical or modelling ambiguities that would need clarification before implementation.
    Do **not** generate any code yet — this is a reading and synthesis task only.

Monitoring enhancement — columns, validation, plots
===================================================
    ## Context & objective

    The project is a microgrid EMS-RL (Energy Management System via Reinforcement Learning). The codebase has:
    - A custom Gymnasium env (`envs/base_microgrid_env.py`) for the RL agent (SAC via Stable-Baselines3)
    - A MILP baseline (`baselines/milp_solver.py`) solved with CVXPY + HiGHS
    - A monitoring layer (`monitoring/monitoring_table.py`) that currently stores only 6 columns:
      ["t", "Pp", "Pl", "SoC", "Pb", "Pg"]
    - Three plot modules: `plot_power.py`, `plot_monitoring.py`, `plot_monitoring_milp.py`

    The goal is to enrich the monitoring pipeline so that post-hoc validation and diagnosis of
    both the MILP and the RL agent are made significantly easier.

    ---

    ## Task 1 — extend monitoring columns (both solvers)

    ## MonitoringTable (`monitoring/monitoring_table.py`)

    Extend the column schema from 6 to a richer set. The new target schema is:

      SHARED (both RL and MILP):
        t, Pp, Pl, SoC [%], Pb, Pg,
        P_imp [kW, ≥ 0],    # = max(Pg, 0)  — power drawn from grid
        P_exp [kW, ≥ 0],    # = max(-Pg, 0) — power exported to grid
        price_imp [€/kWh],  # import price at this step (scalar from PriceSignal)
        price_exp [€/kWh],  # export price at this step
        r_eco [€],          # instantaneous economic reward
        r_soc,              # SoC penalty (0 if no violation)
        reward              # total step reward = r_eco + r_soc

      MILP-ONLY (pass NaN from RL):
        Pb_charge [kW, ≥ 0],    # auxiliary decomposition variable from CVXPY
        Pb_discharge [kW, ≥ 0], # auxiliary decomposition variable from CVXPY
        b_int [0 or 1]           # binary mutual-exclusion variable (b[t] from milp_solver.py)

      RL-ONLY (pass NaN from MILP):
        action_raw               # raw action ∈ [-1, 1] before scaling
        Pb_command               # scaled action in kW before battery clamp

    Implementation requirements:
    - Update `COLUMN_NAMES`, `NUM_COLUMNS`, and all `COL_*` index constants.
    - `insert()` must accept the new keys as optional kwargs (unknown keys → silently ignored,
      for backward compatibility). Provide sane defaults (NaN) for missing optional columns.
    - `get_total_cost()` stays unchanged (already uses Pg column directly).
    - `to_dataframe()` must expose all new columns.
    - Update `to_csv()` accordingly (no logic change needed — just verify header).
    - Do NOT change the sign conventions already established in the docstring.

    ---

    ## Task 2 — MILP instrumentation

    ## MILP solver (`baselines/milp_solver.py`) + replay script

    1. After the CVXPY solve, add these fields to the `history` dict:
        "Pb_charge"    → Pb_charge.value  (already computed, just expose it)
        "Pb_discharge" → Pb_discharge.value
        "b_int"        → b.value (the boolean array — keep as float for CSV)
        "P_imp"        → np.maximum(Pg_sol, 0)
        "P_exp"        → np.maximum(-Pg_sol, 0)
        "price_imp"    → price_imp  (already sliced to [:T])
        "price_exp"    → price_exp

    2. In `monitoring/run_milp_optimization_example.py`, when calling `env.monitoring_table.insert()`
      inside the replay loop, pass the new MILP-specific columns:
        Pb_charge, Pb_discharge, b_int, P_imp, P_exp, price_imp, price_exp, r_eco, reward
      Use the MILP history values directly (they are already aligned by step index k).

    3. Derive r_eco inside the replay loop from the MILP history (not the env step info),
      so the recorded economics reflect the MILP plan, not the env-replayed deviation.

    ---

    ## Task 3 — RL instrumentation

    ## RL env (`envs/base_microgrid_env.py`) + run script

    In `env.step()`, extend the `monitoring_table.insert()` call (around line 110) with:
      action_raw     → action_val (the clipped [-1,1] float, before kW scaling)
      Pb_command     → Pb_command (the kW value after scaling, before battery clamp)
      P_imp          → max(P_grid, 0)
      P_exp          → max(-P_grid, 0)
      price_imp      → price_imp (already computed as `price_imp` in step())
      price_exp      → price_exp
      r_eco          → r_eco
      r_soc          → r_soc
      reward         → reward

    No changes to the action/reward logic — only richer logging.

    ---

    ## Task 4 — new validation plots

    ## New plot: `monitoring/plot_validation_milp.py`

    Create a new 4-panel validation figure specifically for MILP diagnostics.
    The goal is to expose any constraint violation or unexpected decision at a glance.

    Panel 1 — Charge/discharge decomposition:
      Top half:  filled-stairs of Pb_discharge (kW, green, label "discharge")
      Bottom half: filled-stairs of -Pb_charge (kW, blue, label "charge")
      Overlay:   step-line of Pb (net, black dashed) for cross-check
      Title: "Pb decomposition — charge vs discharge"

    Panel 2 — Mutual exclusion check (b_int):
      Bar chart of b_int (0 or 1) at each step.
      Overlay a red scatter of steps where b_int is neither 0 nor 1 (should be empty with HiGHS).
      Color bars: green when b_int=1 (discharge), blue when b_int=0 (charge).
      Title: "b_int — mutual exclusion flag (1=discharge)"

    Panel 3 — Import/export split:
      Filled-stairs: P_imp (grey, upper half) and P_exp (dark grey, lower half, plotted negative).
      Overlay: dashed step-line of Pg for cross-check (Pg = P_imp - P_exp must hold).
      Add a secondary y-axis for the instantaneous import price (orange line, thin).
      Title: "Grid: import / export split + import price"

    Panel 4 — Economic cost decomposition:
      Stacked bar per step: import cost (red, positive) and export revenue (teal, negative).
      Overlay: cumulative net cost as a running sum line (black, right y-axis).
      Title: "Step cost / revenue and cumulative net cost"

    All panels share the x-axis (DatetimeIndex). Same palette and cross-axis convention
    as `plot_monitoring_milp.py`. Export signature:
      plot_validation_milp(monitoring_df, milp_plan_df, delta_t_minutes, show=True)
      → (fig, axes)  # 4-tuple of Axes

    ## Enhance existing `plot_monitoring_milp.py`

    Add a 5th panel below the existing 4 (currently: Pp, Pb, Pg, SoC) showing the
    step-level economic reward breakdown:
      - Bar: r_eco per step (green positive, red negative)
      - Overlay: cumulative sum of r_eco as a line (right y-axis)

    Preserve all existing panel layout and colour conventions.

    ## Add reward/cost panel to `plot_monitoring.py` (RL)

    Same addition as above (5th panel) for the RL monitoring plot, using:
      - r_eco (from monitoring_df if present, else from info dict)
      - r_soc (scatter dots in orange on violations)
      - cumulative reward as a line (right y-axis)

    ## Update `run_milp_optimization_example.py`

    After the existing two plot calls, add:
      from monitoring.plot_validation_milp import plot_validation_milp
      plot_validation_milp(monitoring_view, milp_view,
                          delta_t_minutes=env.delta_t_min, show=False)

    so the validation figure opens alongside the existing two windows.

    ---

    ## Constraints & style

    - Do NOT change reward logic, action scaling, or battery physics.
    - Backward compatibility: `insert()` must still work if called with only the original
      5 keys {pp, pl, soc, pb, pg}. New keys are optional.
    - The `b_int` column should be stored as float64 (not int) for CSV compatibility.
    - All new plot functions must follow the same pattern as the existing ones:
        def plot_*(df, ..., show=True) → (fig, axes)
      with `show=False` path for tests/CI.
    - Shared palette: keep _C_* constants from plot_power.py / plot_monitoring_milp.py.
    - Use `fig.subplots_adjust()` (not `tight_layout`) for any plot with cross-axes spines.
    - Do NOT add heavy dependencies. Only matplotlib, numpy, pandas (already in requirements).

---

# Task: Simulation pipeline refactoring — data extraction + run scripts

    ## ⚠️ ENVIRONMENT SAFEGUARD — READ FIRST, ENFORCE THROUGHOUT

    All shell commands, pip installs, python invocations, and tests in this task
    MUST run inside the micromamba environment named `stageCorse`. Before running
    ANY command, verify the active environment:

    ```bash
    micromamba run -n stageCorse python --version
    ```

    If the environment is not found or the command fails, STOP and report the error
    without attempting to fall back to another conda/micromamba environment or the
    system Python. Do NOT activate any other environment (base, py311, venv, etc.).
    All `python` calls must use `micromamba run -n stageCorse python ...` or be
    issued inside a shell where `micromamba activate stageCorse` has already been
    confirmed.

    ---

    ## Context

    The project is an RL-based Energy Management System for a microgrid
    (PV + battery + grid). The existing codebase is documented in the files
    provided. The key files to modify or create are:

    - `data/extract_pyrano_simu.py` — NEW
    - `monitoring/run_optimization_example.py` — MODIFY
    - `monitoring/run_milp_optimization_example.py` — MODIFY
    - `scripts/rl/run_exp01.sh`, `scripts/rl/run_exp02.sh` — MODIFY
    - `scripts/milp/run_exp01.sh`, `scripts/milp/run_exp02.sh` — MODIFY

    Source data exclusively used in this task: `data/Pyrano1Y_clean.csv`
    (full-year pyranometer data, unseen by the agent = simulates real deployment).

    Before creating any code check for cross-dependencies or additional needed modifications.

    ---

    ## Task 1 — Create `data/extract_pyrano_simu.py`

    ### Purpose
    Extract a contiguous block of days from `data/Pyrano1Y_clean.csv` and write
    the result to `data/pyrano_simu.csv`, overwriting any previous file.
    This simulates providing the agent with "new" real-world data for a deployment
    run, independently of training data.

    ### CLI interface
    python data/extract_pyrano_simu.py --nbD <int> [--month <1-12>] [--startDate <1-28>]
    | Argument      | Required | Default | Description                                           |
    |---------------|----------|---------|-------------------------------------------------------|
    | `--nbD`       | YES      | —       | Number of consecutive days to extract                 |
    | `--month`     | NO       | 1       | Month (1–12) where extraction starts                  |
    | `--startDate` | NO       | 1       | Day of month (1–28) where extraction starts           |

    ### Behaviour

    1. Parse the `Time` column of `Pyrano1Y_clean.csv` as datetimes.
    2. Locate the first timestamp that falls on day `startDate` of month `month`.
    3. Extract all rows from that timestamp up to (exclusive) `startDate + nbD` days later.
    4. If the extraction window exceeds the end of the available data (last timestamp
      in the CSV), raise a clear `ValueError` with a human-readable message showing
      the requested end date and the actual last date in the file. Do NOT silently
      truncate.
    5. Write the extracted slice to `data/pyrano_simu.csv`, preserving all original
      columns exactly (no column renaming, no added columns).
    6. Print a short summary to stdout:
    Extracted N rows | start: YYYY-MM-DD HH:MM | end: YYYY-MM-DD HH:MM → data/pyrano_simu.csv

    ### Validation to perform after implementation

    Run the script inside `stageCorse` for two sanity cases:

    ```bash
    # Case A — valid: 3 days from January 1
    micromamba run -n stageCorse python data/extract_pyrano_simu.py --nbD 3 --month 1 --startDate 1

    # Case B — invalid: must raise an error (request exceeds end of year)
    micromamba run -n stageCorse python data/extract_pyrano_simu.py --nbD 400 --month 1 --startDate 1
    ```

    Confirm Case A writes `data/pyrano_simu.csv` and Case B exits with a non-zero
    status and a readable error message.

    ---

    ## Task 2 — Refactor `monitoring/run_optimization_example.py` (RL)

    ### Two conceptual changes

    #### 2a. Separate `measures` from `forecast`

    In real deployment the agent receives:
    - **Measured state at step t**: actual PV production, actual load, actual prices
      → read from `--measures` CSV at row index `t`
    - **Forecast for steps t+1 … t+horizon**: predicted future values
      → read from `--forecast` CSV at rows `t+1 … t+horizon`

    For the current development phase (perfect foresight) `--measures` defaults to
    `None` and falls back silently to the forecast CSV, so no existing behaviour
    breaks. The logic must be:

    ```python
    # At step k:
    pv_now  = measures[k]   if measures is not None else forecast[k]
    load_now = measures[k]  if measures is not None else forecast[k]
    # Forecast window passed to the env observation:
    pv_fc   = forecast[k+1 : k+1+horizon_steps]
    load_fc = forecast[k+1 : k+1+horizon_steps]
    ```

    Keep using the existing fallback pattern already in the codebase:
    ```python
    if "pv_forecast" in df.columns:
        pv_fc = df["pv_forecast"].to_numpy()[...]
    else:
        pv_fc = np.array([env.pv.get_irradiance(k) for k in range(n)])
    ```

    #### 2b. Remove the visualization window concept

    Previously the script optimized for N steps but only plotted a 24h sub-window
    (`n_steps` argument). This caused confusion. The new behaviour:

    - **Plot the entire optimization run**, however many steps were simulated.
    - Remove the `--steps` / `n_steps` visualization-window argument.
    - Keep the rollout driven by the length of the forecast CSV (all rows) or
      `env.max_steps`, whichever is smaller. The user controls the duration by
      choosing how many days they extracted with `extract_pyrano_simu.py`.

    ### Updated CLI
    python -m monitoring.run_optimization_example 
    --config configs/exp01_perfect_foresight.yaml 
    --model  results/exp01_perfect_foresight/sac_model.zip 
    --forecast data/pyrano_simu.csv 
    [--measures data/pyrano_simu.csv]   # optional, defaults to forecast
    --out monitoring/runs/rl_exp01_monitoring_table.csv

    The `--steps` argument must be removed (or kept as a hidden deprecated no-op
    if backward compatibility is needed for CI, but not advertised).

    ---

    ## Task 3 — Refactor `monitoring/run_milp_optimization_example.py` (MILP)

    Apply the same two changes as Task 2 (measures/forecast split + remove
    visualization window). Additionally:

    - The `--offset` argument introduced to work around the window concept is no
      longer needed; remove it or deprecate it silently.
    - The MILP solver receives the full forecast horizon; the replayed monitoring
      and both plots must cover the entire solved trajectory, not a 24h slice.

    ### Updated CLI
    python -m monitoring.run_milp_optimization_example 
    --config configs/exp01_perfect_foresight.yaml 
    --forecast data/pyrano_simu.csv 
    [--measures data/pyrano_simu.csv]   # optional, defaults to forecast
    --out  monitoring/runs/milp_exp01_monitoring_table.csv 
    --plan-out monitoring/runs/milp_exp01_plan.csv

    ---

    ## Task 4 — Update bash scripts in `scripts/`

    Update the four scripts below to pass `--forecast data/pyrano_simu.csv`
    instead of any previous hardcoded path. The scripts must also remove any
    `--steps` or `--offset` arguments they previously passed.

    Files:
    - `scripts/rl/run_exp01.sh`
    - `scripts/rl/run_exp02.sh`
    - `scripts/milp/run_exp01.sh`
    - `scripts/milp/run_exp02.sh`

    ---

    ## Task 5 — Cross-check and validation

    After all modifications, perform the following checks **inside `stageCorse`**:

    ### 5a. Import check
    ```bash
    micromamba run -n stageCorse python -c "
    from monitoring.run_optimization_example import run
    from monitoring.run_milp_optimization_example import run
    from data.extract_pyrano_simu import main
    print('All imports OK')
    "
    ```

    ### 5b. Extraction smoke test
    ```bash
    micromamba run -n stageCorse python data/extract_pyrano_simu.py \
        --nbD 2 --month 1 --startDate 1
    ```
    Verify `data/pyrano_simu.csv` exists and has the expected number of rows
    (nbD × rows_per_day — check against the timestep resolution in the CSV).

    ### 5c. CLI --help check
    ```bash
    micromamba run -n stageCorse python -m monitoring.run_optimization_example --help
    micromamba run -n stageCorse python -m monitoring.run_milp_optimization_example --help
    ```
    Confirm `--measures` appears, `--steps` / `--offset` do NOT appear (or are
    clearly marked deprecated), and `--forecast` is present.

    ### 5d. Dry-run logic check (no model needed)
    Without running a full experiment (which requires a trained model), verify the
    `_build_forecast_df`-equivalent function correctly separates measures from
    forecast by unit-testing the slice logic:

    ```bash
    micromamba run -n stageCorse python -c "
    import numpy as np, pandas as pd
    # Simulate 10 rows of forecast data
    n = 10; horizon = 3
    data = np.arange(n, dtype=float)
    # At step k=2: measures[2], forecast[3:6]
    k = 2
    measured_now = data[k]
    fc_window = data[k+1 : k+1+horizon]
    assert measured_now == 2.0
    assert list(fc_window) == [3.0, 4.0, 5.0], f'Got {list(fc_window)}'
    print('Slice logic OK')
    "
    ```

    ---

    ## Style and quality requirements

    - Follow the existing docstring style (Google-style, English comments, English for new docstrings).
    - Do not import any library not already in `requirements.txt` or the standard
      library.
    - Preserve all existing function signatures and return types in the `run()`
      functions; add new parameters as keyword arguments with safe defaults so
      existing call sites (e.g. `experiments/run_experiment.py`) do not break.
    - Do not touch `envs/`, `agents/`, `baselines/`, or `evaluation/` — this
      task is strictly limited to `data/`, `monitoring/`, and `scripts/`.


# Correction Bug #2 : curtailment PV explicite

    ## ⚠️ ENVIRONNEMENT D'EXÉCUTION (RÈGLE ABSOLUE)

    **TOUTE** exécution Python (imports, scripts de vérification, MILP, rollouts, tests)
    doit se faire **exclusivement** dans l'environnement micromamba `stageCorse`, via :

    ```bash
    micromamba run -n stageCorse python <...>
    ```

    Il est **INTERDIT** d'utiliser ou de créer un autre environnement : pas de `venv`,
    pas de `conda base`, pas de `python` système, pas de `pip install` ailleurs.
    L'environnement `stageCorse` **existe déjà** — ne le recrée pas, ne le vérifie pas,
    n'installe rien de nouveau. Si une commande Python est lancée hors de
    `micromamba run -n stageCorse`, c'est une erreur.

    ---

    ## Contexte du bug

    Le bilan de puissance peut être physiquement violé quand le surplus PV dépasse la
    limite d'export et que la batterie ne peut plus absorber :

    - **Côté env** (`envs/base_microgrid_env.py`) : `P_grid = clip(Pl - Pp - Pb, …)` est
      un *fail-soft* qui fait disparaître silencieusement l'excès (résidu Kirchhoff ≠ 0).
    - **Côté MILP** (`baselines/milp_solver.py`) : l'égalité `Pg == Pl - Pp - Pb` couplée
      à `P_exp ≤ max_exp` rend le problème **infaisable** (*fail-hard*, `RuntimeError`)
      dès qu'un step exige `-(Pp - Pl + Pb) > max_exp`.

    Remède commun : introduire une variable explicite de **curtailment PV** `Pcurt ≥ 0`,
    de sens physique direct (l'onduleur écrête le PV).

    ## Formule de curtailment (⚠️ attention au signe)

    `P_grid_raw = load - pv - Pb_effective` (bilan non borné ; `< 0` ⇒ surplus à exporter).

    ```
    Pcurt = max(0, pv + Pb_effective - load - max_export)
          = max(0, -P_grid_raw - max_export)
    ```

    **Propriété clé à exploiter pour un code propre :**
    `clip(P_grid_raw + Pcurt, -max_export, max_import) ≡ clip(P_grid_raw, -max_export, max_import)`.
    Donc `P_grid` et `r_eco` sont **identiques** dans les deux modes ; seul `r_curt`
    diffère. C'est une ablation propre (différence uniquement dans la reward).

    ## Deux modes pour l'agent RL (nouvelle option de config)

    Nouvelle clé `grid.curtailment` ∈ `{"clip", "penal"}`, défaut `"clip"` :

    - **`clip`** : revenu d'export naturellement plafonné par le clip
      (`revenu = price_export × max_export × Δt`). Pas de terme de reward supplémentaire.
      Numériquement identique au comportement actuel côté export, mais `Pcurt` est
      désormais **calculé et tracé**.
    - **`penal`** : même `P_grid` / `r_eco` que `clip`, plus une pénalité explicite
      `r_curt = -price_export × Pcurt × Δt` ajoutée à la reward (l'énergie gaspillée
      coûte autant qu'un kWh non vendu). Pas de double comptage : `r_eco` est déjà capé
      par le clip ; `r_curt` pénalise l'énergie *au-delà* du cap.

    Le MILP n'a **pas** de mode : il utilise toujours la variable `Pcurt` explicite,
    objectif inchangé (le curtailment est « gratuit » dans l'objectif et ne s'active que
    pour la faisabilité — c'est la référence optimale).

    ---

    ## MODIFICATIONS REQUISES

    ### 1. `monitoring/monitoring_table.py`
    - Ajouter une constante `COL_PCURT` **à la fin** des indices de colonnes (après
      `COL_PB_COMMAND = 17`, donc `COL_PCURT = 18`). **Ne pas** insérer au milieu :
      renuméroter les 12 constantes existantes serait risqué. L'ordre CSV légèrement
      non-canonique (colonne partagée placée après les colonnes RL-only) est un coût
      cosmétique acceptable.
    - Ajouter `"Pcurt"` à la fin de `COLUMN_NAMES` (cohérent avec l'indice 18).
      `NUM_COLUMNS` se met à jour automatiquement.
    - Ajouter `"pcurt": COL_PCURT` dans `_OPTIONAL_KEYS` (curtailment = colonne optionnelle,
      NaN si absente — **ne pas** la mettre dans `_REQUIRED_KEYS`).
    - Mettre à jour le docstring de tête (le compte de colonnes est déjà périmé) :
      documenter `Pcurt [kW, ≥0]` comme colonne partagée.

    ### 2. `envs/base_microgrid_env.py`
    - Dans `__init__`, lire et valider le mode :
      ```python
      self.curtailment_mode = config["grid"].get("curtailment", "clip")
      if self.curtailment_mode not in ("clip", "penal"):
          raise ValueError(
              f"Unknown grid.curtailment={self.curtailment_mode!r}; use 'clip' or 'penal'."
          )
      ```
    - Dans `step()`, remplacer le bloc bilan/reward actuel par :
      ```python
      # Bilan brut (non borné). P_grid_raw < 0 ⇒ surplus à exporter.
      P_grid_raw = load_t - pv_t - Pb_effective

      # Curtailment PV côté export : l'onduleur écrête le PV quand le surplus
      # dépasse la limite d'export et que la batterie ne peut plus absorber.
      # Signe : Pcurt = max(0, pv + Pb - load - max_export) = max(0, -P_grid_raw - max_export)
      Pcurt = max(0.0, -P_grid_raw - self.max_export_kw)

      # Côté import : vraie limite réseau (charge non satisfaite), conservée en clip.
      P_grid = float(np.clip(P_grid_raw, -self.max_export_kw, self.max_import_kw))

      price_imp = self.price_signal.get_import_price(self.step_index)
      price_exp = self.price_signal.get_export_price(self.step_index)
      r_eco = -(
          price_imp * max(P_grid, 0.0)
          - price_exp * max(-P_grid, 0.0)
      ) * self.delta_t_h

      r_soc = -self.sigma_soc * (
          max(0.0, self.soc_safe_min - new_soc)
          + max(0.0, new_soc - self.soc_safe_max)
      )

      # "clip" : revenu naturellement capé (aucun terme additionnel).
      # "penal" : pénalise en plus le PV gaspillé au prix d'export.
      if self.curtailment_mode == "penal":
          r_curt = -price_exp * Pcurt * self.delta_t_h
      else:  # "clip"
          r_curt = 0.0

      reward = r_eco + r_soc + r_curt
      ```
    - Dans l'appel `self.monitoring_table.insert(...)`, ajouter la clé `"pcurt": float(Pcurt),`.
      La colonne `reward` insérée doit refléter le total **incluant `r_curt`** (c'est déjà
      le cas si on insère la variable `reward` ci-dessus).
    - Ajouter `"Pcurt": Pcurt,` au dict `info` retourné par `step()`.

    ### 3. `baselines/milp_solver.py`
    - Ajouter la variable et corriger le bilan :
      ```python
      Pb = cp.Variable(T)
      Pg = cp.Variable(T)
      P_imp = cp.Variable(T, nonneg=True)
      P_exp = cp.Variable(T, nonneg=True)
      Pcurt = cp.Variable(T, nonneg=True)   # NEW : curtailment PV explicite
      soc = cp.Variable(T + 1)

      constraints = []
      constraints.append(soc[0] == init_soc)

      for t in range(T):
          # bilan corrigé : Pp_used = pv - Pcurt
          constraints.append(Pg[t] == load_vals[t] - (pv_vals[t] - Pcurt[t]) - Pb[t])
          constraints.append(Pg[t] == P_imp[t] - P_exp[t])
          constraints.append(Pcurt[t] <= pv_vals[t])   # pas plus que le PV disponible
      ```
    - **Objectif inchangé** (`minimize cost_import - revenue_export`).
    - Ajouter dans le dict `history` : `"Pcurt": np.asarray(Pcurt.value, dtype=np.float64),`.

    ### 4. `monitoring/run_milp_optimization_example.py`
    - Dans le `mt.insert(k, {...})` qui réécrit les colonnes côté MILP, ajouter :
      `"pcurt": float(hist["Pcurt"][k]),` pour que le CSV env-replay reflète le
      curtailment planifié par le MILP.

    ### 5. Configs (`configs/`)
    - Ajouter explicitement la ligne `curtailment: clip` dans la section `grid:` de
      **tous** les configs existants (`exp01_perfect_foresight.yaml`,
      `exp01_bis_perfect_foresight.yaml`, `exp01_bis_gamma.yaml`,
      `exp02_variable_price.yaml`) — pour rendre l'option visible/explicite.
    - Créer **au moins un config jumeau** pour la comparaison, p. ex.
      `configs/exp02_curtail_penal.yaml` = copie de `exp02_variable_price.yaml` avec
      `experiment.name: exp02_curtail_penal` et `grid.curtailment: penal`.

    ---

    ## MODIFICATIONS OPTIONNELLES (clairement séparées, ne pas casser le défaut)

    - **`Pcurt` dans l'observation** : ajouter une clé `grid.observe_curtailment`
      (défaut `false`). Quand `true`, appliquer **identiquement aux deux modes** (pour garder
      l'ablation propre) : initialiser `self._last_pcurt = 0.0` dans `reset()`, l'ajouter à la
      fin du vecteur d'obs dans `_get_obs()`, mettre à jour `self._last_pcurt = float(Pcurt)`
      en fin de `step()`, et incrémenter `obs_dim` de 1 dans `__init__`. ⚠️ Cela change
      `obs_dim` et **impose un ré-entraînement** ; laisser le défaut `false` pour rester
      rétro-compatible avec les modèles `.zip` existants.
    - **Métrique de curtailment** : optionnellement, enregistrer `info["Pcurt"]` dans
      `history` côté `agents/sac_agent.py::evaluate_sac` et ajouter
      `energy_curtailed_kwh = sum(Pcurt) * delta_t_h` dans `evaluation/metrics.py`.

    ---

    ## CONVENTIONS À RESPECTER (ne pas casser)

    - Convention de signe inchangée : `Pb > 0` = décharge, `Pb < 0` = charge ;
      `Pg > 0` = import, `Pg < 0` = export.
    - **Ne pas renommer** `PVSource.get_irradiance` (convention imposée par le superviseur).
    - `Pcurt ≥ 0` et `Pcurt ≤ Pp` en permanence.
    - En mode `clip`, `r_curt = 0` ; le défaut `clip` doit rester numériquement identique
      au comportement actuel côté export (modèles existants encore exploitables en inférence).

    ---

    ## VÉRIFICATIONS À EXÉCUTER (toutes dans `stageCorse`)

    1. Imports sans erreur :
      ```bash
      micromamba run -n stageCorse python -c "import envs, baselines, monitoring, evaluation"
      ```
    2. Self-test batterie existant toujours vert :
      ```bash
      micromamba run -n stageCorse python -m envs.components.battery
      ```
    3. Écrire `scratch/verify_curtailment.py` et l'exécuter :
      ```bash
      micromamba run -n stageCorse python scratch/verify_curtailment.py
      ```
      Ce script doit (sans nouvelle donnée : forcer le régime en abaissant
      `grid.max_export_kw`, p. ex. à `0.5`, sur le config exp01 chargé puis modifié
      en mémoire, et/ou en injectant un step à fort PV) **asserter** :
      - **Bilan corrigé** sur tous les steps d'un rollout :
        `abs((Pl - (Pp - Pcurt) - Pb) - Pg) ≤ 1e-6` (résidu nul, plus de fuite silencieuse).
      - `Pcurt ≥ 0` et `Pcurt ≤ Pp` partout.
      - **Égalité clip ≡ penal** sur `P_grid` et `r_eco` step par step, et
        `reward_penal ≤ reward_clip` (strictement inférieure dès que `Pcurt > 0`),
        l'écart valant exactement `price_export × Pcurt × Δt`.
      - **MILP faisable** sur le cas stressé (`max_export` faible) :
        `run_milp(...)["solver_status"] in ("optimal", "optimal_inaccurate")`,
        aucun `RuntimeError`, et `Pcurt` MILP `≥ 0`, `≤ pv_vals`.
      - La `MonitoringTable` expose bien la colonne `Pcurt` après `to_dataframe()`.

    Ne considérer la tâche terminée que si toutes les assertions passent **dans
    `micromamba run -n stageCorse`**.

    ---

    ## RAPPEL FINAL

    Toutes les commandes Python passent par `micromamba run -n stageCorse python ...`.
    Aucun autre environnement n'est autorisé. `stageCorse` existe déjà — ne pas le créer
    ni le vérifier.

## Clean load and extract training/test data
    J'ai reçu le csv d'un profil de charges realiste et j'ai besoin de lui appliquer un pretraitement pour pouvoir l'utiliser dans l'entrainement de l'agent RL

    Ainsi j'ai besoin de deux taches, la premiere le nettoyage du profil des charges et ensuit l'extraction des données nécessaire a l'entrainement.

    Tache 1 - pretraitement des charges  : 

    Le csv de charges data/energy_community.csv possede les colonnes suivantes :  ['Date', 'Prod_pv', 'Conso_resident', 'Conso_bureau', 'Conso_ecole']
    et est fourni avec un pas de temps de 1h et dont toute les valeurs sont des kW (a l'exception des dates)

    ainsi un exemple d'une ligne succesives est : 
    01/01/2025 06:00	2.29	27.25	4.26	0.3

    Ce que l'exemple veut dire : 

    Le 01/01/2025 de 6:00 à 7:00 (utilisation pendant 1h) la consommation residente a était de 27.25 kW,  la consommation bureau a était de 4.26 kW et  la consommation ecole a était de 0.3 kW (tot = 31.81 kW).

    Ainsi une division des charges a un pas de 15 min pourrais correspondre a des valeurs comme : 

    6:00 -> tot : 31.81/4
    6:15 -> tot : 31.81/4
    6:30 -> tot : 31.81/4
    6:45-> tot : 31.81/4 
    et ansi la somme des 4*(31.81/4) = 31.81

    Evidemment diviser par 4 ne correspond pas a l'interpolation, mais on pourrais, peut etre, considerer des options natives de DataFrame.interpolate() comme method = 'time' (**a verifier que cela marche dans notre cas**), l'idée c'est que la somme des données entre chaque heure correspond a la charge originale (ou du moins avec une marge d'erreur negligeable epsilon)

    Ainsi j'ai besoin d'un fichier data/clean_load.py qui reçoit le fichier data/energy_community.csv et qui genere un nouveau csv data/load_profile.csv qui tout d'abord se debarrase de la colonne 'Prod_PV' puis dans la colonne Date on change l'année 2025 par 2023 au format 2023-01-01 00:00:00 et le nom de la colonne "Date"a "Time" (i.e. même format que les données data/Pyrano*).
    Ensuite il faudra appliquer le resamplig selon un pas de temps qu'on pourras choisir comme variable globale au debut du fichier (dans ce cas on choisi le pas de temps delta t =  15 min). Finalement le nouveau CSV devra avoir une nouvelle colonne 'Conso_tot' qui correspond a la somme des 'Conso_resident', 'Conso_bureau', 'Conso_ecole'

    Tache 2 (data extraction for training) : 
    L'etat du projet evolve, maintenant on disposera d'un fichier d'irradiance pv (qu'on utilisera comme prevision pour les prevision parfaites) ceci est deja implementer avec les données data/Pyrano, mais aussi d'un fichier load_profile.csv (resultat tache 1) avec un profil (et donc prevision de loads), ceci pourra etre fourni a l'agent lors des configs dans la partie data et donc pourra etre reutiliser par la fonction load_forecast. 

    Ainsi pour realiser cette extraction on utilisera deux fichiers data/Pyrano1Y_clean.csv pour l'irradiance solaire et data/load_profile.csv pour le profil de charges. On peut continuer avec la même idée d'implementation pour le choix de la colonne utile, i.e. dans configs/* on chosi : 
    `pv_column` avec : 
    data:
      pv_column: "Global30_kW"
    et maintenant on pourra rajouter load_column : "Conso_tot"

    Ainsi pour l'extraction on pourra reutiliser la logique de data/extract_pyrano_simu.py, i.e. on fourni un CLI python data/extract_pyrano_simu.py --nbD <int> [--month <1-12>] [--startDate <1-28>] pour indiquer le nombre de jours, et la date de debut pour l'exctraction. CEPENDANT il faudra faire attention a l'extraction des données dans data/Pyrano1Y_clean.csv car les données extraits doivent coincider avec les timestamps de load_profile.csv, i.e. si dans load_profile.csv on a utiliser un pas de 15 min les données extraits de Pyrano1Y_clean.csv seront que ceux qui coincides avec load_profile.csv; pour ceci les deux fichier partagerons le meme format d'heure (decidé dans la tache 1) ainsi il suffira d'utiliser la colonne "Time" comme clé. 

    Le resultat de cette extraction doit etre 2 fichier csv : irradiance_training.csv and load_training.csv, ensuite ces fichiers seront utiliser dans l'entrainement du RL mais le split train/test est deja gerer donc pas de changement a rajouter.

    NOTE : Verifier les implementations existantes et utilisation des données csv pour valider les nouvelles idees de data 
    NOTE IMPORTANTE : tout test devra etre verifie EXCLUSIVEMENT dans micromamba stageCorse, tout autre env conda/micromamba est INTERDIT d'etre utiliser


Plan — Diagnostiquer & observer l'anomalie RL < MILP (exp_testCluster)
=======================================================================

    Context

    Sur results/exp_testCluster (config configs/expTestCluster.yaml, RL SAC entraîné sur une
    config sous-dimensionnée : import réseau 20 kW, batterie 40 kWh), le RL affiche un
    net_cost inférieur au MILP (244.89 € vs 257.51 €, -4.9 %), ce qui est contre-intuitif
    puisque le MILP est censé être optimal.

    Cause racine identifiée (analyse de metrics.json + lecture du pipeline) :
    evaluation/metrics.py:compute_metrics calcule net_cost uniquement depuis P_grid, qui
    est plafonné à 20 kW. La charge non servie au-delà (la puissance fantôme Pph)
    n'apparaît nulle part dans le coût. Or sur le test set ~47 % des pas requièrent du phantom :
    - MILP : phantom = 1258.95 kWh (objectif pénalise le phantom à 1000 €/kWh → il priorise de
    servir la charge, importe 2158 kWh).
    - RL : phantom ≈ 1505 kWh (déduit de total_reward ≈ -1000 × E_phantom) → laisse ~246 kWh
    de charge en plus non servie, importe 2097 kWh (-61 kWh ≈ -12.3 € à ~0.2 €/kWh), ce qui
    explique exactement l'écart de net_cost (-12.62 €).

    ➜ Le RL n'est pas meilleur : il « triche » en laissant ~20 % de charge en plus non servie.
    net_cost et self_consumption_rate récompensent ce comportement ; les total_reward sont en
    plus calculés différemment (MILP exclut la pénalité phantom, RL l'inclut). Note séparée : la PV
    est sur-dimensionnée (surface_m2=1600 × eta_ref=0.20 ≈ 320 kWp, max 361.6 kW) → curtailment
    massif du surplus, mais cela affecte les deux approches identiquement et n'est pas la cause
    de l'anomalie net_cost (qui vient du déficit/phantom le soir).

    Objectif : (1) rendre la comparaison honnête (métriques phantom-aware) et (2) fournir une
    méthode d'observation (monitoring + plots) rejouable sur le test split exact ET une
    fenêtre de simulation contiguë, en réutilisant les plots phantom-aware déjà en place.

    Décisions utilisateur : données = les deux (test split + simu) ; visualisation =
    réutiliser les plots par modèle existants (pas de nouveau plot combiné) ; métriques =
    oui, phantom-aware. Tests dans l'env micromamba stageCorse uniquement.

    Partie 1 — Métriques phantom-aware (comparaison honnête)

    - evaluation/metrics.py:compute_metrics — ajouter un paramètre phantom_penalty: float = 1e3.
    Quand "P_phantom" in history, calculer et toujours retourner :
      - phantom_energy_kwh, phantom_steps
      - served_load_ratio = (total_load_energy - phantom_energy_kwh) / total_load_energy (garde div0)
      - net_cost_adjusted = net_cost + phantom_penalty * phantom_energy_kwh (coût VOLL de la charge
    non servie — la seule grandeur comparable entre politiques à phantom différent).
    Sans P_phantom : phantom_energy_kwh=0, served_load_ratio=1.0, net_cost_adjusted=net_cost.
    - agents/sac_agent.py:evaluate_sac — ajouter "P_phantom": [] (et "Pcurt") à history,
    append(info["P_phantom"]) dans la boucle, et passer
    phantom_penalty=env.cfg["grid"].get("phantom_penalty", 1e3) à compute_metrics.
    (info["P_phantom"]/info["Pcurt"] existent déjà — base_microgrid_env.py:230-231.)
    - baselines/milp_solver.py — rendre total_reward comparable : la history["reward"]
    (l. 113) ne contient que r_eco_sol ; ajouter la pénalité phantom
    history["reward"] = r_eco_sol - phantom_penalty * P_phantom_sol * delta_t_h. Passer
    phantom_penalty à compute_metrics et supprimer le calcul manuel dupliqué de
    phantom_energy_kwh/phantom_steps (l. 131-133, désormais fournis par compute_metrics).
    - evaluation/compare.py — ajouter au dict + au tableau imprimé + à comparison.json :
    *_phantom_energy_kwh, *_served_load_ratio, *_net_cost_adjusted, et un
    relative_gap_adjusted calculé sur net_cost_adjusted. Quand max(phantom RL, MILP) > 0,
    imprimer un avertissement : « net_cost brut non comparable (charge non servie) — voir
    net_cost_adjusted ».

    Partie 2 — Monitoring + plots sur test split ET fenêtre simu (réutilisation)

    Les scripts monitoring/run_optimization_example.py (RL) et run_milp_optimization_example.py
    (MILP) produisent déjà les plots phantom-aware (plot_power, plot_monitoring,
    plot_monitoring_milp) mais consomment le CSV entier sans appliquer le split. Deux ajouts :

    - run_milp_optimization_example.py : ajouter --load-csv (param load_csv dans run() +
    arg CLI) et le passer à _build_deployment_env(config_path, deployment_csv, load_csv) (le 3e
    paramètre est déjà supporté côté helper). Met le MILP au niveau du RL pour consommer
    load_simulation.csv.
    - Option --split {full,test,train} (défaut full) sur les deux scripts. Ajouter dans
    run_optimization_example.py un helper _build_split_env(config_path, split) (importé par le
    script MILP, qui importe déjà _build_deployment_env de là) : appelle
    envs.registry.make_env(config_path) → (train_env, test_env, cfg), sélectionne l'env, fixe
    env.max_steps = env.pv.n_steps (couverture complète comme en déploiement), et retourne
    (env, cfg). run() branche : split == "full" → _build_deployment_env (CSV/--forecast/
    --load-csv) ; split in {test, train} → _build_split_env (données = split registry, qui
    reproduit exactement le test set de comparison.json).

    Usage résultant (env stageCorse) :
    - Test split exact : --config configs/expTestCluster.yaml --split test (RL ajoute
    --model results/exp_testCluster/sac_model.zip).
    - Fenêtre simu contiguë : --config configs/expTestCluster.yaml --forecast data/irradiance_simulation.csv --measures data/irradiance_simulation.csv --load-csv
    data/load_simulation.csv (les CSV simu existent déjà ; ré-extraire si besoin via
    data/extract_pyrano_simu.py --nbD 7 --month 2 --startDate 1 --usage simu).

    Chaque run écrit la table de monitoring (colonne Pph peuplée des deux côtés depuis les travaux
    précédents) et ouvre les plots où la couche violette Phantom rend la charge non servie visible.

    Partie 3 — Note de dimensionnement PV (recommandation, pas de modif auto)

    Signaler dans le rapport/analyse que surface_m2=1600/eta_ref=0.20 produit ~320 kWp (max
    361.6 kW) face à un réseau de 20 kW : à revoir (surface/eta réalistes ou limites réseau) pour des
    expériences cohérentes. Ne pas modifier la config sans validation (cela change la sémantique
    de l'expérience).

    Vérification (env micromamba stageCorse uniquement)

    1. Unit : ajouter un test (tests/) pour compute_metrics avec un history contenant
    P_phantom>0 → vérifier phantom_energy_kwh, served_load_ratio < 1, et
    net_cost_adjusted == net_cost + phantom_penalty*phantom_energy_kwh. Relancer
    pytest tests/ -q.
    2. Métriques honnêtes : recharger results/exp_testCluster/sac_model.zip et relancer une
    comparaison (script ad hoc ou experiments/run_experiment.py allégé) → confirmer que
    net_cost_adjusted(MILP) < net_cost_adjusted(RL) (le MILP redevient meilleur) et que
    l'avertissement phantom s'affiche.
    3. Observation test split : python -m monitoring.run_optimization_example --config configs/expTestCluster.yaml --model results/exp_testCluster/sac_model.zip --split test --out
    monitoring/runs/rl_testsplit.csv puis l'équivalent MILP --split test → vérifier
    Pph>0 dans les CSV et la bande violette dans les plots.
    4. Observation simu : mêmes scripts avec --forecast/--measures data/irradiance_simulation.csv --load-csv data/load_simulation.csv → plots contigus propres, comparer RL vs MILP côte
    à côte.
    5. Backend Agg (MPLBACKEND=Agg) + --no-show pour la validation non interactive ; inspecter
    les CSV (Pph, métriques) pour confirmer les écarts RL/MILP.
