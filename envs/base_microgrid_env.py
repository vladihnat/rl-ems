"""Gymnasium environment for microgrid energy management.

Action: normalized battery command in [-1, 1].
Observation: temporal features + system state + economic signals + PV forecast.
Reward: economic cost/revenue + SoC penalty.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.components.battery import BatteryModel
from envs.components.load import LoadModel
from envs.components.price_signal import PriceSignal
from envs.components.pv_source import PVSource
from monitoring.monitoring_table import MonitoringTable


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        pv_source: PVSource,
        load_model: LoadModel,
        battery: BatteryModel,
        price_signal: PriceSignal,
        config: dict,
    ):
        super().__init__()
        self.pv = pv_source
        self.load = load_model
        self.battery = battery
        self.price_signal = price_signal
        self.cfg = config

        self.delta_t_min = config["time"]["delta_t_min"]
        self.delta_t_h = self.delta_t_min / 60.0
        self.horizon_steps = int(config["time"]["horizon_h"] * 60 / self.delta_t_min)

        self.max_import_kw = config["grid"]["max_import_kw"]
        self.max_export_kw = config["grid"]["max_export_kw"]

        # Pénalité de puissance fantôme (charge non servie) — identique à celle du MILP
        # (baselines/milp_solver.py) pour que RL et MILP modélisent la même physique.
        self.phantom_penalty = config["grid"].get("phantom_penalty", 1e3)

        self.curtailment_mode = config["grid"].get("curtailment", "clip")
        if self.curtailment_mode not in ("clip", "penal"):
            raise ValueError(
                f"Unknown grid.curtailment={self.curtailment_mode!r}; use 'clip' or 'penal'."
            )

        self.sigma_soc = config["reward"]["sigma_soc"]
        self.soc_safe_min = config["reward"]["soc_safe_min"]
        self.soc_safe_max = config["reward"]["soc_safe_max"]

        self.max_charge_kw = config["battery"]["max_charge_kw"]
        self.max_discharge_kw = config["battery"]["max_discharge_kw"]

        self._load_forecast_in_obs = config.get("observation", {}).get("load_forecast", False)
        self._spread_penalty = config["reward"].get("spread_penalty", False)
        self._sigma_bat = config["reward"].get("sigma_bat", 0.0)
        self._random_soc = config.get("training", {}).get("random_soc", False)
        self._pb_max = max(self.max_charge_kw, self.max_discharge_kw)

        price_forecast_dim = self.horizon_steps if self.price_signal.has_forecast else 0
        load_forecast_dim = self.horizon_steps if self._load_forecast_in_obs else 0
        obs_dim = 9 + self.horizon_steps + load_forecast_dim + price_forecast_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.max_steps = self.pv.n_steps - self.horizon_steps
        self.step_index = 0

        # Monitoring buffer — accumulates one row per call to step() during a
        # post-training rollout. See monitoring/monitoring_table.py.
        # Allocated lazily on reset() once we know the rollout length.
        self.monitoring_table: MonitoringTable | None = None

    def _get_obs(self) -> np.ndarray:
        h_sin, h_cos, d_sin, d_cos = self.pv.get_temporal_features(self.step_index)
        soc = self.battery.soc
        load_t = self.load.get_load(self.step_index)
        pv_t = self.pv.get_irradiance(self.step_index)
        pv_forecast = self.pv.get_forecast(self.step_index, self.horizon_steps)
        p_imp = self.price_signal.get_import_price(self.step_index)
        p_exp = self.price_signal.get_export_price(self.step_index)

        parts = [
            np.array([h_sin, h_cos, d_sin, d_cos], dtype=np.float32),
            np.array([soc, load_t], dtype=np.float32),
            np.array([p_imp, p_exp], dtype=np.float32),
            np.array([pv_t], dtype=np.float32),
            pv_forecast,
        ]
        if self._load_forecast_in_obs:
            parts.append(self.load.get_forecast(self.step_index, self.horizon_steps))
        if self.price_signal.has_forecast:
            parts.append(self.price_signal.get_import_forecast(self.step_index, self.horizon_steps))

        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        if self._random_soc:
            eps = 0.05
            self.battery.soc = float(self.np_random.uniform(
                self.battery.soc_min + eps,
                self.battery.soc_max - eps,
            ))
        else:
            self.battery.reset()

        # (Re)allocate the monitoring buffer for this rollout.
        # MATLAB equivalent: MPC/simu.m:15  obj.MicroGrid.Monitoring = nan(nPoints, 6).
        start_ts = pd.Timestamp(self.pv.timestamps[0])
        self.monitoring_table = MonitoringTable(
            n_steps=self.max_steps,
            start_time=start_ts,
            delta_t_min=self.delta_t_min,
        )
        return self._get_obs(), {}

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))

        if action_val < 0:
            Pb_command = action_val * self.max_charge_kw
        else:
            Pb_command = action_val * self.max_discharge_kw

        Pb_effective, new_soc = self.battery.step(Pb_command, self.delta_t_h)

        pv_t = self.pv.get_irradiance(self.step_index)
        load_t = self.load.get_load(self.step_index)

        # Bilan brut (non borné). P_grid_raw < 0 ⇒ surplus à exporter.
        P_grid_raw = load_t - pv_t - Pb_effective

        # Curtailment PV côté export : l'onduleur écrête le PV quand le surplus
        # dépasse la limite d'export et que la batterie ne peut plus absorber.
        Pcurt = max(0.0, -P_grid_raw - self.max_export_kw)

        P_grid = float(np.clip(P_grid_raw, -self.max_export_kw, self.max_import_kw))

        # Puissance fantôme = charge non servie au-delà de la limite d'import (ce que le
        # clip ci-dessus écartait silencieusement). Pénalisée comme dans le MILP.
        P_phantom = max(0.0, P_grid_raw - self.max_import_kw)
        r_phantom = -self.phantom_penalty * P_phantom * self.delta_t_h

        price_imp = self.price_signal.get_import_price(self.step_index)
        price_exp = self.price_signal.get_export_price(self.step_index)
        r_eco = -(
            price_imp * max(P_grid, 0.0)
            - price_exp * max(-P_grid, 0.0)
        ) * self.delta_t_h

        # SoC penality is dead code when soc_safe_min = soc_min and soc_safe_max = soc_max (clip in battery.step) 
        r_soc = -self.sigma_soc * (
            max(0.0, self.soc_safe_min - new_soc)
            + max(0.0, new_soc - self.soc_safe_max)
        )

        
        if self.curtailment_mode == "penal":
            r_curt = -price_exp * Pcurt * self.delta_t_h
        else:  # "clip"
            r_curt = 0.0

        # r_bat_power = -self._sigma_bat * (Pb_effective / self._pb_max) ** 2

        # Ce que ça détecte : la batterie est en train de décharger (Pb_effective > 0) alors qu'il y a un surplus PV (pv_t >
        # load_t). C'est une décision sous-optimale : la batterie gaspille de l'énergie stockée au lieu de laisser le PV couvrir
        # la charge directement.
        r_spread = 0.0
        if self._spread_penalty:
            pv_surplus = max(0.0, pv_t - load_t)
            if Pb_effective > 0.0 and pv_surplus > 0.0:
                r_spread = -(price_imp - price_exp) * min(Pb_effective, pv_surplus) * self.delta_t_h

        # reward = r_eco + r_soc + r_curt + r_bat_power + r_spread
        reward = r_eco + r_soc + r_curt +  r_spread + r_phantom

        # reward = r_eco + r_soc + r_curt

        # Record this decision step in the monitoring buffer BEFORE incrementing
        # step_index, so row k contains the action+state at decision step k.
        # MATLAB equivalent: PMS/follow.m:3  obj.MicroGrid.insert_monitoring_data(state).
        # SoC is stored in percent (0-100); battery.soc is a fraction in [0,1].
        if self.monitoring_table is not None and self.step_index < self.max_steps:
            self.monitoring_table.insert(
                self.step_index,
                {
                    "pp": float(pv_t),
                    "pl": float(load_t),
                    "soc": float(new_soc) * 100.0,
                    "pb": float(Pb_effective),
                    "pg": float(P_grid),
                    "action_raw": float(action_val),
                    "pb_command": float(Pb_command),
                    "p_imp":     float(max(P_grid, 0.0)),
                    "p_exp":     float(max(-P_grid, 0.0)),
                    "price_imp": float(price_imp),
                    "price_exp": float(price_exp),
                    "r_eco":     float(r_eco),
                    "r_soc":     float(r_soc),
                    # "r_bat_power": float(r_bat_power),
                    "r_spread":  float(r_spread),
                    "r_phantom": float(r_phantom),
                    "reward":    float(reward),
                    "pcurt":     float(Pcurt),
                    "pph":       float(P_phantom),
                },
            )

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False

        info = {
            "Pb_effective": Pb_effective,
            "P_grid": P_grid,
            "soc": new_soc,
            "r_eco": r_eco,
            "r_soc": r_soc,
            # "r_bat_power": r_bat_power,
            "r_spread": r_spread,
            "r_phantom": r_phantom,
            "P_phantom": P_phantom,
            "Pcurt": Pcurt,
            "pv_t": pv_t,
            "load_t": load_t,
            "price_import": price_imp,
            "price_export": price_exp,
        }

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, info
