"""Gymnasium environment for microgrid energy management.

Action: normalized battery command in [-1, 1].
Observation: temporal features + system state + economic signals + PV forecast.
Reward: economic cost/revenue + SoC penalty.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.components.battery import BatteryModel
from envs.components.load import LoadModel
from envs.components.pv_source import PVSource


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        pv_source: PVSource,
        load_model: LoadModel,
        battery: BatteryModel,
        config: dict,
    ):
        super().__init__()
        self.pv = pv_source
        self.load = load_model
        self.battery = battery
        self.cfg = config

        self.delta_t_min = config["time"]["delta_t_min"]
        self.delta_t_h = self.delta_t_min / 60.0
        self.horizon_steps = int(config["time"]["horizon_h"] * 60 / self.delta_t_min)

        self.price_import = config["grid"]["price_import"]
        self.price_export = config["grid"]["price_export"]
        self.max_import_kw = config["grid"]["max_import_kw"]
        self.max_export_kw = config["grid"]["max_export_kw"]

        self.sigma_soc = config["reward"]["sigma_soc"]
        self.soc_safe_min = config["reward"]["soc_safe_min"]
        self.soc_safe_max = config["reward"]["soc_safe_max"]

        self.max_charge_kw = config["battery"]["max_charge_kw"]
        self.max_discharge_kw = config["battery"]["max_discharge_kw"]

        obs_dim = 9 + self.horizon_steps
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.max_steps = self.pv.n_steps - self.horizon_steps
        self.step_index = 0

    def _get_obs(self) -> np.ndarray:
        h_sin, h_cos, d_sin, d_cos = self.pv.get_temporal_features(self.step_index)
        soc = self.battery.soc
        load_t = self.load.get_load(self.step_index)
        pv_t = self.pv.get_irradiance(self.step_index)
        pv_forecast = self.pv.get_forecast(self.step_index, self.horizon_steps)

        obs = np.concatenate([
            np.array([h_sin, h_cos, d_sin, d_cos], dtype=np.float32),
            np.array([soc, load_t], dtype=np.float32),
            np.array([self.price_import, self.price_export], dtype=np.float32),
            np.array([pv_t], dtype=np.float32),
            pv_forecast,
        ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.battery.reset()
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

        P_grid = load_t - pv_t - Pb_effective
        P_grid = np.clip(P_grid, -self.max_export_kw, self.max_import_kw)

        r_eco = -(
            self.price_import * max(P_grid, 0.0)
            - self.price_export * max(-P_grid, 0.0)
        ) * self.delta_t_h

        r_soc = -self.sigma_soc * (
            max(0.0, self.soc_safe_min - new_soc)
            + max(0.0, new_soc - self.soc_safe_max)
        )

        reward = r_eco + r_soc

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False

        info = {
            "Pb_effective": Pb_effective,
            "P_grid": P_grid,
            "soc": new_soc,
            "r_eco": r_eco,
            "r_soc": r_soc,
            "pv_t": pv_t,
            "load_t": load_t,
        }

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, info
