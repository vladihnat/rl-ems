"""Environment factory: loads config, splits data temporally, returns train/test envs."""

import copy

import numpy as np
import pandas as pd
import yaml

from envs.base_microgrid_env import MicrogridEnv
from envs.components.battery import BatteryModel
from envs.components.load import LoadModel
from envs.components.pv_source import PVSource


def _temporal_split(pv_source: PVSource, split_ratio: float):
    """Split data by unique dates — first split_ratio dates for train, rest for test.

    Returns (train_indices, test_indices) as integer arrays into the original data.
    """
    unique_dates = np.unique(pv_source.dates)
    n_train_dates = int(len(unique_dates) * split_ratio)
    train_dates = set(unique_dates[:n_train_dates].tolist())

    train_idx = np.array([i for i, d in enumerate(pv_source.dates) if d in train_dates])
    test_idx = np.array([i for i, d in enumerate(pv_source.dates) if d not in train_dates])
    return train_idx, test_idx


def make_env(config_path: str):
    """Create train and test MicrogridEnv instances from a YAML config.

    Returns:
        (train_env, test_env, config_dict)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pv_full = PVSource(cfg["pv"], cfg["data"])

    split_ratio = cfg["training"]["train_split"]
    train_idx, test_idx = _temporal_split(pv_full, split_ratio)

    def _build_env(indices, cfg_dict):
        pv = PVSource(cfg_dict["pv"], cfg_dict["data"])
        pv.set_data_slice(indices)

        load = LoadModel(
            cfg_dict["load"],
            n_steps=len(indices),
            delta_t_min=cfg_dict["time"]["delta_t_min"],
            timestamps=pv.timestamps,
        )

        battery = BatteryModel(cfg_dict["battery"])
        return MicrogridEnv(pv, load, battery, cfg_dict)

    train_env = _build_env(train_idx, copy.deepcopy(cfg))
    test_env = _build_env(test_idx, copy.deepcopy(cfg))

    return train_env, test_env, cfg
