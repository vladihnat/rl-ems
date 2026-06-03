"""Environment factory: loads config, splits data temporally, returns train/test envs."""

import copy

import numpy as np
import pandas as pd
import yaml

from envs.base_microgrid_env import MicrogridEnv
from envs.components.battery import BatteryModel
from envs.components.load import LoadModel
from envs.components.price_signal import PriceSignal
from envs.components.pv_source import PVSource


def _temporal_split(pv_source: PVSource, split_ratio: float):
    """Split data by unique dates, stratified by month.

    Within each month, the first split_ratio dates go to train, the rest to
    test — so every season present in the data appears in both train and test.
    With a single month present (single-window extraction) this is identical to
    the previous chronological split.

    Returns (train_indices, test_indices) as integer arrays into the original data.
    """
    dates = pv_source.dates
    months = pd.DatetimeIndex(dates).month.to_numpy()
    train_dates: set = set()
    for m in np.unique(months):
        block = np.unique(dates[months == m])          # dates triées du mois
        n_train = int(len(block) * split_ratio)
        train_dates.update(block[:n_train].tolist())

    train_idx = np.array([i for i, d in enumerate(dates) if d in train_dates])
    test_idx = np.array([i for i, d in enumerate(dates) if d not in train_dates])
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
            cfg_dict["data"],
            n_steps=len(indices),
            delta_t_min=cfg_dict["time"]["delta_t_min"],
            timestamps=pv.timestamps,
        )
        if load.load_type != "fixed":
            assert load.n_steps == pv_full.n_steps, (
                f"load_csv length ({load.n_steps}) != pv_csv length "
                f"({pv_full.n_steps}); the two must be row-aligned on Time."
            )
            load.set_data_slice(indices)

        battery = BatteryModel(cfg_dict["battery"])
        price_signal = PriceSignal(
            cfg_dict["grid"], pv.timestamps, cfg_dict["time"]["delta_t_min"]
        )
        return MicrogridEnv(pv, load, battery, price_signal, cfg_dict)

    train_env = _build_env(train_idx, copy.deepcopy(cfg))
    test_env = _build_env(test_idx, copy.deepcopy(cfg))

    return train_env, test_env, cfg
