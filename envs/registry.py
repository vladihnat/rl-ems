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


def _temporal_split3(pv_source: PVSource, split_ratio: float, val_split: float):
    """3-way month-stratified split: train / validation / test.

    Le split mensuel train/test est IDENTIQUE à ``_temporal_split`` (donc ``test_idx``
    est inchangé et le gap reste comparable au MILP et aux runs passés). La validation
    est ensuite découpée du *train pool* comme un sous-ensemble ENTRELACÉ (1 date sur
    ``round(1/val_split)``), réparti sur tous les mois — un découpage par mois échoue
    ici (28 jours épars → ~2 jours train/mois, ``int(2*0.2)=0``). ``val_split<=0`` ⇒
    validation vide (comportement legacy à 2 voies).

    Returns (train_indices, val_indices, test_indices).
    """
    dates = pv_source.dates
    months = pd.DatetimeIndex(dates).month.to_numpy()
    train_pool: set = set()
    test_set: set = set()
    for m in np.unique(months):
        block = np.unique(dates[months == m])          # dates triées du mois
        n_train = int(len(block) * split_ratio)
        train_pool.update(block[:n_train].tolist())
        test_set.update(block[n_train:].tolist())       # == test de _temporal_split

    pool_sorted = sorted(train_pool)
    if val_split and val_split > 0 and len(pool_sorted) >= 2:
        stride = max(2, round(1.0 / val_split))          # 0.2 -> 5 (1 jour val sur 5)
        val_set = set(pool_sorted[stride - 1::stride])   # offset: garde la 1re date en train
        if not val_set:
            val_set = {pool_sorted[-1]}                  # garantit ≥1 jour de validation
        train_set = set(pool_sorted) - val_set
    else:
        val_set = set()
        train_set = set(pool_sorted)

    train_idx = np.array([i for i, d in enumerate(dates) if d in train_set])
    val_idx = np.array([i for i, d in enumerate(dates) if d in val_set])
    test_idx = np.array([i for i, d in enumerate(dates) if d in test_set])
    return train_idx, val_idx, test_idx


def make_env(config_path: str, with_val: bool = False):
    """Create train (val) and test MicrogridEnv instances from a YAML config.

    Returns:
        ``(train_env, test_env, config_dict)`` par défaut (rétro-compat).
        Si ``with_val=True`` : ``(train_env, val_env, test_env, config_dict)`` où
        ``val_env`` est ``None`` quand ``training.val_split`` est absent/≤0.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pv_full = PVSource(cfg["pv"], cfg["data"])

    split_ratio = cfg["training"]["train_split"]

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

    if with_val:
        val_split = cfg["training"].get("val_split", 0.0)
        train_idx, val_idx, test_idx = _temporal_split3(pv_full, split_ratio, val_split)
        train_env = _build_env(train_idx, copy.deepcopy(cfg))
        val_env = _build_env(val_idx, copy.deepcopy(cfg)) if len(val_idx) > 0 else None
        test_env = _build_env(test_idx, copy.deepcopy(cfg))
        return train_env, val_env, test_env, cfg

    train_idx, test_idx = _temporal_split(pv_full, split_ratio)
    train_env = _build_env(train_idx, copy.deepcopy(cfg))
    test_env = _build_env(test_idx, copy.deepcopy(cfg))
    return train_env, test_env, cfg
