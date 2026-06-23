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
    est la **queue chronologique du bloc train de chaque mois** : les ``n_val`` dernières
    dates train de chaque mois (juste avant le bloc test), réparties sur toutes les
    saisons. Ce découpage rend la validation « test-like » (prédire la fin de la
    fenêtre à partir du début, comme le test) au lieu de l'ancien sous-ensemble
    entrelacé trop in-distribution. ``n_val = max(1, int(n_train * val_split))``
    garantit ≥1 jour de validation par mois (ici 5 jours train/mois → 1 jour val).
    ``val_split<=0`` ⇒ validation vide (comportement legacy à 2 voies).

    Returns (train_indices, val_indices, test_indices).
    """
    dates = pv_source.dates
    months = pd.DatetimeIndex(dates).month.to_numpy()
    train_set: set = set()
    val_set: set = set()
    test_set: set = set()
    for m in np.unique(months):
        block = np.unique(dates[months == m])          # dates triées du mois
        n_train = int(len(block) * split_ratio)
        train_block = block[:n_train]
        test_set.update(block[n_train:].tolist())       # == test de _temporal_split
        if val_split and val_split > 0 and n_train >= 2:
            n_val = max(1, int(n_train * val_split))     # 0.2 * 5 -> 1 jour val / mois
            val_set.update(train_block[-n_val:].tolist())   # queue chronologique du train
            train_set.update(train_block[:-n_val].tolist())
        else:
            train_set.update(train_block.tolist())

    train_idx = np.array([i for i, d in enumerate(dates) if d in train_set])
    val_idx = np.array([i for i, d in enumerate(dates) if d in val_set])
    test_idx = np.array([i for i, d in enumerate(dates) if d in test_set])
    return train_idx, val_idx, test_idx


def _build_env(indices, cfg_dict, pv_full, is_train=False):
    """Construit un MicrogridEnv tranché sur ``indices``.

    Lifté de ``make_env`` (où il était imbriqué) pour être réutilisé par
    ``make_env_overfit``. ``pv_full`` sert uniquement à l'assert d'alignement load↔pv.

    random_soc ne doit randomiser le SoC initial QU'À L'ENTRAÎNEMENT. Sinon val/test
    tirent un SoC initial non contrôlé (RNG seedé par entropie, cf.
    base_microgrid_env.reset) ≠ init_soc fixe du MILP (milp_solver.py:init_soc) :
    gap_best/gap_final deviennent incomparables (2 reset = 2 tirages) et la comparaison
    RL↔MILP est inéquitable (SoC initial élevé = énergie « gratuite »). On force donc
    random_soc=False hors entraînement.
    """
    if not is_train:
        cfg_dict.setdefault("training", {})["random_soc"] = False
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


def make_env_overfit(config_path: str):
    """Train + eval envs couvrant TOUTE la fenêtre CSV (sanity check « overfit »).

    Pour la validation 2-jours/N-jours demandée par le superviseur : on entraîne ET on
    évalue sur la MÊME fenêtre (aucun held-out), pour répondre à « le RL peut-il au moins
    approcher l'optimum MILP sur des données qu'il a vues ? ». Chaque appel renvoie des
    instances NEUVES (à appeler une fois par rôle : train / validation / test / MILP, afin
    d'éviter toute fuite d'état entre évaluations). ``train_env`` garde ``random_soc`` du
    config ; ``eval_env`` force ``random_soc=False`` (même SoC initial que le MILP).

    Returns ``(train_env, eval_env, config_dict)``.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pv_full = PVSource(cfg["pv"], cfg["data"])
    all_idx = np.arange(pv_full.n_steps)
    train_env = _build_env(all_idx, copy.deepcopy(cfg), pv_full, is_train=True)
    eval_env = _build_env(all_idx, copy.deepcopy(cfg), pv_full, is_train=False)
    return train_env, eval_env, cfg


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

    if with_val:
        val_split = cfg["training"].get("val_split", 0.0)
        train_idx, val_idx, test_idx = _temporal_split3(pv_full, split_ratio, val_split)
        train_env = _build_env(train_idx, copy.deepcopy(cfg), pv_full, is_train=True)
        val_env = _build_env(val_idx, copy.deepcopy(cfg), pv_full) if len(val_idx) > 0 else None
        test_env = _build_env(test_idx, copy.deepcopy(cfg), pv_full)
        return train_env, val_env, test_env, cfg

    train_idx, test_idx = _temporal_split(pv_full, split_ratio)
    train_env = _build_env(train_idx, copy.deepcopy(cfg), pv_full, is_train=True)
    test_env = _build_env(test_idx, copy.deepcopy(cfg), pv_full)
    return train_env, test_env, cfg
