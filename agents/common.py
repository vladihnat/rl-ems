"""Helpers partagés par les agents SB3 alternatifs (PPO/TD3/DDPG).

Ces utilitaires reproduisent à l'identique le câblage de ``agents/sac_agent.py``
(VecNormalize optionnel, net_arch optionnel, et la boucle d'évaluation qui
normalise l'observation mais step sur l'env Gymnasium brut) afin que la
comparaison entre algorithmes se fasse à obs / budget / évaluation IDENTIQUES.
``sac_agent.py`` n'est volontairement pas modifié (code en cours dans l'arbre git).
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.metrics import compute_metrics


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards for plotting training curves (idem sac_agent)."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def make_train_env(env, t_cfg: dict):
    """Wrap ``env`` in VecNormalize when norm_obs/norm_reward are set.

    Returns ``(train_env, vec_env)`` where ``vec_env`` is the VecNormalize
    wrapper (or None). Identique au comportement de ``train_sac``.
    """
    norm_obs    = t_cfg.get("norm_obs",    False)
    norm_reward = t_cfg.get("norm_reward", False)
    if norm_obs or norm_reward:
        vec_env   = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=norm_obs, norm_reward=norm_reward)
        return vec_env, vec_env
    return env, None


def build_policy_kwargs(t_cfg: dict) -> dict:
    """net_arch optionnel (défaut SB3 si absent). Axe de sweep partagé."""
    policy_kwargs = {}
    net_arch = t_cfg.get("net_arch")
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)
    return policy_kwargs


def evaluate_policy(model, env, vec_normalize=None) -> dict:
    """Roll out a trained policy on ``env`` and return metrics + history.

    Générique : fonctionne pour tout modèle SB3 exposant ``.predict``
    (SAC/PPO/TD3/DDPG). Copie fidèle de ``evaluate_sac`` pour garantir une
    évaluation strictement identique entre algorithmes.
    """
    obs, _ = env.reset()
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs)
    history = {
        "P_grid": [],
        "Pb_effective": [],
        "soc": [],
        "pv_t": [],
        "load_t": [],
        "r_eco": [],
        "r_soc": [],
        "reward": [],
        "P_phantom": [],
        "Pcurt": [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if vec_normalize is not None:
            obs = vec_normalize.normalize_obs(obs)
        done = terminated or truncated

        history["P_grid"].append(info["P_grid"])
        history["Pb_effective"].append(info["Pb_effective"])
        history["soc"].append(info["soc"])
        history["pv_t"].append(info["pv_t"])
        history["load_t"].append(info["load_t"])
        history["r_eco"].append(info["r_eco"])
        history["r_soc"].append(info["r_soc"])
        history["reward"].append(reward)
        history["P_phantom"].append(info["P_phantom"])
        history["Pcurt"].append(info["Pcurt"])

    for k in history:
        history[k] = np.array(history[k])

    delta_t_h = env.delta_t_h
    soc_min = env.cfg["reward"]["soc_safe_min"]
    soc_max = env.cfg["reward"]["soc_safe_max"]
    T = len(history["P_grid"])
    price_import = env.price_signal.import_prices[:T]
    price_export = env.price_signal.export_prices[:T]
    phantom_penalty = env.cfg["grid"].get("phantom_penalty", 1e3)

    metrics = compute_metrics(history, delta_t_h, soc_min, soc_max,
                              price_import, price_export, phantom_penalty)
    metrics["history"] = history
    return metrics
