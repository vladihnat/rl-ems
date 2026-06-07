"""Helpers partagés par les agents SB3 alternatifs (PPO/TD3/DDPG).

Ces utilitaires reproduisent à l'identique le câblage de ``agents/sac_agent.py``
(VecNormalize optionnel, net_arch optionnel, et la boucle d'évaluation qui
normalise l'observation mais step sur l'env Gymnasium brut) afin que la
comparaison entre algorithmes se fasse à obs / budget / évaluation IDENTIQUES.
``sac_agent.py`` n'est volontairement pas modifié (code en cours dans l'arbre git).
"""

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
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


class SaveVecNormalizeOnBest(BaseCallback):
    """Sauve les stats VecNormalize d'entraînement à chaque nouveau best.

    SB3 2.x ``EvalCallback`` ne sauve que ``best_model.zip`` au nouveau best (PAS la
    normalisation). Branché en ``callback_on_new_best`` pour persister les obs_rms
    courantes, afin que l'éval test du best-model utilise les bonnes statistiques.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        # self.model peut ne pas être initialisé sur le child ; fallback via parent (EvalCallback).
        model = getattr(self, "model", None) or getattr(getattr(self, "parent", None), "model", None)
        if model is not None:
            venv = model.get_vec_normalize_env()
            if venv is not None:
                venv.save(self.save_path)
        return True


def make_eval_callback(eval_env, t_cfg: dict, output_dir, total_timesteps):
    """EvalCallback partagé : sélection best-model sur une tranche de VALIDATION.

    - enveloppe ``eval_env`` en VecNormalize (mêmes flags qu'à l'entraînement,
      ``training=False``, ``norm_reward=False``) ⇒ EvalCallback synchronise les stats
      train→val à chaque éval (``sync_envs_normalization``) ;
    - sauve ``best_model.zip`` (EvalCallback) + ``best_vecnormalize.pkl``
      (``callback_on_new_best``) dans ``output_dir`` ;
    - ``log_path`` ⇒ ``evaluations.npz`` (courbe de validation = détection du sur-apprentissage).

    Retourne ``None`` si pas d'env de validation (comportement legacy : aucune sélection).
    """
    if eval_env is None or output_dir is None:
        return None
    norm_obs = t_cfg.get("norm_obs", False)
    norm_reward = t_cfg.get("norm_reward", False)
    if norm_obs or norm_reward:
        eval_venv = VecNormalize(
            DummyVecEnv([lambda: eval_env]),
            norm_obs=norm_obs, norm_reward=False, training=False,
        )
        on_best = SaveVecNormalizeOnBest(os.path.join(output_dir, "best_vecnormalize.pkl"))
    else:
        eval_venv = eval_env
        on_best = None
    eval_freq = max(2000, int(total_timesteps) // 100)   # ~100 évals quel que soit le budget
    return EvalCallback(
        eval_venv,
        best_model_save_path=output_dir,
        log_path=output_dir,
        n_eval_episodes=1,                # 1 épisode = toute la tranche val (déterministe)
        eval_freq=eval_freq,
        deterministic=True,
        warn=False,
        callback_on_new_best=on_best,
    )


def build_callback(eval_env, t_cfg: dict, output_dir, total_timesteps):
    """Assemble (callback à passer à ``learn``, RewardLoggerCallback).

    Combine le logger de reward (courbe d'entraînement) et, si une tranche de validation
    est fournie, l'EvalCallback de sélection best-model. Le logger est renvoyé à part pour
    récupérer ``episode_rewards`` après l'entraînement.
    """
    reward_logger = RewardLoggerCallback()
    eval_cb = make_eval_callback(eval_env, t_cfg, output_dir, total_timesteps)
    if eval_cb is None:
        return reward_logger, reward_logger
    return CallbackList([reward_logger, eval_cb]), reward_logger


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
