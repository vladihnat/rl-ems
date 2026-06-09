"""Helpers partagés par les agents SB3 alternatifs (PPO/TD3/DDPG).

Ces utilitaires reproduisent à l'identique le câblage de ``agents/sac_agent.py``
(VecNormalize optionnel, net_arch optionnel, et la boucle d'évaluation qui
normalise l'observation mais step sur l'env Gymnasium brut) afin que la
comparaison entre algorithmes se fasse à obs / budget / évaluation IDENTIQUES.
``sac_agent.py`` n'est volontairement pas modifié (code en cours dans l'arbre git).
"""

import copy
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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


class BestGapEvalCallback(BaseCallback):
    """Sélection best-model sur le COÛT NET AJUSTÉ de validation (plus bas = mieux).

    Remplace l'``EvalCallback`` SB3 (qui sélectionnait sur le *reward* de validation —
    un proxy bruité de l'objectif économique). ``net_cost_adjusted = net_cost +
    phantom_penalty·phantom_kWh`` est l'objectif réel et **intègre la garde phantom** :
    un checkpoint qui ne sert pas la charge prend +VOLL et n'est jamais retenu. Le coût
    MILP-val étant constant, minimiser ce coût ≡ minimiser ``relative_gap_adjusted``
    (sélection « sur le gap », sans solve MILP redondant à chaque éval).

    À chaque ``eval_freq`` pas : rollout déterministe sur la tranche val via
    ``evaluate_policy`` (mêmes stats VecNormalize qu'à l'entraînement, synchronisées et
    en mode éval). Sauve ``best_model.zip`` + ``best_vecnormalize.pkl`` au nouveau best,
    et écrit ``evaluations.npz`` (courbe de validation = détection du sur-apprentissage).
    """

    def __init__(self, eval_env, norm_obs: bool, output_dir: str,
                 eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env          # env Gymnasium brut (tranche val)
        self.norm_obs = norm_obs
        self.eval_freq = max(1, int(eval_freq))
        self.best_cost = np.inf
        self.best_model_path = os.path.join(output_dir, "best_model.zip")
        self.best_vec_path = os.path.join(output_dir, "best_vecnormalize.pkl")
        self.npz_path = os.path.join(output_dir, "evaluations.npz")
        self._timesteps: list = []
        self._costs: list = []
        self._served: list = []

    def _make_eval_vecnorm(self):
        """VecNormalize val en mode éval, stats synchronisées depuis l'env d'entraînement."""
        if not self.norm_obs:
            return None
        eval_venv = VecNormalize(
            DummyVecEnv([lambda: self.eval_env]),
            norm_obs=True, norm_reward=False, training=False,
        )
        train_venv = self.model.get_vec_normalize_env()
        if train_venv is not None:
            eval_venv.obs_rms = copy.deepcopy(train_venv.obs_rms)
        return eval_venv

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        vecnorm = self._make_eval_vecnorm()
        metrics = evaluate_policy(self.model, self.eval_env, vecnorm)
        cost = float(metrics["net_cost_adjusted"])
        self._timesteps.append(int(self.num_timesteps))
        self._costs.append(cost)
        self._served.append(float(metrics["served_load_ratio"]))
        if cost < self.best_cost:
            self.best_cost = cost
            self.model.save(self.best_model_path)
            if vecnorm is not None:
                vecnorm.save(self.best_vec_path)
            if self.verbose:
                print(f"[best-model] new best @ {self.num_timesteps}: "
                      f"net_cost_adj={cost:.2f}  served={metrics['served_load_ratio']:.4f}")
        return True

    def _on_training_end(self) -> None:
        if self._timesteps:
            np.savez(
                self.npz_path,
                timesteps=np.array(self._timesteps),
                results=np.array(self._costs),        # coût net ajusté (plus bas = mieux)
                served=np.array(self._served),
            )


def make_eval_callback(eval_env, t_cfg: dict, output_dir, total_timesteps):
    """Callback de sélection best-model sur une tranche de VALIDATION.

    Sélectionne sur ``net_cost_adjusted`` (= gap, garde phantom incluse). Retourne
    ``None`` s'il n'y a pas d'env de validation (comportement legacy : aucune sélection).
    """
    if eval_env is None or output_dir is None:
        return None
    norm_obs = t_cfg.get("norm_obs", False)
    eval_freq = max(2000, int(total_timesteps) // 100)   # ~100 évals quel que soit le budget
    return BestGapEvalCallback(eval_env, norm_obs, output_dir, eval_freq, verbose=1)


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
