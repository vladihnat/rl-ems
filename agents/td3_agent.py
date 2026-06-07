"""TD3 agent: training and evaluation using Stable-Baselines3.

Off-policy déterministe (réutilise buffer/train_freq/gradient_steps comme SAC).
TD3 a une politique déterministe -> bruit d'action requis pour l'exploration
(NormalActionNoise, sigma configurable via training.action_noise_sigma).
Pas de terme d'entropie : ent_coef de la config est ignoré.
"""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from agents.common import (
    build_callback,
    build_policy_kwargs,
    evaluate_policy,
    make_train_env,
)


def make_action_noise(env, t_cfg: dict):
    """NormalActionNoise centré, sigma = training.action_noise_sigma (0 -> None)."""
    sigma = t_cfg.get("action_noise_sigma", 0.1)
    if not sigma or sigma <= 0:
        return None
    n = int(np.prod(env.action_space.shape))
    return NormalActionNoise(mean=np.zeros(n), sigma=float(sigma) * np.ones(n))


def train_td3(env, config: dict, eval_env=None, output_dir=None):
    """Train a TD3 agent. Returns (model, episode_rewards, vec_env).

    ``eval_env`` (+ ``output_dir``) ⇒ sélection best-model sur validation (cf. common.build_callback).
    """
    t_cfg = config["training"]
    train_env, vec_env = make_train_env(env, t_cfg)
    policy_kwargs = build_policy_kwargs(t_cfg)

    td3_kwargs = dict(
        learning_rate=t_cfg["learning_rate"],
        batch_size=t_cfg["batch_size"],
        buffer_size=t_cfg["buffer_size"],
        gamma=t_cfg.get("gamma", 0.99),
        tau=t_cfg.get("tau", 0.005),
        train_freq=t_cfg.get("train_freq", 1),
        gradient_steps=t_cfg.get("gradient_steps", 1),
        learning_starts=t_cfg.get("learning_starts", 100),
        action_noise=make_action_noise(env, t_cfg),
        seed=config["experiment"]["seed"],
        verbose=1,
    )
    if policy_kwargs:
        td3_kwargs["policy_kwargs"] = policy_kwargs

    model = TD3("MlpPolicy", train_env, **td3_kwargs)
    callbacks, reward_logger = build_callback(eval_env, t_cfg, output_dir, t_cfg["total_timesteps"])
    model.learn(total_timesteps=t_cfg["total_timesteps"], callback=callbacks)

    return model, reward_logger.episode_rewards, vec_env


def evaluate_td3(model, env, vec_normalize=None) -> dict:
    return evaluate_policy(model, env, vec_normalize)
