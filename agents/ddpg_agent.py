"""DDPG agent: training and evaluation using Stable-Baselines3.

Off-policy déterministe (SB3 DDPG = TD3 sans double-Q ni target smoothing).
Même interface off-policy que TD3 ; bruit d'action requis pour l'exploration.
"""

import numpy as np
from stable_baselines3 import DDPG
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


def train_ddpg(env, config: dict, eval_env=None, output_dir=None):
    """Train a DDPG agent. Returns (model, episode_rewards, vec_env).

    ``eval_env`` (+ ``output_dir``) ⇒ sélection best-model sur validation (cf. common.build_callback).
    """
    t_cfg = config["training"]
    train_env, vec_env = make_train_env(env, t_cfg)
    policy_kwargs = build_policy_kwargs(t_cfg)

    ddpg_kwargs = dict(
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
        ddpg_kwargs["policy_kwargs"] = policy_kwargs

    model = DDPG("MlpPolicy", train_env, **ddpg_kwargs)
    callbacks, reward_logger = build_callback(eval_env, t_cfg, output_dir, t_cfg["total_timesteps"])
    model.learn(total_timesteps=t_cfg["total_timesteps"], callback=callbacks)

    return model, reward_logger.episode_rewards, vec_env


def evaluate_ddpg(model, env, vec_normalize=None) -> dict:
    return evaluate_policy(model, env, vec_normalize)
