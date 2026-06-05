"""PPO agent: training and evaluation using Stable-Baselines3.

On-policy : pas de replay buffer ; expose n_steps/n_epochs/gae_lambda/clip_range.
Évaluation strictement identique aux autres algos (voir agents/common.py).
"""

from stable_baselines3 import PPO

from agents.common import (
    RewardLoggerCallback,
    build_policy_kwargs,
    evaluate_policy,
    make_train_env,
)


def train_ppo(env, config: dict):
    """Train a PPO agent. Returns (model, episode_rewards, vec_env)."""
    t_cfg = config["training"]
    train_env, vec_env = make_train_env(env, t_cfg)
    policy_kwargs = build_policy_kwargs(t_cfg)

    ppo_kwargs = dict(
        learning_rate=t_cfg["learning_rate"],
        n_steps=t_cfg.get("n_steps", 2048),
        batch_size=t_cfg["batch_size"],          # minibatch ; doit diviser n_steps
        n_epochs=t_cfg.get("n_epochs", 10),
        gamma=t_cfg.get("gamma", 0.99),
        gae_lambda=t_cfg.get("gae_lambda", 0.95),
        clip_range=t_cfg.get("clip_range", 0.2),
        ent_coef=t_cfg.get("ent_coef", 0.0),     # float (PAS "auto" comme SAC)
        seed=config["experiment"]["seed"],
        verbose=1,
    )
    if policy_kwargs:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    callback = RewardLoggerCallback()
    model.learn(total_timesteps=t_cfg["total_timesteps"], callback=callback)

    return model, callback.episode_rewards, vec_env


def evaluate_ppo(model, env, vec_normalize=None) -> dict:
    return evaluate_policy(model, env, vec_normalize)
