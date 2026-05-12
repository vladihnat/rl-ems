"""SAC agent: training and evaluation using Stable-Baselines3."""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.metrics import compute_metrics


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards for plotting training curves."""

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


def train_sac(env, config: dict):
    """Train a SAC agent on the given environment.

    Returns:
        (model, episode_rewards): trained SB3 model and list of episode rewards.
    """
    t_cfg = config["training"]

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=t_cfg["learning_rate"],
        batch_size=t_cfg["batch_size"],
        buffer_size=t_cfg["buffer_size"],
        seed=config["experiment"]["seed"],
        verbose=1,
    )

    callback = RewardLoggerCallback()
    model.learn(total_timesteps=t_cfg["total_timesteps"], callback=callback)

    return model, callback.episode_rewards


def evaluate_sac(model, env) -> dict:
    """Run the trained policy on an environment and return metrics.

    Returns:
        dict with episode history and computed metrics.
    """
    obs, _ = env.reset()
    history = {
        "P_grid": [],
        "Pb_effective": [],
        "soc": [],
        "pv_t": [],
        "load_t": [],
        "r_eco": [],
        "r_soc": [],
        "reward": [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        history["P_grid"].append(info["P_grid"])
        history["Pb_effective"].append(info["Pb_effective"])
        history["soc"].append(info["soc"])
        history["pv_t"].append(info["pv_t"])
        history["load_t"].append(info["load_t"])
        history["r_eco"].append(info["r_eco"])
        history["r_soc"].append(info["r_soc"])
        history["reward"].append(reward)

    for k in history:
        history[k] = np.array(history[k])

    delta_t_h = env.delta_t_h
    soc_min = env.cfg["reward"]["soc_safe_min"]
    soc_max = env.cfg["reward"]["soc_safe_max"]
    price_import = env.price_import
    price_export = env.price_export

    metrics = compute_metrics(history, delta_t_h, soc_min, soc_max, price_import, price_export)
    metrics["history"] = history
    return metrics
