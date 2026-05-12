"""Single entry point for running microgrid experiments.

Usage:
    python experiments/run_experiment.py --config configs/exp01_perfect_foresight.yaml
"""

import argparse
import importlib
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.registry import make_env
from baselines.milp_solver import run_milp
from evaluation.compare import compare_results


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def plot_training_curves(episode_rewards: list, output_path: str, algo_name: str = "RL"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")

    if len(episode_rewards) > 10:
        window = min(20, len(episode_rewards) // 5)
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"Moving avg ({window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"{algo_name} Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run microgrid experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    seed = cfg["experiment"]["seed"]
    algo_name = cfg["training"]["algorithm"]
    algo_lower = algo_name.lower()
    set_seed(seed)

    # Dynamically import the appropriate agent module and functions based on config yaml
    agent_module = importlib.import_module(f"agents.{algo_lower}_agent")
    train_fn = getattr(agent_module, f"train_{algo_lower}")
    evaluate_fn = getattr(agent_module, f"evaluate_{algo_lower}")

    print(f"=== Experiment: {exp_name} ===")
    print(f"Seed: {seed}")
    print(f"Algorithm: {algo_name}")

    output_dir = os.path.join("results", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy2(args.config, os.path.join(output_dir, "config_used.yaml"))

    print("\n[1/5] Creating environments...")
    train_env, test_env, cfg = make_env(args.config)
    print(f"  Train steps: {train_env.max_steps}, Test steps: {test_env.max_steps}")

    print(f"\n[2/5] Training {algo_name} agent...")
    model, episode_rewards = train_fn(train_env, cfg)

    model_path = os.path.join(output_dir, f"{algo_lower}_model.zip")
    model.save(model_path)
    print(f"  Model saved to {model_path}")

    plot_training_curves(episode_rewards, os.path.join(output_dir, "training_curves.png"), algo_name)

    print(f"\n[3/5] Evaluating {algo_name} on test set...")
    rl_metrics = evaluate_fn(model, test_env)
    print(f"  RL net cost: {rl_metrics['net_cost']:.4f} EUR")
    print(f"  RL self-consumption: {rl_metrics['self_consumption_rate']:.2%}")

    print("\n[4/5] Running MILP baseline on test set...")
    test_env_milp, _, _ = make_env(args.config)
    _, test_env_milp, _ = make_env(args.config)
    milp_metrics = run_milp(test_env_milp, cfg)
    print(f"  MILP net cost: {milp_metrics['net_cost']:.4f} EUR")
    print(f"  MILP status: {milp_metrics['solver_status']}")

    print("\n[5/5] Comparing results...")
    comparison = compare_results(rl_metrics, milp_metrics, output_dir)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    all_metrics = {
        "rl": {k: to_serializable(v) for k, v in rl_metrics.items() if k != "history"},
        "milp": {k: to_serializable(v) for k, v in milp_metrics.items() if k != "history"},
        "comparison": {k: to_serializable(v) for k, v in comparison.items()},
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=to_serializable)
    print(f"\nAll metrics saved to {metrics_path}")

    print(f"\n=== Experiment {exp_name} complete ===")


if __name__ == "__main__":
    main()
