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
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.registry import make_env, make_env_overfit
from baselines.milp_solver import run_milp
from evaluation.compare import compare_results
from evaluation.metrics import (
    boundary_soc_credit,
    compute_metrics,
    metrics_by_season,
    season_labels,
)


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


def plot_eval_curve(output_dir: str, output_path: str):
    """Trace la courbe du COÛT NET AJUSTÉ de validation (``evaluations.npz``).

    C'est le critère de sélection best-model (plus bas = mieux). Révèle le début du
    sur-apprentissage (creux de validation puis remontée du coût), preuve visuelle
    directe du diagnostic. No-op s'il n'y a pas de tranche de validation.
    """
    npz = os.path.join(output_dir, "evaluations.npz")
    if not os.path.exists(npz):
        return
    data = np.load(npz)
    ts = data["timesteps"]
    results = data["results"]
    if results.ndim > 1:                       # rétro-compat (ancien format n_evals×n_episodes)
        results = results.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, results, marker=".", label="Validation net cost adj.")
    best_i = int(np.argmin(results))           # coût : meilleur = minimum
    ax.axvline(ts[best_i], color="green", ls="--", alpha=0.7,
               label=f"best @ {int(ts[best_i])} ({results[best_i]:.1f})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Validation net cost adj. (EUR, lower = better)")
    ax.set_title("Validation Curve (best-model selection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Eval curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run microgrid experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--rescore", action="store_true",
        help="Ne PAS entraîner : recharger le modèle déjà sauvegardé dans results/<name>/ "
             "et seulement re-scorer (éval + MILP + compare) pour régénérer "
             "metrics.json/comparison.json avec le scoring courant (ex. crédit SoC de bord). "
             "Passer --config results/<sweep>/run_NNN/config_used.yaml. Les .zip/.pkl/.npz "
             "restent intacts. Aucun re-entraînement.",
    )
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

    # En --rescore, args.config EST déjà results/<name>/config_used.yaml ⇒ ne pas se copier
    # sur soi-même (SameFileError) et ne pas écraser le config du run.
    if not args.rescore:
        shutil.copy2(args.config, os.path.join(output_dir, "config_used.yaml"))

    print("\n[1/5] Creating environments...")
    eval_on_train = bool(cfg["training"].get("eval_on_train", False))
    if eval_on_train:
        # Sanity check « overfit » (demande superviseur) : on entraîne ET on évalue sur la
        # MÊME fenêtre CSV complète (aucun held-out), pour répondre à « le RL peut-il au moins
        # approcher l'optimum MILP sur des données qu'il a vues ? ». Instances NEUVES par rôle
        # (val / test / MILP) pour éviter toute fuite d'état entre évaluations.
        train_env, val_env, cfg = make_env_overfit(args.config)
        _, test_env, _ = make_env_overfit(args.config)
        print("  [eval_on_train] train = val = test = fenêtre CSV complète (overfit).")
    else:
        train_env, val_env, test_env, cfg = make_env(args.config, with_val=True)
    # Garde-fou : l'éval (val/test) DOIT partir d'un SoC fixe = init_soc (= celui du MILP),
    # sinon gap_best/gap_final sont incomparables et la comparaison RL↔MILP est inéquitable.
    # random_soc ne s'applique qu'au train (cf. registry._build_env).
    assert getattr(test_env, "_random_soc", False) is False, "test_env must NOT randomize SoC"
    if val_env is not None:
        assert getattr(val_env, "_random_soc", False) is False, "val_env must NOT randomize SoC"
    val_steps = val_env.max_steps if val_env is not None else 0
    print(f"  Train steps: {train_env.max_steps}, Val steps: {val_steps}, Test steps: {test_env.max_steps}")

    # Scoring N−1 (eval_on_train uniquement) : nb de pas de tête à scorer pour eval/MILP/compare.
    # Le MILP planifie sur env.max_steps et brade le stock au dernier pas (dump SoC terminal) ;
    # scorer une fenêtre plus courte que l'horizon de plan exclut ce dump. None = pleine fenêtre.
    score_steps = None
    score_days = cfg["training"].get("score_days") if eval_on_train else None
    if score_days is not None:
        steps_per_day = round(24 * 60 / cfg["time"]["delta_t_min"])
        score_steps = min(int(test_env.max_steps), int(score_days) * steps_per_day)
        print(f"  [score_days={score_days}] coût scoré sur les {score_steps} premiers pas.")

    def _rescore(metrics, env):
        """N−1 : recalcule les métriques sur les ``score_steps`` premiers pas seulement.

        Tranche l'history retourné par evaluate_*/run_milp et relance compute_metrics +
        metrics_by_season à l'identique (cf. agents/sac_agent.evaluate_sac et
        baselines/milp_solver.run_milp). No-op si ``score_steps`` est None : les runs
        normaux restent bit-identiques.
        """
        if score_steps is None:
            return metrics
        hist = metrics.get("history")
        if hist is None:
            return metrics
        sub = {k: (np.asarray(v)[:score_steps] if np.ndim(v) >= 1 else v)
               for k, v in hist.items()}
        dt = env.delta_t_h
        soc_min = env.cfg["reward"]["soc_safe_min"]
        soc_max = env.cfg["reward"]["soc_safe_max"]
        pen = env.cfg["grid"].get("phantom_penalty", 1e3)
        p_imp = env.price_signal.import_prices[:score_steps]
        p_exp = env.price_signal.export_prices[:score_steps]
        # Repart de ``metrics`` pour préserver les clés non recalculées par compute_metrics
        # (ex. run_milp ajoute solver_status / objective_value après coup), puis écrase les
        # scalaires par-fenêtre avec leur version N−1.
        out = dict(metrics)
        out.update(compute_metrics(sub, dt, soc_min, soc_max, p_imp, p_exp, pen))
        labels = season_labels(env.pv.timestamps[:score_steps])
        out["by_season"] = metrics_by_season(sub, labels, dt, soc_min, soc_max,
                                             p_imp, p_exp, pen)
        # Crédit de SoC de bord : la fenêtre scorée est un préfixe STRICT du rollout (le
        # plan MILP/RL continue au-delà) ⇒ valoriser l'énergie batterie restante au bord,
        # SYMÉTRIQUEMENT RL et MILP (ce _rescore est appelé sur les deux). Sinon, drainer la
        # batterie avant le bord paraît moins cher : artefact pur sous prix fixes (cf. exp22
        # hiver, run_030 : gap −61 % → ~0 %, cohérent avec le plein horizon 88.99 vs 88.98).
        # ``hist["soc"]`` est en FRACTION (MILP soc_sol[1:], RL info["soc"]).
        soc_full = np.asarray(hist["soc"])
        if score_steps < soc_full.shape[0]:
            credit = boundary_soc_credit(
                soc_end_frac=float(soc_full[score_steps - 1]),
                soc_init_frac=float(env.cfg["battery"]["init_soc"]),
                capacity_kwh=float(env.cfg["battery"]["capacity_kwh"]),
                eta_discharge=float(env.cfg["battery"]["efficiency_discharge"]),
                store_value=float(env._store_value(score_steps - 1)),
            )
            out["net_cost"] -= credit
            out["net_cost_adjusted"] -= credit
            # by_season : crédit appliqué uniquement en mono-saison (cas overfit exp22, le
            # seul où score_steps est défini). En multi-saison, chaque tranche aurait son
            # propre bord ⇒ on laisse le split brut plutôt que d'appliquer un crédit faux.
            season_metrics = out.get("by_season", {})
            if len(season_metrics) == 1:
                for s in season_metrics.values():
                    s["net_cost"] -= credit
                    s["net_cost_adjusted"] -= credit
        out["history"] = sub
        return out

    if args.rescore:
        # RE-SCORE : aucun entraînement. On recharge le modèle FINAL déjà sauvegardé et on
        # laisse le reste ([3/5]→écriture) régénérer metrics.json/comparison.json avec le
        # scoring courant (ex. crédit SoC de bord, cf. boundary_soc_credit). best_model.zip,
        # *vec_normalize.pkl et evaluations.npz restent intacts (sélection best inchangée).
        from monitoring.run_optimization_example import _load_model
        print("\n[2/5] [RESCORE] Skipping training — reloading saved model...")
        final_model_path = os.path.join(output_dir, f"{algo_lower}_model.zip")
        model = _load_model(final_model_path, algo_name)
        episode_rewards = []
        vec_env = None
        vec_normalize_path = os.path.join(output_dir, "vec_normalize.pkl")
        if not os.path.exists(vec_normalize_path):
            vec_normalize_path = None
        # Préserver training_time_s de l'ancien metrics.json (sinon remis à 0).
        prev_metrics = os.path.join(output_dir, "metrics.json")
        training_time_s = 0.0
        if os.path.exists(prev_metrics):
            with open(prev_metrics) as f:
                training_time_s = json.load(f).get("training", {}).get("training_time_s", 0.0)
        print(f"  Reloaded final model from {final_model_path} (no training).")
    else:
        print(f"\n[2/5] Training {algo_name} agent...")
        t_train_start = time.time()
        result = train_fn(train_env, cfg, eval_env=val_env, output_dir=output_dir)
        training_time_s = time.time() - t_train_start
        print(f"  Training time: {training_time_s:.1f}s")
        model, episode_rewards = result[0], result[1]
        vec_env = result[2] if len(result) > 2 else None

        # Modèle FINAL (repro) + ses stats VecNormalize finales.
        final_model_path = os.path.join(output_dir, f"{algo_lower}_model.zip")
        model.save(final_model_path)
        print(f"  Final model saved to {final_model_path}")

        vec_normalize_path = None
        if vec_env is not None:
            vec_normalize_path = os.path.join(output_dir, "vec_normalize.pkl")
            vec_env.save(vec_normalize_path)
            print(f"  Final VecNormalize stats saved to {vec_normalize_path}")

        plot_training_curves(episode_rewards, os.path.join(output_dir, "training_curves.png"), algo_name)
        plot_eval_curve(output_dir, os.path.join(output_dir, "eval_curve.png"))

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _load_eval_vecnorm(pkl_path):
        if pkl_path is None or not os.path.exists(pkl_path):
            return None
        venv = VecNormalize.load(pkl_path, DummyVecEnv([lambda: test_env]))
        venv.training = False
        venv.norm_reward = False
        return venv

    # Sélection best-model (EvalCallback sur la tranche de validation) ; fallback = modèle final.
    best_model_path = os.path.join(output_dir, "best_model.zip")
    best_vec_path = os.path.join(output_dir, "best_vecnormalize.pkl")
    used_best = os.path.exists(best_model_path)
    if used_best:
        eval_model = type(model).load(best_model_path)
        eval_vec_src = best_vec_path if os.path.exists(best_vec_path) else vec_normalize_path
        print(f"  Using BEST-model checkpoint: {best_model_path}")
    else:
        eval_model = model
        eval_vec_src = vec_normalize_path
        print("  No best-model checkpoint (no validation slice) — evaluating FINAL model.")

    print(f"\n[3/5] Evaluating {algo_name} on test set...")
    rl_metrics = evaluate_fn(eval_model, test_env, _load_eval_vecnorm(eval_vec_src))
    rl_metrics = _rescore(rl_metrics, test_env)
    print(f"  RL net cost: {rl_metrics['net_cost']:.4f} EUR")
    print(f"  RL self-consumption: {rl_metrics['self_consumption_rate']:.2%}")

    selection = {"used_best_model": used_best, "gap_best": None, "gap_final": None}

    print("\n[4/5] Running MILP baseline on test set...")
    try:
        import cvxpy  # noqa: F401
        if eval_on_train:
            _, test_env_milp, _ = make_env_overfit(args.config)
        else:
            _, test_env_milp, _ = make_env(args.config)
        milp_metrics = run_milp(test_env_milp, cfg)
        milp_metrics = _rescore(milp_metrics, test_env_milp)
        print(f"  MILP net cost: {milp_metrics['net_cost']:.4f} EUR")
        print(f"  MILP status: {milp_metrics['solver_status']}")
        print("\n[5/5] Comparing results...")
        comparison = compare_results(rl_metrics, milp_metrics, output_dir)

        # Gap stratifié par saison (Hiver/Été), sur le modèle évalué (best si dispo).
        # Les sous-métriques par saison sont produites par evaluate_*/run_milp via
        # evaluation.metrics.metrics_by_season (réutilise compute_metrics par tranche).
        rl_by = rl_metrics.pop("by_season", {})
        milp_by = milp_metrics.pop("by_season", {})
        by_season = {}
        for season in ("hiver", "ete"):
            if season not in rl_by or season not in milp_by:
                continue
            rl_net, milp_net = rl_by[season]["net_cost"], milp_by[season]["net_cost"]
            rl_adj, milp_adj = rl_by[season]["net_cost_adjusted"], milp_by[season]["net_cost_adjusted"]
            by_season[season] = {
                "gap": (rl_net - milp_net) / abs(milp_net) if abs(milp_net) > 1e-9 else None,
                "gap_adjusted": (rl_adj - milp_adj) / abs(milp_adj) if abs(milp_adj) > 1e-9 else None,
                # Gap ABSOLU (€) : toujours défini. En hiver le coût net MILP est ~0 €
                # (faible PV → peu d'export), donc le gap RELATIF (÷|milp_net|) explose à
                # des milliers de % : artefact de dénominateur≈0, ininterprétable. Le gap
                # absolu reste fiable quel que soit le régime (cf. aggregate_results.py).
                "gap_abs_eur": rl_net - milp_net,
                "served": rl_by[season]["served_load_ratio"],
                "phantom": rl_by[season]["phantom_energy_kwh"],
                "rl_net_cost": rl_net,
                "milp_net_cost": milp_net,
                "n_steps": rl_by[season]["n_steps"],
            }
            g = by_season[season]["gap"]
            g_str = f"{g:+.1%}" if g is not None else "n/a"
            print(f"  Gap [{season}] : {g_str}  (gap_abs={by_season[season]['gap_abs_eur']:+.1f} EUR, "
                  f"n={by_season[season]['n_steps']} steps)")
        comparison["by_season"] = by_season

        # Référence : gap du modèle FINAL (diagnostic « loterie de point d'arrêt »).
        if used_best:
            final_metrics = evaluate_fn(model, test_env, _load_eval_vecnorm(vec_normalize_path))
            final_metrics = _rescore(final_metrics, test_env)
            final_comparison = compare_results(final_metrics, milp_metrics, output_dir=None)
            selection["gap_best"] = comparison.get("relative_gap")
            selection["gap_final"] = final_comparison.get("relative_gap")
            print(f"  Gap best-model: {selection['gap_best']:+.1%}   "
                  f"Gap final-model: {selection['gap_final']:+.1%}")
        else:
            selection["gap_final"] = comparison.get("relative_gap")
    except ImportError:
        print("  [SKIP] cvxpy not available — MILP baseline skipped.")
        milp_metrics = {}
        comparison = {}

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    all_metrics = {
        "training": {"training_time_s": round(training_time_s, 2)},
        "selection": {k: to_serializable(v) for k, v in selection.items()},
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
