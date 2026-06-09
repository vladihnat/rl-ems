#!/usr/bin/env python3
"""Diagnostic dispatch RL vs MILP : pourquoi le gap ~16.5 % ?

Charge le ``best_model.zip`` d'un run, le rejoue sur le test set (``evaluate_policy``),
résout le MILP sur le MÊME test set (``run_milp``), et superpose sur quelques jours-types
(autour de la pointe de prix d'export) : PV/charge, prix import/export, puissance batterie
``Pb`` + SoC, et flux réseau import(+)/export(-). Sert à VOIR l'attracteur :
le RL ne charge jamais depuis le réseau (``peak_import`` ≈ pointe de charge) et sur-exporte
le PV de midi au lieu de le stocker (SC < MILP), d'où le gap.

Usage:
    python evaluation/diagnose_dispatch.py --run-dir results/exp13_exportfc/run_000 --days 3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common import evaluate_policy
from baselines.milp_solver import run_milp
from envs.registry import make_env


def _load_model(run_dir: str):
    from stable_baselines3 import SAC
    for name in ("best_model.zip", "sac_model.zip"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            print(f"[diag] modèle : {path}")
            return SAC.load(path), name
    sys.exit(f"[diag] aucun best_model.zip / sac_model.zip dans {run_dir}")


def _maybe_vecnorm(run_dir: str, cfg: dict, test_env):
    if not cfg["training"].get("norm_obs", False):
        return None
    pkl = os.path.join(run_dir, "best_vecnormalize.pkl")
    if not os.path.exists(pkl):
        pkl = os.path.join(run_dir, "vec_normalize.pkl")
    if not os.path.exists(pkl):
        return None
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv = VecNormalize.load(pkl, DummyVecEnv([lambda: test_env]))
    venv.training = False
    venv.norm_reward = False
    return venv


def main():
    ap = argparse.ArgumentParser(description="Diagnostic dispatch RL vs MILP sur le test set.")
    ap.add_argument("--run-dir", required=True, help="ex. results/exp13_exportfc/run_000")
    ap.add_argument("--days", type=int, default=0, help="Nb de jours test tracés (0 = tous)")
    ap.add_argument("--out", default=None, help="PNG de sortie (def. <run-dir>/diagnose_dispatch.png)")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        sys.exit(f"[diag] config_used.yaml introuvable dans {args.run_dir}")

    # Deux test envs distincts : un pour le rollout RL (stepé), un frais pour le MILP.
    _, _, test_env_rl, cfg = make_env(cfg_path, with_val=True)
    _, _, test_env_milp, _ = make_env(cfg_path, with_val=True)

    model, _ = _load_model(args.run_dir)
    rl = evaluate_policy(model, test_env_rl, _maybe_vecnorm(args.run_dir, cfg, test_env_rl))
    milp = run_milp(test_env_milp, cfg)
    rlh, mh = rl["history"], milp["history"]

    T = len(rlh["P_grid"])
    delta_t_h = cfg["time"]["delta_t_min"] / 60.0
    steps_per_day = int(24 * 60 / cfg["time"]["delta_t_min"])
    ts = pd.to_datetime(np.asarray(test_env_rl.pv.timestamps)[:T])
    price_imp = np.asarray(test_env_rl.price_signal.import_prices[:T])
    price_exp = np.asarray(test_env_rl.price_signal.export_prices[:T])

    rl_imp = np.maximum(rlh["P_grid"], 0.0)
    rl_exp = np.maximum(-rlh["P_grid"], 0.0)

    # Résumé chiffré (rappel du diagnostic).
    print(f"[diag] RL   : SC={rl['self_consumption_rate']:.1%}  import={rl['energy_imported_kwh']:.0f} kWh  "
          f"peak_import={rl['peak_grid_import']:.0f} kW  net={rl['net_cost']:.0f} EUR")
    print(f"[diag] MILP : SC={milp['self_consumption_rate']:.1%}  import={milp['energy_imported_kwh']:.0f} kWh  "
          f"peak_import={milp['peak_grid_import']:.0f} kW  net={milp['net_cost']:.0f} EUR")

    # Le test set est un ÉCHANTILLON de journées-types (split mensuel) : jours ISOLÉS, non
    # contigus dans le calendrier. On trace donc sur un axe HEURES synthétique (jours test
    # concaténés) avec un séparateur + la date de chaque jour. --days 0 = tous les jours.
    ndays_total = T // steps_per_day
    ndays = ndays_total if args.days <= 0 else min(args.days, ndays_total)
    b = ndays * steps_per_day
    sl = slice(0, b)
    x = np.arange(b) * delta_t_h

    fig, axes = plt.subplots(4, 1, figsize=(max(12, 1.7 * ndays), 11), sharex=True)

    ax = axes[0]
    ax.plot(x, np.asarray(rlh["pv_t"])[sl], label="PV", color="orange")
    ax.plot(x, np.asarray(rlh["load_t"])[sl], label="Charge", color="black")
    ax.set_ylabel("kW"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title(f"Dispatch RL vs MILP — {args.run_dir}  ({ndays} jours test concaténés)")

    ax = axes[1]
    ax.plot(x, price_imp[sl], label="Prix import", color="firebrick")
    ax.plot(x, price_exp[sl], label="Prix export", color="seagreen")
    ax.set_ylabel("€/kWh"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, np.asarray(rlh["Pb_effective"])[sl], label="Pb RL", color="tab:blue")
    ax.plot(x, np.asarray(mh["Pb_effective"])[sl], label="Pb MILP", color="tab:red", ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Pb (kW)\n>0 décharge"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    axt = ax.twinx()
    axt.plot(x, 100 * np.asarray(rlh["soc"])[sl], color="tab:blue", alpha=0.35, lw=1)
    axt.plot(x, 100 * np.asarray(mh["soc"])[sl], color="tab:red", alpha=0.35, lw=1, ls="--")
    axt.set_ylabel("SoC (%)")

    ax = axes[3]
    ax.plot(x, rl_imp[sl], label="Import RL", color="tab:blue")
    ax.plot(x, -rl_exp[sl], label="Export RL", color="tab:blue", ls=":")
    ax.plot(x, np.asarray(mh["P_imp"])[sl], label="Import MILP", color="tab:red", ls="--")
    ax.plot(x, -np.asarray(mh["P_exp"])[sl], label="Export MILP", color="tab:red", ls="-.")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Réseau (kW)\n+import / -export"); ax.legend(loc="upper right", ncol=2); ax.grid(alpha=0.3)
    ax.set_xlabel("Heures (jours test concaténés)")

    # Séparateurs + date de chaque jour test (axe non calendaire).
    ymax0 = axes[0].get_ylim()[1]
    for d in range(ndays):
        h0 = d * steps_per_day * delta_t_h
        for a_ in axes:
            a_.axvline(h0, color="grey", lw=0.8, ls=":")
        axes[0].text(h0 + 0.3, ymax0 * 0.95,
                     pd.Timestamp(ts[d * steps_per_day]).strftime("%Y-%m-%d"),
                     fontsize=8, rotation=90, va="top")

    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "diagnose_dispatch.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[diag] figure → {out}")


if __name__ == "__main__":
    main()
