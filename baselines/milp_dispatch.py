"""Warm-start SAC depuis le MILP : dispatch optimal roulant (par jour) → replay buffer.

L'attracteur RL plafonne car il ne découvre jamais la charge-depuis-réseau ni
l'auto-consommation optimale (cf. plan binary-churning-pretzel). On amorce donc le buffer
de SAC avec des transitions issues du **dispatch MILP** : un enseignant à foresight parfaite.

Choix « MILP roulant par jour » : un MILP plein-horizon sur ~28 k pas (une booléenne/pas)
est intraitable ; on résout donc le MILP **fenêtre par fenêtre de 24 h** (= l'horizon de
forecast que voit déjà l'agent), avec ``init_soc`` = SoC courant de l'env (SoC continu). On
**rejoue ensuite l'env** avec les actions dérivées pour enregistrer les transitions RÉELLES
``(obs, action, reward, next_obs, done)`` — l'env fournit la dynamique/le reward exacts, le
MILP ne fournit que l'action cible.

Réutilise tel quel le solveur vérifié ``baselines.milp_solver.run_milp`` via un shim env
minimal (aucune copie de la formulation, aucun changement du solveur).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.milp_solver import run_milp


class _WindowEnv:
    """Shim exposant EXACTEMENT ce que ``run_milp`` lit, sur des tableaux de fenêtre.

    ``timestamps`` est requis par ``run_milp`` (calcul des métriques by_season). Dans le chemin
    fenêtré ces métriques sont ignorées (on ne lit que ``Pb_effective``), mais le shim doit les
    fournir pour ne pas planter : on passe la tranche de timestamps réelle de la fenêtre, ou un
    repère synthétique de bonne longueur en dernier recours.
    """

    def __init__(self, pv, load, price_imp, price_exp, timestamps=None):
        pv = np.asarray(pv, dtype=np.float64)
        load = np.asarray(load, dtype=np.float64)
        self.max_steps = len(pv)
        if timestamps is None:
            timestamps = np.arange(len(pv), dtype="datetime64[h]")
        self.pv = SimpleNamespace(get_irradiance=lambda t, _a=pv: float(_a[t]),
                                  timestamps=np.asarray(timestamps))
        self.load = SimpleNamespace(get_load=lambda t, _a=load: float(_a[t]))
        self.price_signal = SimpleNamespace(
            import_prices=np.asarray(price_imp, dtype=np.float64),
            export_prices=np.asarray(price_exp, dtype=np.float64),
        )


def solve_milp_window(pv, load, price_imp, price_exp, init_soc: float, config: dict,
                      timestamps=None) -> np.ndarray:
    """Dispatch batterie ``Pb`` (kW, >0 = décharge) optimal sur la fenêtre via ``run_milp``."""
    cfg = copy.deepcopy(config)
    cfg["battery"]["init_soc"] = float(init_soc)
    shim = _WindowEnv(pv, load, price_imp, price_exp, timestamps=timestamps)
    metrics = run_milp(shim, cfg)
    return np.asarray(metrics["history"]["Pb_effective"], dtype=np.float64)


def _pb_to_action(pb: float, max_charge: float, max_discharge: float) -> np.ndarray:
    """Inverse de base_microgrid_env.step : action ∈ [-1,1] reproduisant ``pb`` (kW)."""
    denom = max_discharge if pb > 0.0 else max_charge
    return np.array([float(np.clip(pb / denom, -1.0, 1.0))], dtype=np.float32)


def iter_milp_actions(env, config: dict, max_windows=None, window_days: int = 1):
    """Génère ``(action, pb_milp)`` en avançant l'env sous dispatch MILP roulant.

    ``window_days`` = taille de la fenêtre MILP en jours (défaut 1). >1 enseigne l'arbitrage
    multi-jours (charger la nuit pour le lendemain), que le MILP 1-jour (qui vide la batterie
    chaque jour) ne montre pas. ⚠ Avec le dataset actuel (blocs de jours non contigus), garder
    ``window_days`` diviseur de la taille de bloc pour ne pas franchir une frontière de bloc.

    L'env est resetté puis stepé en interne : à utiliser sur l'env d'entraînement. ``env``
    est laissé en fin d'épisode (le caller — SAC — le re-resette à ``learn``).
    """
    steps_per_day = int(24 * 60 / config["time"]["delta_t_min"])
    window_steps = max(1, int(window_days)) * steps_per_day
    max_charge = config["battery"]["max_charge_kw"]
    max_discharge = config["battery"]["max_discharge_kw"]

    env.reset()
    total = int(env.max_steps)
    idx, n_win = 0, 0
    while idx < total:
        if max_windows is not None and n_win >= max_windows:
            return
        w = min(window_steps, total - idx)
        pv_w = np.array([env.pv.get_irradiance(idx + k) for k in range(w)], dtype=np.float64)
        load_w = np.array([env.load.get_load(idx + k) for k in range(w)], dtype=np.float64)
        pimp_w = np.asarray(env.price_signal.import_prices[idx:idx + w], dtype=np.float64)
        pexp_w = np.asarray(env.price_signal.export_prices[idx:idx + w], dtype=np.float64)
        ts_w = np.asarray(env.pv.timestamps)[idx:idx + w]
        Pb = solve_milp_window(pv_w, load_w, pimp_w, pexp_w, float(env.battery.soc), config,
                               timestamps=ts_w)
        for k in range(w):
            pb = float(Pb[k])
            yield _pb_to_action(pb, max_charge, max_discharge), pb
        idx += w
        n_win += 1


def prefill_replay_buffer_with_milp(model, env, config: dict, max_windows=None,
                                    verbose: bool = True, window_days: int = 1,
                                    buffer=None) -> int:
    """Remplit un replay buffer avec les transitions réelles du dispatch MILP roulant.

    Args:
        model: SAC SB3 (sert à dériver l'espace obs/action ; ``model.replay_buffer`` par défaut).
        env:   env Gymnasium BRUT d'entraînement (obs non normalisées — cf. garde dans sac_agent).
        config: config complète de l'expérience.
        max_windows: limite le nb de fenêtres amorcées (None = tout l'épisode train).
        window_days: taille de la fenêtre MILP en jours (cf. ``iter_milp_actions``).
        buffer: replay buffer cible (None = ``model.replay_buffer``). Permet de remplir un
            **buffer démo dédié** (WS-1c, ancre persistante) avec exactement ce code vérifié.

    Returns:
        Nombre de transitions ajoutées.
    """
    buf = buffer if buffer is not None else model.replay_buffer
    steps_per_day = int(24 * 60 / config["time"]["delta_t_min"])
    window_steps = max(1, int(window_days)) * steps_per_day
    max_charge = config["battery"]["max_charge_kw"]
    max_discharge = config["battery"]["max_discharge_kw"]

    obs, _ = env.reset()
    total = int(env.max_steps)
    idx, n_win, n_added = 0, 0, 0
    while idx < total:
        if max_windows is not None and n_win >= max_windows:
            break
        w = min(window_steps, total - idx)
        pv_w = np.array([env.pv.get_irradiance(idx + k) for k in range(w)], dtype=np.float64)
        load_w = np.array([env.load.get_load(idx + k) for k in range(w)], dtype=np.float64)
        pimp_w = np.asarray(env.price_signal.import_prices[idx:idx + w], dtype=np.float64)
        pexp_w = np.asarray(env.price_signal.export_prices[idx:idx + w], dtype=np.float64)
        ts_w = np.asarray(env.pv.timestamps)[idx:idx + w]
        Pb = solve_milp_window(pv_w, load_w, pimp_w, pexp_w, float(env.battery.soc), config,
                               timestamps=ts_w)

        done = False
        for k in range(w):
            action = _pb_to_action(float(Pb[k]), max_charge, max_discharge)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buf.add(
                np.asarray(obs, dtype=np.float32).reshape(1, -1),
                np.asarray(next_obs, dtype=np.float32).reshape(1, -1),
                action.reshape(1, -1),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=bool),
                [info],
            )
            obs = next_obs
            n_added += 1
            if done:
                break
        idx += w
        n_win += 1
        if done:
            break

    if verbose:
        print(f"[warm-start] MILP prefill: {n_added} transitions / {n_win} fenêtres "
              f"(replay_buffer size = {buf.size()})")
    return n_added


def collect_milp_demos(env, config: dict, window_days: int = 1, max_windows=None):
    """Rejoue le dispatch MILP roulant sur l'env et renvoie les paires ``(obs, action)``.

    Dataset d'imitation pour le **pré-entraînement supervisé de l'acteur** (BC) : ``obs`` est
    enregistrée AVANT le step (l'état où l'action MILP est prise), ``action`` est l'action
    normalisée ∈[-1,1] reproduisant le ``Pb`` MILP. Obs BRUTES (norm_obs=False requis, cf.
    garde dans ``sac_agent``). Ne touche pas au replay buffer — complémentaire du warm-start.

    Returns:
        ``(obs[N, obs_dim], actions[N, 1])`` en float32.
    """
    steps_per_day = int(24 * 60 / config["time"]["delta_t_min"])
    window_steps = max(1, int(window_days)) * steps_per_day
    max_charge = config["battery"]["max_charge_kw"]
    max_discharge = config["battery"]["max_discharge_kw"]

    obs, _ = env.reset()
    total = int(env.max_steps)
    idx, n_win = 0, 0
    obs_list, act_list = [], []
    done = False
    while idx < total:
        if max_windows is not None and n_win >= max_windows:
            break
        w = min(window_steps, total - idx)
        pv_w = np.array([env.pv.get_irradiance(idx + k) for k in range(w)], dtype=np.float64)
        load_w = np.array([env.load.get_load(idx + k) for k in range(w)], dtype=np.float64)
        pimp_w = np.asarray(env.price_signal.import_prices[idx:idx + w], dtype=np.float64)
        pexp_w = np.asarray(env.price_signal.export_prices[idx:idx + w], dtype=np.float64)
        ts_w = np.asarray(env.pv.timestamps)[idx:idx + w]
        Pb = solve_milp_window(pv_w, load_w, pimp_w, pexp_w, float(env.battery.soc), config,
                               timestamps=ts_w)
        for k in range(w):
            action = _pb_to_action(float(Pb[k]), max_charge, max_discharge)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if bool(terminated or truncated):
                done = True
                break
        idx += w
        n_win += 1
        if done:
            break

    if not obs_list:
        return (np.zeros((0,) + env.observation_space.shape, dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32))
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def check_reproduction(config_path: str, max_windows: int = 5) -> float:
    """Vérif : rejouer l'env sous actions dérivées reproduit le ``Pb`` MILP (à la tol. batterie)."""
    from envs.registry import make_env

    train_env, _, _, cfg = make_env(config_path, with_val=True)
    max_charge = cfg["battery"]["max_charge_kw"]
    max_discharge = cfg["battery"]["max_discharge_kw"]
    steps_per_day = int(24 * 60 / cfg["time"]["delta_t_min"])

    train_env.reset()
    total = int(train_env.max_steps)
    idx, n_win, n, max_diff = 0, 0, 0, 0.0
    while idx < total and n_win < max_windows:
        w = min(steps_per_day, total - idx)
        pv_w = np.array([train_env.pv.get_irradiance(idx + k) for k in range(w)], dtype=np.float64)
        load_w = np.array([train_env.load.get_load(idx + k) for k in range(w)], dtype=np.float64)
        pimp_w = np.asarray(train_env.price_signal.import_prices[idx:idx + w], dtype=np.float64)
        pexp_w = np.asarray(train_env.price_signal.export_prices[idx:idx + w], dtype=np.float64)
        ts_w = np.asarray(train_env.pv.timestamps)[idx:idx + w]
        Pb = solve_milp_window(pv_w, load_w, pimp_w, pexp_w, float(train_env.battery.soc), cfg,
                               timestamps=ts_w)
        for k in range(w):
            pb = float(Pb[k])
            action = _pb_to_action(pb, max_charge, max_discharge)
            _, _, term, trunc, info = train_env.step(action)
            max_diff = max(max_diff, abs(float(info["Pb_effective"]) - pb))
            n += 1
            if term or trunc:
                break
        idx += w
        n_win += 1
    print(f"[check] {n_win} jours / {n} pas — max |Pb_env - Pb_milp| = {max_diff:.4f} kW "
          f"(écart attendu petit, non nul aux saturations SoC)")
    return max_diff


def run_receding_horizon_milp(env, config: dict, horizon_steps=None,
                              replan_every=None, verbose: bool = True) -> dict:
    """Oracle MILP à horizon glissant = contrôleur à information CAUSALE (comme le RL).

    À chaque ``replan_every`` pas, résout le MILP sur ``horizon_steps`` pas devant (init_soc =
    SoC courant de l'env), committe les ``replan_every`` premières actions, avance, re-résout.
    Défauts : ``horizon_steps = env.horizon_steps`` (24 h, = le lookahead de l'obs RL),
    ``replan_every = 1`` (vrai horizon glissant ; mettre = steps/jour pour une version rapide).

    À utiliser sur le **test env** (reset → init_soc fixe, comme le MILP global). Renvoie les
    mêmes métriques que ``run_milp`` (via compute_metrics). L'écart oracle↔MILP global = part du
    gap due au **seul horizon de prévision** (irréductible) ; le reste du gap RL est réductible
    par l'apprentissage (même information disponible).
    """
    from evaluation.metrics import compute_metrics

    steps_per_day = int(24 * 60 / config["time"]["delta_t_min"])
    H = int(horizon_steps) if horizon_steps else int(env.horizon_steps)
    K = int(replan_every) if replan_every else 1
    max_charge = config["battery"]["max_charge_kw"]
    max_discharge = config["battery"]["max_discharge_kw"]

    history = {k: [] for k in ("P_grid", "Pb_effective", "soc", "pv_t", "load_t",
                               "r_eco", "r_soc", "reward", "P_phantom", "Pcurt")}
    env.reset()
    total = int(env.max_steps)
    idx, n_solve, done = 0, 0, False
    while idx < total and not done:
        h = min(H, env.pv.n_steps - idx)   # lookahead dispo (données jusqu'à n_steps)
        pv_w = np.array([env.pv.get_irradiance(idx + k) for k in range(h)], dtype=np.float64)
        load_w = np.array([env.load.get_load(idx + k) for k in range(h)], dtype=np.float64)
        pimp_w = np.asarray(env.price_signal.import_prices[idx:idx + h], dtype=np.float64)
        pexp_w = np.asarray(env.price_signal.export_prices[idx:idx + h], dtype=np.float64)
        ts_w = np.asarray(env.pv.timestamps)[idx:idx + h]
        Pb = solve_milp_window(pv_w, load_w, pimp_w, pexp_w, float(env.battery.soc), config,
                               timestamps=ts_w)
        n_solve += 1
        for k in range(min(K, total - idx)):
            action = _pb_to_action(float(Pb[k]), max_charge, max_discharge)
            _obs, reward, terminated, truncated, info = env.step(action)
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
            if bool(terminated or truncated):
                done = True
                break
        idx += min(K, total - idx)

    for key in history:
        history[key] = np.asarray(history[key])
    T = len(history["P_grid"])
    metrics = compute_metrics(
        history, env.delta_t_h,
        env.cfg["reward"]["soc_safe_min"], env.cfg["reward"]["soc_safe_max"],
        env.price_signal.import_prices[:T], env.price_signal.export_prices[:T],
        env.cfg["grid"].get("phantom_penalty", 1e3),
    )
    metrics["history"] = history
    metrics["n_solve"] = n_solve
    if verbose:
        print(f"[oracle] H={H} replan_every={K} → {n_solve} solves | "
              f"net={metrics['net_cost']:.2f} import={metrics['energy_imported_kwh']:.1f} "
              f"SC={metrics['self_consumption_rate']*100:.1f}% peak={metrics['peak_grid_import']:.1f} "
              f"served={metrics['served_load_ratio']:.4f}")
    return metrics


def run_oracle_vs_global(config_path: str, replan_every=None, horizon_steps=None) -> dict:
    """Compare l'oracle 24 h (info causale) au MILP global (foresight parfaite) sur le test."""
    from envs.registry import make_env
    from baselines.milp_solver import run_milp

    _, _, test_env, cfg = make_env(config_path, with_val=True)
    glob = run_milp(test_env, cfg)
    # MILP global ne step pas l'env → le re-faire pour l'oracle est sûr.
    _, _, test_env2, _ = make_env(config_path, with_val=True)
    orac = run_receding_horizon_milp(test_env2, cfg, horizon_steps=horizon_steps,
                                     replan_every=replan_every)
    gn, on = glob["net_cost"], orac["net_cost"]
    gap = (on - gn) / abs(gn) if abs(gn) > 1e-9 else float("inf")
    print("\n=== Oracle 24h (causal) vs MILP global (perfect foresight) ===")
    print(f"  global : net={gn:.2f}  import={glob['energy_imported_kwh']:.1f}  "
          f"SC={glob['self_consumption_rate']*100:.1f}%  peak={glob['peak_grid_import']:.1f}")
    print(f"  oracle : net={on:.2f}  import={orac['energy_imported_kwh']:.1f}  "
          f"SC={orac['self_consumption_rate']*100:.1f}%  peak={orac['peak_grid_import']:.1f}")
    print(f"  -> écart oracle↔global = {gap:+.1%}  (= part IRRÉDUCTIBLE due à l'horizon 24h)")
    print(f"     rappel : attracteur RL ≈ +16.6% ⇒ part réductible (apprentissage) ≈ "
          f"{(0.166 - gap)*100:.1f} pp")
    return {"global": glob, "oracle": orac, "gap_oracle_vs_global": gap}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MILP roulant : warm-start, vérif repro, oracle 24h.")
    ap.add_argument("--config", default="configs/exp14_Batt400_explore.yaml")
    ap.add_argument("--max-windows", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="Vérifier la reproduction Pb env vs MILP.")
    ap.add_argument("--oracle", action="store_true", help="Oracle 24h glissant vs MILP global (test).")
    ap.add_argument("--replan-every", type=int, default=None,
                    help="Pas entre 2 re-solves (défaut 1 = vrai glissant ; =96 ≈ journalier rapide).")
    ap.add_argument("--horizon-steps", type=int, default=None,
                    help="Lookahead du MILP roulant (défaut = env.horizon_steps = 24h).")
    args = ap.parse_args()
    if args.oracle:
        run_oracle_vs_global(args.config, replan_every=args.replan_every,
                             horizon_steps=args.horizon_steps)
    elif args.check:
        check_reproduction(args.config, args.max_windows)
    else:
        print("Rien à faire : passez --check (repro) ou --oracle (oracle 24h vs global).")
