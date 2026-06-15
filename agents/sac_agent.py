"""SAC agent: training and evaluation using Stable-Baselines3."""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.common import build_callback
from evaluation.metrics import compute_metrics, metrics_by_season, season_labels


def _bc_pretrain_actor(model, env, config: dict, epochs: int, lr: float,
                       batch_size: int, window_days: int, verbose: bool = True) -> int:
    """Pré-entraîne l'acteur SAC par imitation (BC) du dispatch MILP, AVANT le RL.

    Régression supervisée dans l'espace **pré-tanh** : ``μ(obs) ≈ atanh(a_MILP)``. On NE régresse
    PAS sur l'action squashée ``tanh(μ)`` : avec des cibles bang-bang (beaucoup de ±1), la MSE
    pousse μ→±∞, ``tanh`` sature et le gradient s'annule (acteur figé à ±1). Régresser sur
    ``atanh`` (linéaire en μ) évite cette pathologie. L'espace d'action est [-1,1] donc l'action
    déterministe de SAC est exactement ``tanh(μ)`` (pas de rescale). Met le comportement de
    charge-réseau dans les poids de la policy (non dilué, contrairement au warm-start) ; le
    critique reste vierge — SAC l'apprend pendant ``learn``.
    """
    import torch as th
    from baselines.milp_dispatch import collect_milp_demos

    obs, act = collect_milp_demos(env, config, window_days=window_days)
    n = len(obs)
    if n == 0:
        print("[bc-pretrain] SKIP : aucune démo MILP collectée.")
        return 0
    device = model.device
    obs_t = th.as_tensor(obs, device=device)
    # Cible pré-tanh : atanh borné (atanh(±1)=±∞). 1e-3 → atanh≈±3.8, atteignable par μ.
    act_t = th.as_tensor(act, device=device).clamp(-1.0 + 1e-3, 1.0 - 1e-3)
    target_pre = th.atanh(act_t)
    optim = th.optim.Adam(model.actor.parameters(), lr=lr)
    bs = max(1, min(int(batch_size), n))
    model.policy.set_training_mode(True)
    for ep in range(int(epochs)):
        perm = th.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            mean_actions, _log_std, _ = model.actor.get_action_dist_params(obs_t[sel])
            loss = th.nn.functional.mse_loss(mean_actions, target_pre[sel])
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * len(sel)
        if verbose:
            # MSE rapportée sur l'ACTION squashée (interprétable, ∈[0,4]).
            with th.no_grad():
                mu, _, _ = model.actor.get_action_dist_params(obs_t)
                a_mse = float(th.nn.functional.mse_loss(th.tanh(mu), th.as_tensor(act, device=device)))
            print(f"[bc-pretrain] epoch {ep + 1}/{int(epochs)}  action_mse={a_mse:.5f}")
    print(f"[bc-pretrain] acteur pré-entraîné sur {n} paires (obs, action) MILP.")
    return n


def train_sac(env, config: dict, eval_env=None, output_dir=None):
    """Train a SAC agent on the given environment.

    Args:
        eval_env: optional validation env. Si fourni (avec ``output_dir``), un callback
            sélectionne le meilleur checkpoint sur le coût net ajusté de validation
            (best_model.zip + best_vecnormalize.pkl), au lieu de ne garder que le modèle final.

    Returns:
        (model, episode_rewards, vec_env): trained SB3 model, list of episode rewards,
        and the VecNormalize wrapper (or None if norm_obs/norm_reward are both False).
    """
    t_cfg = config["training"]
    norm_obs    = t_cfg.get("norm_obs",    False)
    norm_reward = t_cfg.get("norm_reward", False)

    if norm_obs or norm_reward:
        vec_env   = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=norm_obs, norm_reward=norm_reward)
        train_env = vec_env
    else:
        vec_env   = None
        train_env = env

    # Architecture réseau optionnelle (défaut SB3 = [256, 256] si non spécifiée).
    policy_kwargs = {}
    net_arch = t_cfg.get("net_arch")
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)

    # Tous les kwargs SAC tunables : leurs défauts reproduisent les défauts SB3
    # pour préserver la rétro-compat des configs qui ne les déclarent pas.
    sac_kwargs = dict(
        learning_rate=t_cfg["learning_rate"],
        batch_size=t_cfg["batch_size"],
        buffer_size=t_cfg["buffer_size"],
        gamma=t_cfg.get("gamma", 0.99),
        tau=t_cfg.get("tau", 0.005),
        ent_coef=t_cfg.get("ent_coef", "auto"),
        train_freq=t_cfg.get("train_freq", 1),
        gradient_steps=t_cfg.get("gradient_steps", 1),
        learning_starts=t_cfg.get("learning_starts", 100),
        seed=config["experiment"]["seed"],
        verbose=1,
    )
    if policy_kwargs:
        sac_kwargs["policy_kwargs"] = policy_kwargs

    # WS-1c — ancre MILP persistante (anti-washout) : maintient le signal « imite le MILP »
    # pendant tout learn() (buffer démo dédié et/ou terme BC dans la loss acteur), au lieu de
    # ne l'injecter qu'à l'init (bc_pretrain) ou dilué une fois (warm_start). Combinable avec eux.
    # Comme le warm_start/bc_pretrain, exige des obs BRUTES (norm_obs=norm_reward=False).
    bc_anchor_demo_buffer = bool(t_cfg.get("bc_anchor_demo_buffer", False))
    bc_anchor_loss        = bool(t_cfg.get("bc_anchor_loss", False))
    anchor_requested = bc_anchor_demo_buffer or bc_anchor_loss
    anchor_active = anchor_requested and not (norm_obs or norm_reward)
    if anchor_requested and not anchor_active:
        print("[bc-anchor] SKIP : bc_anchor_* exige norm_obs=norm_reward=False (obs démo brutes).")

    # AWAC (washoutBC.md pt 3) : pondère le terme BC par l'avantage critique. N'a d'effet que si le
    # terme BC est actif (bc_anchor_loss=True) — sinon il n'y a pas de paires démo à pondérer.
    bc_anchor_awac = bool(t_cfg.get("bc_anchor_awac", False))
    if bc_anchor_awac and not bc_anchor_loss:
        print("[bc-anchor] AWAC ignoré : bc_anchor_awac exige bc_anchor_loss=True (terme BC actif).")
        bc_anchor_awac = False

    if anchor_active:
        from agents.bc_sac import BCSAC
        model = BCSAC(
            "MlpPolicy", train_env,
            demo_frac=float(t_cfg.get("bc_anchor_demo_frac", 0.25)) if bc_anchor_demo_buffer else 0.0,
            bc_lambda=float(t_cfg.get("bc_anchor_lambda", 1.0)) if bc_anchor_loss else 0.0,
            bc_lambda_final=t_cfg.get("bc_anchor_lambda_final") if bc_anchor_loss else None,
            bc_loss_batch=t_cfg.get("bc_anchor_loss_batch"),
            bc_awac=bc_anchor_awac,
            bc_awac_beta=float(t_cfg.get("bc_anchor_awac_beta", 1.0)),
            bc_awac_wmax=float(t_cfg.get("bc_anchor_awac_wmax", 20.0)),
            **sac_kwargs,
        )
    else:
        model = SAC("MlpPolicy", train_env, **sac_kwargs)

    # Fenêtre du MILP enseignant (jours) — partagée par le warm-start et le BC. >1 = arbitrage
    # multi-jours (charger la nuit pour le lendemain), invisible avec une fenêtre 1-jour.
    milp_window_days = int(t_cfg.get("milp_window_days", 1))

    # Warm-start optionnel : amorcer le replay buffer avec le dispatch MILP roulant pour injecter
    # le comportement d'arbitrage que l'exploration ne découvre pas. Requiert des obs BRUTES
    # (norm_obs=norm_reward=False) car les transitions stockées doivent matcher l'entrée policy.
    if t_cfg.get("warm_start_milp", False):
        if norm_obs or norm_reward:
            print("[warm-start] SKIP : warm_start_milp exige norm_obs=norm_reward=False "
                  "(obs du buffer ≠ entrée policy sinon).")
        else:
            from baselines.milp_dispatch import prefill_replay_buffer_with_milp
            prefill_replay_buffer_with_milp(model, env, config,
                                            max_windows=t_cfg.get("warm_start_max_days"),
                                            window_days=milp_window_days)

    # Pré-entraînement BC de l'acteur (le plus haut potentiel : non dilué dans le buffer).
    # Mêmes obs brutes requises que le warm-start.
    bc_epochs = int(t_cfg.get("bc_pretrain_epochs", 0))
    if bc_epochs > 0:
        if norm_obs or norm_reward:
            print("[bc-pretrain] SKIP : bc_pretrain_epochs exige norm_obs=norm_reward=False.")
        else:
            _bc_pretrain_actor(
                model, env, config,
                epochs=bc_epochs,
                lr=float(t_cfg.get("bc_pretrain_lr", t_cfg["learning_rate"])),
                batch_size=int(t_cfg.get("bc_pretrain_batch", t_cfg["batch_size"])),
                window_days=milp_window_days,
            )

    # Remplissage de l'état démo de l'ancre (après l'init bc_pretrain, avant learn). Le buffer
    # démo et le terme BC réutilisent les mêmes démos MILP : si le buffer est rempli, on en dérive
    # les paires (obs, action) du terme BC pour éviter un 2e solve MILP.
    if anchor_active:
        import torch as th
        demo_buffer = None
        if bc_anchor_demo_buffer:
            from stable_baselines3.common.buffers import ReplayBuffer
            from baselines.milp_dispatch import prefill_replay_buffer_with_milp
            demo_buffer = ReplayBuffer(
                buffer_size=int(env.max_steps) + 1,
                observation_space=model.observation_space,
                action_space=model.action_space,
                device=model.device,
                n_envs=1,
                handle_timeout_termination=False,
            )
            prefill_replay_buffer_with_milp(model, env, config, buffer=demo_buffer,
                                            window_days=milp_window_days)
            model.demo_buffer = demo_buffer
        if bc_anchor_loss:
            if demo_buffer is not None:
                size = demo_buffer.size()
                obs = np.asarray(demo_buffer.observations[:size, 0])
                act = np.asarray(demo_buffer.actions[:size, 0])
            else:
                from baselines.milp_dispatch import collect_milp_demos
                obs, act = collect_milp_demos(env, config, window_days=milp_window_days)
            obs_t = th.as_tensor(obs, device=model.device).float()
            act_t = th.as_tensor(act, device=model.device).float().clamp(-1.0 + 1e-3, 1.0 - 1e-3)
            model.demo_obs = obs_t
            model.demo_atanh_act = th.atanh(act_t)
            lam_msg = f"{model.bc_lambda}" + (f"→{model.bc_lambda_final}"
                                              if model.bc_lambda_final is not None else " (constant)")
            awac_msg = (f", AWAC β={model.bc_awac_beta} wmax={model.bc_awac_wmax}"
                        if model.bc_awac else "")
            print(f"[bc-anchor] terme BC permanent sur {obs_t.shape[0]} paires (obs, action) MILP, "
                  f"λ={lam_msg}{awac_msg}")

    callbacks, reward_logger = build_callback(eval_env, t_cfg, output_dir, t_cfg["total_timesteps"])
    model.learn(total_timesteps=t_cfg["total_timesteps"], callback=callbacks)

    return model, reward_logger.episode_rewards, vec_env


def evaluate_sac(model, env, vec_normalize=None) -> dict:
    """Run the trained policy on an environment and return metrics.

    Args:
        vec_normalize: optional VecNormalize (in eval mode) used to normalise
            observations before feeding them to the model. The raw Gymnasium env
            is still used for stepping and attribute access.

    Returns:
        dict with episode history and computed metrics.
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
    labels = season_labels(env.pv.timestamps[:T])
    metrics["by_season"] = metrics_by_season(history, labels, delta_t_h, soc_min,
                                             soc_max, price_import, price_export,
                                             phantom_penalty)
    metrics["history"] = history
    return metrics
