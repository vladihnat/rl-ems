"""SAC + ancre MILP persistante (WS-1c anti-washout).

Le pré-entraînement BC (``_bc_pretrain_actor``) n'initialise l'acteur qu'à *t=0* ; le gradient
SAC vanilla le « lave » en quelques milliers de pas et la politique retombe sur l'attracteur
~16 %. ``BCSAC`` maintient le signal « imite le MILP » **pendant tout** ``learn()`` via deux
mécanismes configurables, combinables (cf. ``temp_md/washoutBC.md``) :

1. **Buffer démo dédié** (DDPGfD / "Overcoming Exploration") : un 2ᵉ replay buffer ne contenant
   QUE des transitions MILP, jamais dilué. À chaque gradient step, une fraction fixe ``demo_frac``
   du minibatch en provient (le reste du buffer normal). Aucun changement de formule de loss —
   seulement quelles transitions alimentent le gradient (critique + acteur).
2. **Terme BC permanent dans la loss acteur** (style TD3+BC) :
   ``L_acteur = L_SAC + λ · MSE(μ(s_démo), atanh(a_milp))`` calculé en continu sur des paires
   ``(s, a)`` MILP. ``λ`` constant (``bc_lambda``) ou décroissant vers ``bc_lambda_final`` (ancre
   forte au début, relâchée à la fin).
3. **AWAC** (raffinement optionnel de 2, ``bc_awac=True``) : pondère chaque paire démo de la MSE par
   l'avantage estimé par le critique courant, ``w(s,a) = exp(A(s,a)/β)`` avec ``A = Q(s,a_milp) −
   Q(s,a_π)``. Là où le critique juge l'action MILP sous-optimale (ex. dump SoC fin de fenêtre du
   MILP myope), le poids tombe et la policy peut s'en écarter ; là où le MILP est bon, ``w`` est fort
   et l'imitation continue. L'avantage est standardisé (``β`` ~O(1) quelle que soit l'échelle des Q)
   et le poids clampé (``bc_awac_wmax``) + normalisé (préserve l'échelle de ``λ``).

L'état démo (``demo_buffer``, ``demo_obs``, ``demo_atanh_act``) est rempli APRÈS construction par
``agents.sac_agent.train_sac`` (qui a accès à l'env brut et au MILP), pas dans ``__init__``.

⚠ ``train()`` recopie le corps de ``SAC.train`` de **SB3 2.8.0** (pinné dans requirements.txt) avec
deux modifs minimales : le sampling mixte et le terme BC. Toute montée de version SB3 doit revérifier
cette copie.
"""

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class BCSAC(SAC):
    """SAC dont l'acteur reste ancré au dispatch MILP tout au long de ``learn()``."""

    def __init__(self, *args, demo_frac: float = 0.0, bc_lambda: float = 0.0,
                 bc_lambda_final=None, bc_loss_batch=None,
                 bc_awac: bool = False, bc_awac_beta: float = 1.0,
                 bc_awac_wmax: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Hyperparamètres de l'ancre (sauvés/restaurés par SB3 ; sans effet en éval).
        self.demo_frac = float(demo_frac)
        self.bc_lambda = float(bc_lambda)
        self.bc_lambda_final = bc_lambda_final
        self.bc_loss_batch = bc_loss_batch
        # AWAC (washoutBC.md pt 3) : pondère le terme BC par l'avantage critique exp(A/β) → relâche
        # l'ancre là où le critique juge l'action MILP sous-optimale (fenêtre myope). OFF par défaut
        # = comportement BC standard inchangé. N'a d'effet que si le terme BC est actif (λ>0).
        self.bc_awac = bool(bc_awac)
        self.bc_awac_beta = float(bc_awac_beta)
        self.bc_awac_wmax = float(bc_awac_wmax)
        # État démo lourd : rempli par train_sac, EXCLU de la sauvegarde (cf. _excluded_save_params).
        self.demo_buffer = None        # ReplayBuffer MILP dédié (approche 1)
        self.demo_obs = None           # tenseur (N, obs_dim) — états démo (approche 2)
        self.demo_atanh_act = None     # tenseur (N, act_dim) — cibles pré-tanh atanh(a_milp)

    def _excluded_save_params(self) -> list[str]:
        # L'éval recharge le best-model via ``type(model).load`` (run_experiment.py). On exclut
        # l'état démo (buffer + tenseurs) du pickle : non picklable proprement / inutile en éval,
        # et la construction throwaway de SB3.load les remet à None via __init__.
        return super()._excluded_save_params() + ["demo_buffer", "demo_obs", "demo_atanh_act"]

    def _current_bc_lambda(self) -> float:
        """λ courant : constant si ``bc_lambda_final is None``, sinon décroissance linéaire."""
        if self.bc_lambda_final is None:
            return self.bc_lambda
        # _current_progress_remaining : 1.0 au début → 0.0 à la fin (attribut SB3).
        frac = self._current_progress_remaining
        return float(self.bc_lambda_final) + (self.bc_lambda - float(self.bc_lambda_final)) * frac

    def _norm_demo_obs(self, s: th.Tensor) -> th.Tensor:
        """Normalise des obs démo BRUTES avec les stats VecNormalize VIVANTES (cas norm_obs=True).

        Les démos MILP sont collectées en obs brutes (cf. ``collect_milp_demos``) ; sous norm_obs la
        policy attend des obs normalisées. On réplique ``VecNormalize.normalize_obs`` en torch à
        partir des ``obs_rms`` COURANTS (mobile : relu à chaque step ⇒ même espace que celui où vit
        l'acteur). Le buffer démo (approche 1) est déjà normalisé au sampling via ``sample(env=...)``
        ; ce helper couvre le terme BC (approche 2) + les critiques AWAC. Identité si pas de
        VecNormalize ou norm_obs=False (rétro-compat stricte des runs en obs brutes).
        """
        venv = self._vec_normalize_env
        if venv is None or not getattr(venv, "norm_obs", False):
            return s
        mean = th.as_tensor(venv.obs_rms.mean, device=s.device, dtype=s.dtype)
        var = th.as_tensor(venv.obs_rms.var, device=s.device, dtype=s.dtype)
        return th.clamp((s - mean) / th.sqrt(var + venv.epsilon), -venv.clip_obs, venv.clip_obs)

    def _sample_mixed(self, batch_size: int) -> ReplayBufferSamples:
        """Minibatch = ``demo_frac`` du buffer démo + le reste du buffer de replay normal."""
        if self.demo_buffer is None or self.demo_frac <= 0.0:
            return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        n_demo = min(int(round(batch_size * self.demo_frac)), self.demo_buffer.size())
        n_main = batch_size - n_demo
        if n_demo == 0:
            return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        main = self.replay_buffer.sample(n_main, env=self._vec_normalize_env)
        demo = self.demo_buffer.sample(n_demo, env=self._vec_normalize_env)
        # discounts reste None (pas de n-step ici) → train() retombe sur self.gamma.
        return ReplayBufferSamples(
            observations=th.cat([main.observations, demo.observations], dim=0),
            actions=th.cat([main.actions, demo.actions], dim=0),
            next_observations=th.cat([main.next_observations, demo.next_observations], dim=0),
            dones=th.cat([main.dones, demo.dones], dim=0),
            rewards=th.cat([main.rewards, demo.rewards], dim=0),
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # === Copie de SAC.train (SB3 2.8.0) ; modifs balisées par "# [BCSAC]". ===
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        bc_losses = []  # [BCSAC] suivi du terme BC
        bc_awac_w_means, bc_awac_adv_stds = [], []  # [BCSAC] diagnostics AWAC

        for gradient_step in range(gradient_steps):
            # [BCSAC] sampling mixte buffer démo + buffer normal (au lieu de replay_buffer.sample).
            replay_data = self._sample_mixed(batch_size)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called entropy temperature or alpha
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # [BCSAC] terme BC permanent : ancre l'acteur sur le dispatch MILP (style TD3+BC).
            # Si bc_awac : chaque paire démo est pondérée par l'avantage critique exp(A/β)
            # (washoutBC.md pt 3) au lieu d'une MSE uniforme — l'ancre se relâche là où le critique
            # estime l'action MILP sous-optimale (ex. dump SoC fin de fenêtre du MILP myope).
            lam = self._current_bc_lambda()
            if self.demo_obs is not None and lam > 0.0:
                bs = int(self.bc_loss_batch) if self.bc_loss_batch else batch_size
                idx = th.randint(0, self.demo_obs.shape[0], (bs,), device=self.device)
                # Obs démo brutes → normalisées avec les stats VecNormalize vivantes (norm_obs=True),
                # sinon identité. Couvre l'acteur ET les critiques AWAC (q_demo/q_pi ci-dessous).
                s_demo = self._norm_demo_obs(self.demo_obs[idx])
                target = self.demo_atanh_act[idx]
                mean_actions, _, _ = self.actor.get_action_dist_params(s_demo)
                per_sample = ((mean_actions - target) ** 2).mean(dim=1)  # MSE par échantillon (bs,)
                if self.bc_awac:
                    with th.no_grad():
                        a_milp = th.tanh(target)  # action démo ∈[-1,1] (≈ a_milp)
                        q_demo = th.min(th.cat(self.critic(s_demo, a_milp), dim=1), dim=1).values
                        a_pi, _ = self.actor.action_log_prob(s_demo)  # V(s) ≈ Q(s, a_π)
                        q_pi = th.min(th.cat(self.critic(s_demo, a_pi), dim=1), dim=1).values
                        adv = q_demo - q_pi
                        adv_std = adv.std()
                        adv = (adv - adv.mean()) / (adv_std + 1e-8)  # standardisé → β O(1), scale-free
                        w = th.exp(adv / self.bc_awac_beta).clamp(max=self.bc_awac_wmax)
                        w = w / (w.mean() + 1e-8)  # normalisé → garde l'échelle de λ
                    bc_loss = (w * per_sample).mean()
                    bc_awac_w_means.append(float(w.mean()))
                    bc_awac_adv_stds.append(float(adv_std))
                else:
                    bc_loss = per_sample.mean()  # ≡ F.mse_loss (rétro-compat stricte)
                actor_loss = actor_loss + lam * bc_loss
                bc_losses.append(bc_loss.item())

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # [BCSAC] diagnostics de l'ancre.
        if len(bc_losses) > 0:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_lambda", self._current_bc_lambda())
        if len(bc_awac_w_means) > 0:
            self.logger.record("train/bc_awac_w_mean", np.mean(bc_awac_w_means))
            self.logger.record("train/bc_awac_adv_std", np.mean(bc_awac_adv_stds))
