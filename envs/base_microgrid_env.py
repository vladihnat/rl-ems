"""Gymnasium environment for microgrid energy management.

Action: normalized battery command in [-1, 1].
Observation: temporal features + system state + economic signals + PV forecast.
Reward: economic cost/revenue + SoC penalty.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.components.battery import BatteryModel
from envs.components.load import LoadModel
from envs.components.price_signal import PriceSignal
from envs.components.pv_source import PVSource
from monitoring.monitoring_table import MonitoringTable


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        pv_source: PVSource,
        load_model: LoadModel,
        battery: BatteryModel,
        price_signal: PriceSignal,
        config: dict,
    ):
        super().__init__()
        self.pv = pv_source
        self.load = load_model
        self.battery = battery
        self.price_signal = price_signal
        self.cfg = config

        self.delta_t_min = config["time"]["delta_t_min"]
        self.delta_t_h = self.delta_t_min / 60.0
        self.horizon_steps = int(config["time"]["horizon_h"] * 60 / self.delta_t_min)

        self.max_import_kw = config["grid"]["max_import_kw"]
        self.max_export_kw = config["grid"]["max_export_kw"]

        # Pénalité de puissance fantôme (charge non servie) — identique à celle du MILP
        # (baselines/milp_solver.py) pour que RL et MILP modélisent la même physique.
        self.phantom_penalty = config["grid"].get("phantom_penalty", 1e3)

        self.curtailment_mode = config["grid"].get("curtailment", "clip")
        if self.curtailment_mode not in ("clip", "penal"):
            raise ValueError(
                f"Unknown grid.curtailment={self.curtailment_mode!r}; use 'clip' or 'penal'."
            )

        self.sigma_soc = config["reward"]["sigma_soc"]
        self.soc_safe_min = config["reward"]["soc_safe_min"]
        self.soc_safe_max = config["reward"]["soc_safe_max"]

        self.max_charge_kw = config["battery"]["max_charge_kw"]
        self.max_discharge_kw = config["battery"]["max_discharge_kw"]

        self._load_forecast_in_obs = config.get("observation", {}).get("load_forecast", False)
        self._price_export_fc_in_obs = config.get("observation", {}).get("price_export_forecast", False)
        # Feature de timing (optimum-safe, pur input — ne touche ni reward ni dynamique) : « gap à
        # pic » = de combien l'énergie vaut PLUS au meilleur moment futur de l'horizon que maintenant
        # (cf. _get_obs). Donne explicitement le signal « dois-je attendre ? » que l'agent devait
        # sinon extraire lui-même du vecteur forecast (et ratait → dump t0). Exige un forecast prix
        # (has_forecast) ; inerte sous prix fixes (=0). Défaut OFF ⇒ obs des configs existantes inchangées.
        self._timing_feat = config.get("observation", {}).get("timing_feature", False)
        self._spread_penalty = config["reward"].get("spread_penalty", False)
        self._sigma_bat = config["reward"].get("sigma_bat", 0.0)
        self._random_soc = config.get("training", {}).get("random_soc", False)
        self._pb_max = max(self.max_charge_kw, self.max_discharge_kw)

        # PBRS — valeur prix-aware du stock (Ng 1999). σ_store = force du shaping ; γ doit être
        # CELUI de SAC (cf. sac_agent.py) pour que F = γ·Φ(s')−Φ(s) préserve exactement l'optimum.
        self._store_shaping = config["reward"].get("store_value", False)
        self._sigma_store = config["reward"].get("sigma_store", 1.0)
        self._gamma_shaping = config.get("training", {}).get("gamma", 0.99)

        # Pénalité one-sided sur l'export DE STOCK (coût d'opportunité prix-aware). 0.0 = inactif
        # (opt-in, runs existants bit-identiques). Zéro sur l'optimum ⇒ ne déplace pas l'optimum
        # (même philosophie que spread_penalty). Décourage de vendre le stock quand le garder
        # éviterait un import futur plus cher (cf. _export_opportunity_cost).
        self._sigma_export_stock = config["reward"].get("sigma_export_stock", 0.0)

        price_forecast_dim = self.horizon_steps if self.price_signal.has_forecast else 0
        load_forecast_dim = self.horizon_steps if self._load_forecast_in_obs else 0
        export_forecast_dim = (
            self.horizon_steps
            if (self._price_export_fc_in_obs and self.price_signal.has_forecast)
            else 0
        )
        timing_dim = 1 if (self._timing_feat and self.price_signal.has_forecast) else 0
        obs_dim = (
            9 + self.horizon_steps + load_forecast_dim
            + price_forecast_dim + export_forecast_dim + timing_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.max_steps = self.pv.n_steps - self.horizon_steps
        self.step_index = 0

        # Monitoring buffer — accumulates one row per call to step() during a
        # post-training rollout. See monitoring/monitoring_table.py.
        # Allocated lazily on reset() once we know the rollout length.
        self.monitoring_table: MonitoringTable | None = None

    def _get_obs(self) -> np.ndarray:
        h_sin, h_cos, d_sin, d_cos = self.pv.get_temporal_features(self.step_index)
        soc = self.battery.soc
        load_t = self.load.get_load(self.step_index)
        pv_t = self.pv.get_irradiance(self.step_index)
        pv_forecast = self.pv.get_forecast(self.step_index, self.horizon_steps)
        p_imp = self.price_signal.get_import_price(self.step_index)
        p_exp = self.price_signal.get_export_price(self.step_index)

        parts = [
            np.array([h_sin, h_cos, d_sin, d_cos], dtype=np.float32),
            np.array([soc, load_t], dtype=np.float32),
            np.array([p_imp, p_exp], dtype=np.float32),
            np.array([pv_t], dtype=np.float32),
            pv_forecast,
        ]
        if self._load_forecast_in_obs:
            parts.append(self.load.get_forecast(self.step_index, self.horizon_steps))
        if self.price_signal.has_forecast:
            parts.append(self.price_signal.get_import_forecast(self.step_index, self.horizon_steps))
        if self._price_export_fc_in_obs and self.price_signal.has_forecast:
            parts.append(self.price_signal.get_export_forecast(self.step_index, self.horizon_steps))
        if self._timing_feat and self.price_signal.has_forecast:
            # Gap à pic ≥ 0 (=0 quand maintenant EST le meilleur moment de l'horizon) : meilleure
            # valeur de déploiement future (_store_value = max forecast imp/exp) moins celle de
            # maintenant (max prix instantanés). Pré-calcule l'argmax que l'agent ratait.
            parts.append(np.array(
                [self._store_value(self.step_index) - max(p_imp, p_exp)], dtype=np.float32
            ))

        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        if self._random_soc:
            eps = 0.05
            self.battery.soc = float(self.np_random.uniform(
                self.battery.soc_min + eps,
                self.battery.soc_max - eps,
            ))
        else:
            self.battery.reset()

        # (Re)allocate the monitoring buffer for this rollout.
        # MATLAB equivalent: MPC/simu.m:15  obj.MicroGrid.Monitoring = nan(nPoints, 6).
        start_ts = pd.Timestamp(self.pv.timestamps[0])
        self.monitoring_table = MonitoringTable(
            n_steps=self.max_steps,
            start_time=start_ts,
            delta_t_min=self.delta_t_min,
        )
        return self._get_obs(), {}

    def _store_value(self, idx: int) -> float:
        """Valeur prix-aware de l'énergie stockée à l'instant ``idx`` (€/kWh).

        Meilleure opportunité de déploiement futur sur l'horizon : éviter un import (prix import
        futur) OU exporter (prix export futur) → max sur les deux forecasts. En matinée, capte
        déjà le pic du soir (export CoutsProd) ⇒ incite à porter l'énergie jusqu'au soir.
        Repli sur le prix instantané si le signal n'a pas de forecast.
        """
        if not self.price_signal.has_forecast:
            return max(
                self.price_signal.get_import_price(idx),
                self.price_signal.get_export_price(idx),
            )
        imp_fc = self.price_signal.get_import_forecast(idx, self.horizon_steps)
        exp_fc = self.price_signal.get_export_forecast(idx, self.horizon_steps)
        return max(float(imp_fc.max()), float(exp_fc.max()))

    def _export_opportunity_cost(self, idx: int) -> float:
        """Coût d'opportunité (€/kWh) d'exporter le stock à l'instant ``idx`` au lieu de le
        garder pour éviter un import futur :  ``oc = max(0, v_imp_future − price_exp_now)``.

        ``v_imp_future`` = meilleur import évitable sur l'horizon (max du forecast import,
        comme ``_store_value`` ; repli sur le prix import instantané sans forecast). On compare
        l'IMPORT futur (pas l'export futur) : l'objet du terme est « ne pas vendre ce qui
        servirait à éviter un import plus cher ».

        PAS de facteur rendement : l'énergie est DÉJÀ stockée, ``eta_d`` s'applique
        identiquement à « exporter maintenant » et « éviter un import plus tard » ⇒ se
        simplifie. Mettre ``eta_d`` sur-pénaliserait et casserait l'invariance.

        ZÉRO sur l'optimum : un export de stock optimal n'a lieu que si ``price_exp ≥`` meilleur
        import futur évitable ⇒ ``oc = 0`` ⇒ le terme ne déplace pas l'optimum (cf. r_spread).
        """
        price_exp_now = self.price_signal.get_export_price(idx)
        if not self.price_signal.has_forecast:
            v_imp = self.price_signal.get_import_price(idx)
        else:
            v_imp = float(self.price_signal.get_import_forecast(idx, self.horizon_steps).max())
        return max(0.0, v_imp - price_exp_now)

    def step(self, action):
        # Lues AVANT le calcul de Pb_command : la contrainte EDF ci-dessous borne la charge
        # au surplus PV instantané (pv - load).
        pv_t = self.pv.get_irradiance(self.step_index)
        load_t = self.load.get_load(self.step_index)

        action_val = float(np.clip(action[0], -1.0, 1.0))

        if action_val < 0:
            Pb_command = action_val * self.max_charge_kw
        else:
            Pb_command = action_val * self.max_discharge_kw

        # Contrainte EDF : interdiction de charger la batterie depuis le réseau. La batterie
        # ne peut absorber que le surplus PV (pv - load). Pb_command < 0 = charge ; on borne
        # sa magnitude au surplus (si pv ≤ load, surplus = 0 ⇒ charge interdite). Identique à
        # la borne MILP (Pb_charge ≤ max(0, pv - load)) pour que RL et MILP modélisent la même
        # physique. Ainsi la charge ne provoque jamais d'import réseau.
        if Pb_command < 0.0:
            Pb_command = max(Pb_command, -max(0.0, pv_t - load_t))

        soc_old = self.battery.soc  # avant battery.step, pour le potentiel PBRS Φ(s_t)
        Pb_effective, new_soc = self.battery.step(Pb_command, self.delta_t_h)

        # Bilan brut (non borné). P_grid_raw < 0 ⇒ surplus à exporter.
        P_grid_raw = load_t - pv_t - Pb_effective

        # Curtailment PV côté export : l'onduleur écrête le PV quand le surplus
        # dépasse la limite d'export et que la batterie ne peut plus absorber.
        Pcurt = max(0.0, -P_grid_raw - self.max_export_kw)

        P_grid = float(np.clip(P_grid_raw, -self.max_export_kw, self.max_import_kw))

        # Puissance fantôme = charge non servie au-delà de la limite d'import (ce que le
        # clip ci-dessus écartait silencieusement). Pénalisée comme dans le MILP.
        P_phantom = max(0.0, P_grid_raw - self.max_import_kw)
        r_phantom = -self.phantom_penalty * P_phantom * self.delta_t_h

        price_imp = self.price_signal.get_import_price(self.step_index)
        price_exp = self.price_signal.get_export_price(self.step_index)
        r_eco = -(
            price_imp * max(P_grid, 0.0)
            - price_exp * max(-P_grid, 0.0)
        ) * self.delta_t_h

        # SoC penality is dead code when soc_safe_min = soc_min and soc_safe_max = soc_max (clip in battery.step) 
        r_soc = -self.sigma_soc * (
            max(0.0, self.soc_safe_min - new_soc)
            + max(0.0, new_soc - self.soc_safe_max)
        )

        
        if self.curtailment_mode == "penal":
            r_curt = -price_exp * Pcurt * self.delta_t_h
        else:  # "clip"
            r_curt = 0.0

        # r_bat_power = -self._sigma_bat * (Pb_effective / self._pb_max) ** 2

        # Ce que ça détecte : la batterie est en train de décharger (Pb_effective > 0) alors qu'il y a un surplus PV (pv_t >
        # load_t). C'est une décision sous-optimale : la batterie gaspille de l'énergie stockée au lieu de laisser le PV couvrir
        # la charge directement.
        r_spread = 0.0
        if self._spread_penalty:
            pv_surplus = max(0.0, pv_t - load_t)
            if Pb_effective > 0.0 and pv_surplus > 0.0:
                r_spread = -(price_imp - price_exp) * min(Pb_effective, pv_surplus) * self.delta_t_h

        # PBRS — valeur prix-aware du stock (Ng 1999) : F = γ·Φ(s') − Φ(s), Φ = σ·v(t)·E_util,
        # avec E_util = (SoC − soc_min)·capacité. Annule le biais d'actualisation qui pousse à
        # vider la batterie tôt, SANS déplacer l'optimum (la somme actualisée télescope en
        # −Φ(s₀) + γ^T·Φ(s_T) ≈ −Φ(s₀)). On NE met PAS Φ(s_T)=0 (évite un artefact de dump de fin) :
        # à t+1 = max_steps, la fenêtre forecast reste dans les données (max_steps = n_steps − H).
        r_store = 0.0
        if self._store_shaping:
            cap = self.battery.capacity_kwh
            floor = self.battery.soc_min
            phi_t = self._sigma_store * self._store_value(self.step_index) * (soc_old - floor) * cap
            phi_tp1 = self._sigma_store * self._store_value(self.step_index + 1) * (new_soc - floor) * cap
            r_store = self._gamma_shaping * phi_tp1 - phi_t

        # r_export_stock — coût d'opportunité one-sided sur l'export DE STOCK. e_batt_exp =
        # export au-delà du surplus PV instantané ⇒ provient forcément de la décharge batterie.
        # Décourage de vider la batterie à l'export quand garder le stock éviterait un import
        # futur plus cher (cas ete_basse : export soir 0.199 < import HP 0.2475). ZÉRO sur
        # l'optimum ⇒ ne déplace pas l'optimum (cf. r_spread). Pas de garde curtailment :
        # décharger AUGMENTE l'export, ne libère jamais de place PV (cf. _export_opportunity_cost).
        r_export_stock = 0.0
        if self._sigma_export_stock > 0.0:
            pv_surplus = max(0.0, pv_t - load_t)
            e_batt_exp_kw = max(0.0, max(-P_grid, 0.0) - pv_surplus)
            if e_batt_exp_kw > 0.0:
                oc = self._export_opportunity_cost(self.step_index)
                r_export_stock = -self._sigma_export_stock * oc * e_batt_exp_kw * self.delta_t_h

        # reward = r_eco + r_soc + r_curt + r_bat_power + r_spread
        reward = r_eco + r_soc + r_curt +  r_spread + r_phantom + r_store + r_export_stock

        # reward = r_eco + r_soc + r_curt

        # Record this decision step in the monitoring buffer BEFORE incrementing
        # step_index, so row k contains the action+state at decision step k.
        # MATLAB equivalent: PMS/follow.m:3  obj.MicroGrid.insert_monitoring_data(state).
        # SoC is stored in percent (0-100); battery.soc is a fraction in [0,1].
        if self.monitoring_table is not None and self.step_index < self.max_steps:
            self.monitoring_table.insert(
                self.step_index,
                {
                    "pp": float(pv_t),
                    "pl": float(load_t),
                    "soc": float(new_soc) * 100.0,
                    "pb": float(Pb_effective),
                    "pg": float(P_grid),
                    "action_raw": float(action_val),
                    "pb_command": float(Pb_command),
                    "p_imp":     float(max(P_grid, 0.0)),
                    "p_exp":     float(max(-P_grid, 0.0)),
                    "price_imp": float(price_imp),
                    "price_exp": float(price_exp),
                    "r_eco":     float(r_eco),
                    "r_soc":     float(r_soc),
                    # "r_bat_power": float(r_bat_power),
                    "r_spread":  float(r_spread),
                    "r_phantom": float(r_phantom),
                    "r_store":   float(r_store),
                    "r_export_stock": float(r_export_stock),
                    "reward":    float(reward),
                    "pcurt":     float(Pcurt),
                    "pph":       float(P_phantom),
                },
            )

        self.step_index += 1
        # Le cutoff d'horizon est une TRONCATURE temporelle (l'opération microgrid continue),
        # pas un terminal naturel → truncated=True (terminated=False). SB3 (SAC,
        # handle_timeout_termination=True) bootstrappe alors V(s') à la frontière : sans ça,
        # done_effective=True ⇒ aucune valeur de continuation ⇒ myopie de fin d'épisode
        # (l'agent cesse d'investir dans le stock les derniers jours) ET le γ·Φ(s_T) du PBRS
        # r_store ne s'annule pas (biais terminal). Cf. plans_claude (overfit re-frame).
        reached_horizon = self.step_index >= self.max_steps
        terminated = False
        truncated = reached_horizon

        info = {
            "Pb_effective": Pb_effective,
            "P_grid": P_grid,
            "soc": new_soc,
            "r_eco": r_eco,
            "r_soc": r_soc,
            # "r_bat_power": r_bat_power,
            "r_spread": r_spread,
            "r_phantom": r_phantom,
            "r_store": r_store,
            "r_export_stock": r_export_stock,
            "P_phantom": P_phantom,
            "Pcurt": Pcurt,
            "pv_t": pv_t,
            "load_t": load_t,
            "price_import": price_imp,
            "price_export": price_exp,
        }

        # Vraie obs finale (pas des zéros) pour que SB3 bootstrappe V(s') dessus à la
        # troncature (DummyVecEnv la stocke comme terminal_observation). Garde-fou : sur le
        # chemin replay où max_steps est poussé à n_steps, step_index peut atteindre n_steps
        # → zéros (non lus : la boucle de replay break sur `done` avant de consommer l'obs).
        if self.step_index < self.pv.n_steps:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info
