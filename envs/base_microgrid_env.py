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

        # Contrainte EDF no-grid-charging — deux formulations (cf. temp_md/milpIncoherences.md #5),
        # DOIT être identique à celle du MILP (baselines/milp_solver.py) pour que le gap RL↔MILP
        # reste mesuré sur la même physique. "total" (défaut, relâché Duchaud-JL split-node) :
        # charge ≤ pv (le réseau sert la charge en //) ; "surplus" (strict, historique) :
        # charge ≤ max(0, pv−load). Les configs antérieures au flux relâché épinglent surplus.
        self._pv_charge_mode = config["battery"].get("pv_charge_mode", "total")
        if self._pv_charge_mode not in ("surplus", "total"):
            raise ValueError(
                f"Unknown battery.pv_charge_mode={self._pv_charge_mode!r}; use 'surplus' or 'total'."
            )

        self._load_forecast_in_obs = config.get("observation", {}).get("load_forecast", False)
        self._price_export_fc_in_obs = config.get("observation", {}).get("price_export_forecast", False)
        # Feature de timing (optimum-safe, pur input — ne touche ni reward ni dynamique) : « gap à
        # pic » = de combien l'énergie vaut PLUS au meilleur moment futur de l'horizon que maintenant
        # (cf. _get_obs). Donne explicitement le signal « dois-je attendre ? » que l'agent devait
        # sinon extraire lui-même du vecteur forecast (et ratait → dump t0). Exige un forecast prix
        # (has_forecast) ; inerte sous prix fixes (=0). Défaut OFF ⇒ obs des configs existantes inchangées.
        self._timing_feat = config.get("observation", {}).get("timing_feature", False)
        # v2 (exp25) : 4 scalaires au lieu du gap agrégé — le scalaire v1 dit « attendre » mais
        # ni COMBIEN DE TEMPS ni OÙ (il ne distingue pas la bosse export matinale ~9h du pic du
        # soir ~21h de la grille CoutsProd, cf. audit exp23 : décharge du soir démarrée ~1h trop
        # tôt, heure 21-22h à 0.442 ratée). Mêmes garanties que v1 : pur input (optimum-safe),
        # exige has_forecast, défaut OFF. Cumulable avec timing_feature (dims indépendantes).
        self._timing_feats_v2 = config.get("observation", {}).get("timing_features_v2", False)
        # v3 (exp29) : 2 scalaires CHARGE-side — v2 est orienté export-max et ne dit rien du
        # meilleur moment pour CHARGER (= pas de l'horizon au prix d'export le plus BAS parmi
        # ceux où la charge est faisable, i.e. surplus PV > 0). Cf. audit exp28 : biais « charger
        # trop tôt » = 75 % du gap résiduel hiver (charge à 11h/0.201 au lieu de 12-15h/0.181-0.191).
        # Mêmes garanties que v1/v2 : pur input (optimum-safe), exige has_forecast, défaut OFF.
        self._timing_feats_v3 = config.get("observation", {}).get("timing_features_v3", False)
        self._spread_penalty = config["reward"].get("spread_penalty", False)
        self._sigma_bat = config["reward"].get("sigma_bat", 0.0)
        self._random_soc = config.get("training", {}).get("random_soc", False)
        self._pb_max = max(self.max_charge_kw, self.max_discharge_kw)

        # PBRS — valeur prix-aware du stock (Ng 1999). σ_store = force du shaping ; γ doit être
        # CELUI de SAC (cf. sac_agent.py) pour que F = γ·Φ(s')−Φ(s) préserve exactement l'optimum.
        self._store_shaping = config["reward"].get("store_value", False)
        self._sigma_store = config["reward"].get("sigma_store", 1.0)
        self._gamma_shaping = config.get("training", {}).get("gamma", 0.99)
        # Source de la valeur v(t) du potentiel Φ (exp26) : "horizon_max" (défaut, heuristique
        # max forecast — survalorise le stock : ignore la limite de puissance et ne décroît pas
        # après le pic) | "milp_dual" (λ(t) exact du LP à binaires figés, cf.
        # baselines/milp_solver.milp_soc_duals — calculé paresseusement au premier reset()).
        # N'affecte QUE Φ (r_store) : les features d'obs (timing_*) et le crédit SoC de bord
        # (_rescore → _store_value) restent sur le forecast, pour que le scoring reste
        # bit-comparable avec exp21b/exp23/exp25. PBRS invariant quel que soit Φ ⇒ l'optimum
        # et la comparaison RL↔MILP ne bougent pas.
        self._store_value_mode = config["reward"].get("store_value_mode", "horizon_max")
        if self._store_value_mode not in ("horizon_max", "milp_dual"):
            raise ValueError(
                f"Unknown reward.store_value_mode={self._store_value_mode!r}; "
                "use 'horizon_max' or 'milp_dual'."
            )
        self._store_value_series: np.ndarray | None = None  # λ(t) €/kWh, longueur max_steps+1

        # Pénalité one-sided sur l'export DE STOCK (coût d'opportunité prix-aware). 0.0 = inactif
        # (opt-in, runs existants bit-identiques). Zéro sur l'optimum ⇒ ne déplace pas l'optimum
        # (même philosophie que spread_penalty). Décourage de vendre le stock quand le garder
        # éviterait un import futur plus cher (cf. _export_opportunity_cost).
        self._sigma_export_stock = config["reward"].get("sigma_export_stock", 0.0)

        # Miroir EXPORT-side de sigma_export_stock : pénalise l'export DE STOCK maintenant quand un
        # prix d'export PLUS ÉLEVÉ arrive dans l'horizon (coût d'opportunité de « vendre trop tôt »,
        # cf. _export_hold_cost). 0.0 = inactif (opt-in, runs existants bit-identiques). Cible le
        # biais « déployer trop tôt » (exp23/exp27 : le RL vide la batterie AVANT le pic export du
        # soir ~21h à 0.442 → rate l'arbitrage export>import que fait le MILP). λ(t) du milp_dual
        # étant quasi-plat (~0.24 « refill », il ne distingue pas le pic) le PBRS seul ne suffit
        # pas ⇒ ce terme injecte le gradient temporel manquant. ~Zéro sur l'optimum (export optimal
        # au pic ⇒ oc_hold≈0), petit résidu sur la rampe rate-limitée ⇒ prior doux (swept, arm 0).
        self._sigma_hold_export = config["reward"].get("sigma_hold_export", 0.0)

        # Miroir CHARGE-side de sigma_hold_export (exp29) : pénalise la CHARGE maintenant quand un
        # prix d'export plus BAS arrive dans l'horizon sur un pas où la charge est FAISABLE
        # (surplus PV > 0 — la batterie ne charge que du PV, cf. contrainte EDF). Charger coûte
        # l'export sacrifié : oc_charge = max(0, p_exp_now − min faisable futur), cf.
        # _charge_hold_cost. Cible le biais « charger trop tôt » (audit exp28 run_008 : 75 % du
        # gap résiduel hiver = charge démarrée à 11h/0.201 et étalée à puissance partielle, au
        # lieu de concentrée à −100 kW dans la fenêtre 0.181-0.191 comme le MILP). ~Zéro sur
        # l'optimum (le MILP charge aux pas d'export min faisables), petit résidu quand la
        # fenêtre min n'a pas assez de surplus (hiver : ~30 kWh chargés à 0.201) ⇒ prior DOUX
        # comme sigma_hold_export (swept, arm 0). 0.0 = inactif (runs existants bit-identiques).
        self._sigma_charge_hold = config["reward"].get("sigma_charge_hold", 0.0)

        price_forecast_dim = self.horizon_steps if self.price_signal.has_forecast else 0
        load_forecast_dim = self.horizon_steps if self._load_forecast_in_obs else 0
        export_forecast_dim = (
            self.horizon_steps
            if (self._price_export_fc_in_obs and self.price_signal.has_forecast)
            else 0
        )
        timing_dim = 1 if (self._timing_feat and self.price_signal.has_forecast) else 0
        timing_v2_dim = 4 if (self._timing_feats_v2 and self.price_signal.has_forecast) else 0
        timing_v3_dim = 2 if (self._timing_feats_v3 and self.price_signal.has_forecast) else 0
        obs_dim = (
            9 + self.horizon_steps + load_forecast_dim
            + price_forecast_dim + export_forecast_dim + timing_dim + timing_v2_dim
            + timing_v3_dim
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
        if self._timing_feats_v2 and self.price_signal.has_forecast:
            # 4 signaux de timing découplés (import/export séparés + QUAND) :
            #   1. steps_to_argmax_export / horizon ∈ [0,1[ — délai jusqu'au meilleur moment
            #      d'export (0 = c'est maintenant → « vends »), le v1 ne donnait pas le QUAND ;
            #   2. max export futur − p_exp maintenant ≥ 0 — gain d'attente côté export ;
            #   3. max import futur − p_imp maintenant ≥ 0 — gain d'attente côté import évité ;
            #   4. p_exp maintenant − moyenne export future — « bosse locale » : >0 si l'instant
            #      courant est localement cher (bosse matinale 9h : vendre le PV direct),
            #      <0 dans les creux (charger). Distingue les 2 bosses de la grille CoutsProd.
            imp_fc = self.price_signal.get_import_forecast(self.step_index, self.horizon_steps)
            exp_fc = self.price_signal.get_export_forecast(self.step_index, self.horizon_steps)
            parts.append(np.array(
                [
                    float(np.argmax(exp_fc)) / self.horizon_steps,
                    float(exp_fc.max()) - p_exp,
                    float(imp_fc.max()) - p_imp,
                    p_exp - float(exp_fc.mean()),
                ],
                dtype=np.float32,
            ))
        if self._timing_feats_v3 and self.price_signal.has_forecast:
            # 2 signaux de timing CHARGE-side (v2 est orienté export-max, cf. __init__) :
            #   1. steps_to_argmin_export_faisable / horizon ∈ [0,1[ — délai jusqu'au moment le
            #      moins cher pour charger (0 = c'est maintenant → « charge ») ;
            #   2. p_exp maintenant − min faisable — gain d'attente côté charge : ≥ 0 quand la
            #      charge est faisable maintenant (grand → attendre la fenêtre pas chère), peut
            #      être < 0 quand elle ne l'est pas (export courant sous le min faisable — inerte,
            #      la charge est impossible). (0, 0) si aucun pas faisable ; inerte prix fixes.
            found = self._feasible_charge_export_min(self.step_index)
            if found is None:
                parts.append(np.zeros(2, dtype=np.float32))
            else:
                parts.append(np.array(
                    [
                        float(found[0]) / self.horizon_steps,
                        p_exp - found[1],
                    ],
                    dtype=np.float32,
                ))

        return np.concatenate(parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # λ(t) du mode milp_dual : calcul paresseux au premier reset (PAS dans __init__ — le
        # solve MIP+LP prend quelques secondes et tous les chemins de construction d'env
        # (make_env, make_env_overfit, replay) passent par reset). Import local : baselines
        # n'importe pas envs ⇒ pas de cycle.
        if (self._store_shaping and self._store_value_mode == "milp_dual"
                and self._store_value_series is None):
            from baselines.milp_solver import milp_soc_duals
            lam = np.asarray(milp_soc_duals(self, self.cfg), dtype=np.float64)
            assert lam.shape[0] >= self.max_steps + 1, (
                f"milp_soc_duals: {lam.shape[0]} valeurs < max_steps+1 = {self.max_steps + 1}"
            )
            self._store_value_series = lam
            print(f"[store_value_mode=milp_dual] λ(t) calculé : "
                  f"min={lam.min():.4f} max={lam.max():.4f} €/kWh ({lam.shape[0]} pas)")
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

    def _phi_value(self, idx: int) -> float:
        """Valeur v(t) utilisée par le potentiel PBRS Φ = σ·v(t)·E_util (et lui SEUL).

        milp_dual : λ(idx) exact (€/kWh) précalculé au reset ; sinon repli sur l'heuristique
        ``_store_value`` (max forecast). Les obs et le crédit de bord restent sur
        ``_store_value`` — cf. commentaire de ``_store_value_mode`` dans __init__.
        """
        if self._store_value_series is not None:
            return float(self._store_value_series[idx])
        return self._store_value(idx)

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

    def _export_hold_cost(self, idx: int) -> float:
        """Coût d'opportunité (€/kWh) d'exporter le stock MAINTENANT au lieu d'attendre un meilleur
        prix d'export dans l'horizon :  ``oc = max(0, max(export_forecast) − price_exp_now)``.

        Miroir EXPORT-side de ``_export_opportunity_cost`` (qui, lui, compare à l'IMPORT futur
        évitable). Ici on compare l'EXPORT futur : « ne vends pas ton stock maintenant si le pic
        export du soir paie plus ». Numériquement identique à la feature timing_v2 #2, mais
        réutilisée ici comme coût d'opportunité pénalisant (et non comme simple input d'obs).

        PAS de facteur rendement : ``eta_d`` s'applique identiquement à « exporter maintenant » et
        « exporter plus tard » ⇒ se simplifie (même raison que ``_export_opportunity_cost``).

        ~ZÉRO sur l'optimum : l'export optimal a lieu AU pic (``max_future ≤ price_exp_now`` ⇒
        ``oc = 0``) ; petit résidu positif seulement sur la rampe rate-limitée juste avant le pic.
        C'est donc un prior DOUX, pas un invariant strict (contrairement à
        ``_export_opportunity_cost``) ⇒ à balayer (arm σ_hold=0) et juger sur le net_cost réel.
        Repli 0 sans forecast (pas de pic futur connu ⇒ aucun coût d'opportunité).
        """
        if not self.price_signal.has_forecast:
            return 0.0
        price_exp_now = self.price_signal.get_export_price(idx)
        v_exp = float(self.price_signal.get_export_forecast(idx, self.horizon_steps).max())
        return max(0.0, v_exp - price_exp_now)

    def _feasible_charge_export_min(self, idx: int) -> tuple[int, float] | None:
        """(argmin, min) du prix d'export sur les pas de l'horizon où la CHARGE est faisable.

        Charger coûte l'export sacrifié au prix du pas ⇒ le meilleur moment pour charger est le
        pas au prix d'export MINIMAL parmi ceux où la batterie peut effectivement charger :
        surplus PV > 0 en mode ``surplus`` (pv_fc − load_fc), PV > 0 en mode ``total``. Sans ce
        masque, le min serait la nuit (export plancher) où la charge est physiquement impossible.
        Retourne None si aucun pas faisable dans l'horizon ou sans forecast prix. L'élément 0 des
        forecasts = maintenant : si la charge est faisable maintenant, min ≤ p_exp_now garanti.
        """
        if not self.price_signal.has_forecast:
            return None
        pv_fc = self.pv.get_forecast(idx, self.horizon_steps)
        if self._pv_charge_mode == "total":
            feasible = pv_fc > 0.0
        else:
            feasible = (pv_fc - self.load.get_forecast(idx, self.horizon_steps)) > 0.0
        if not feasible.any():
            return None
        exp_fc = self.price_signal.get_export_forecast(idx, self.horizon_steps)
        masked = np.where(feasible, exp_fc, np.inf)
        argmin = int(np.argmin(masked))
        return argmin, float(masked[argmin])

    def _charge_hold_cost(self, idx: int) -> float:
        """Coût d'opportunité (€/kWh) de charger MAINTENANT au lieu d'attendre un pas faisable au
        prix d'export plus bas :  ``oc = max(0, price_exp_now − min faisable(export_forecast))``.

        Miroir CHARGE-side de ``_export_hold_cost`` : là-bas « ne vends pas ton stock avant le pic
        export », ici « ne sacrifie pas de l'export cher pour charger quand une fenêtre d'export
        moins chère (donc une charge moins coûteuse) arrive plus tard ». Le min est restreint aux
        pas où la charge est FAISABLE (cf. ``_feasible_charge_export_min``).

        PAS de facteur rendement : ``eta_c`` s'applique identiquement à « charger maintenant » et
        « charger plus tard » ⇒ se simplifie (même raison que les deux autres termes miroirs).

        ~ZÉRO sur l'optimum : le MILP charge aux pas d'export min faisables (``min ≥ p_exp_now``
        sur ces pas ⇒ ``oc = 0``) ; résidu positif quand la fenêtre min n'a pas assez de surplus
        PV pour tout charger (le MILP déborde sur des pas plus chers) ou via le chevauchement 24h
        avec la fenêtre bon marché du LENDEMAIN. Prior DOUX, pas invariant strict ⇒ à balayer
        (arm σ_charge=0) et juger sur le net_cost réel. Repli 0 sans forecast / sans pas faisable
        (charge impossible de toute façon ⇒ aucun coût d'opportunité).
        """
        found = self._feasible_charge_export_min(idx)
        if found is None:
            return 0.0
        price_exp_now = self.price_signal.get_export_price(idx)
        return max(0.0, price_exp_now - found[1])

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

        # Contrainte EDF : interdiction de charger la batterie DEPUIS LE RÉSEAU. Pb_command < 0 =
        # charge ; on borne sa magnitude selon pv_charge_mode (identique à la borne MILP pour que
        # RL et MILP modélisent la même physique) :
        #   - "surplus" (strict) : charge ≤ max(0, pv − load) — si pv ≤ load, charge interdite.
        #   - "total" (Duchaud-JL) : charge ≤ pv — la batterie absorbe tout le PV pendant que le
        #     réseau sert la charge (P_grid_raw = load ci-dessous ⇒ import = charge, jamais →batt).
        # Dans les deux cas la charge ≤ PV ⇒ ne provoque jamais d'import réseau vers la batterie.
        if Pb_command < 0.0:
            charge_cap = pv_t if self._pv_charge_mode == "total" else max(0.0, pv_t - load_t)
            Pb_command = max(Pb_command, -charge_cap)

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
            phi_t = self._sigma_store * self._phi_value(self.step_index) * (soc_old - floor) * cap
            phi_tp1 = self._sigma_store * self._phi_value(self.step_index + 1) * (new_soc - floor) * cap
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

        # r_hold_export — miroir EXPORT-side de r_export_stock : pénalise l'export DE STOCK
        # maintenant quand un pic export PLUS haut arrive dans l'horizon (oc_hold = max(0,
        # max(export_fc) − price_exp_now), cf. _export_hold_cost). Cible le biais « déployer trop
        # tôt » (le RL vide avant le pic 0.442 du soir → rate l'arbitrage export>import du MILP,
        # que le λ(t) quasi-plat du milp_dual ne pénalise pas). Même e_batt_exp que r_export_stock
        # (export issu de la décharge batterie, pas du surplus PV). ~Zéro sur l'optimum (prior doux).
        r_hold_export = 0.0
        if self._sigma_hold_export > 0.0:
            pv_surplus = max(0.0, pv_t - load_t)
            e_batt_exp_kw = max(0.0, max(-P_grid, 0.0) - pv_surplus)
            if e_batt_exp_kw > 0.0:
                oc_hold = self._export_hold_cost(self.step_index)
                r_hold_export = -self._sigma_hold_export * oc_hold * e_batt_exp_kw * self.delta_t_h

        # r_charge_hold — miroir CHARGE-side de r_hold_export : pénalise la CHARGE maintenant
        # quand une fenêtre d'export plus basse (= charge moins coûteuse) arrive dans l'horizon
        # sur un pas faisable (oc_charge = max(0, price_exp_now − min faisable(export_fc)), cf.
        # _charge_hold_cost). Cible le biais « charger trop tôt » (audit exp28 : le RL charge dès
        # 11h/0.201 à puissance partielle et rate la fenêtre 0.181-0.191 concentrée du MILP, puis
        # exporte son surplus au pire prix). e_charge = charge effective (déjà ≤ surplus PV via la
        # contrainte EDF ci-dessus). ~Zéro sur l'optimum (prior doux, cf. _charge_hold_cost).
        r_charge_hold = 0.0
        if self._sigma_charge_hold > 0.0:
            e_charge_kw = max(0.0, -Pb_effective)
            if e_charge_kw > 0.0:
                oc_charge = self._charge_hold_cost(self.step_index)
                r_charge_hold = -self._sigma_charge_hold * oc_charge * e_charge_kw * self.delta_t_h

        # reward = r_eco + r_soc + r_curt + r_bat_power + r_spread
        reward = (r_eco + r_soc + r_curt + r_spread + r_phantom + r_store
                  + r_export_stock + r_hold_export + r_charge_hold)

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
                    "r_hold_export": float(r_hold_export),
                    "r_charge_hold": float(r_charge_hold),
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
            "r_hold_export": r_hold_export,
            "r_charge_hold": r_charge_hold,
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
