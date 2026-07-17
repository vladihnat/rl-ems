#!/usr/bin/env python3
"""Generate a parameter sweep: N yaml configs + a SLURM manifest.

Usage
-----
python cluster/sweep/generate_sweep.py \\
    --base-config configs/exp03_1M_300.yaml \\
    --sweep-name gamma_lr_sweep \\
    --params gamma=0.99,0.999 lr=0.0003,0.001 \\
    --seeds 42,43,44

Outputs
-------
- configs/sweeps/<sweep_name>/run_<i>.yaml  (one per combination)
- cluster/sweep/manifests/<sweep_name>.json  (indexed by SLURM_ARRAY_TASK_ID)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import yaml

# Mapping: CLI param name → dot-path in YAML
PARAM_MAP = {
    "gamma":          "training.gamma",
    "lr":             "training.learning_rate",
    "batch_size":     "training.batch_size",
    "buffer_size":    "training.buffer_size",
    "sigma_soc":      "reward.sigma_soc",
    "seed":           "experiment.seed",
    # Contrainte EDF no-grid-charging : "surplus" (strict) | "total" (relâché Duchaud-JL, exp24+) :
    "pv_charge_mode": "battery.pv_charge_mode",
    # Leviers ajoutés pour le sweep RL/MILP (exp11) :
    "norm_obs":       "training.norm_obs",
    "norm_reward":    "training.norm_reward",     # VecNormalize reward (dompte la variance du critic)
    "net_arch":       "training.net_arch",       # valeur liste, ex. 256x256 → [256, 256]
    "ent_coef":       "training.ent_coef",
    "tau":            "training.tau",
    "train_freq":     "training.train_freq",
    "gradient_steps": "training.gradient_steps",
    "total_timesteps":"training.total_timesteps",
    # Leviers comportementaux (exp14+) :
    "learning_starts":"training.learning_starts",
    "random_soc":     "training.random_soc",
    "warm_start_milp":"training.warm_start_milp",
    # Behavior cloning depuis le MILP (exp17+) :
    "bc_pretrain_epochs": "training.bc_pretrain_epochs",
    "bc_pretrain_lr":     "training.bc_pretrain_lr",
    "milp_window_days":   "training.milp_window_days",
    # Sanity check « overfit » : entraîner ET évaluer sur la même fenêtre (exp21+) :
    "eval_on_train":  "training.eval_on_train",
    "score_days":     "training.score_days",
    # Feature d'obs de timing « gap à pic » (optimum-safe, exp23+) :
    "timing_feature": "observation.timing_feature",
    # Features de timing v2 — 4 scalaires découplés (QUAND + gaps imp/exp + bosse locale, exp25+) :
    "timing_features_v2": "observation.timing_features_v2",
    # Axe composite : régler v1/v2 comme UNE variable de sweep (none | v1 | v2 | both).
    # Traité spécialement dans main() — pose observation.timing_feature ET timing_features_v2.
    "timing_mode": "__composite__",
    # PBRS : source de la valeur du stock v(t) — "horizon_max" (défaut) | "milp_dual" (exp26+) :
    "store_value_mode": "reward.store_value_mode",
    # PBRS — valeur prix-aware du stock / reward shaping (exp19+) :
    "store_value":    "reward.store_value",
    "sigma_store":    "reward.sigma_store",
    # Coût d'opportunité one-sided sur l'export de stock (exp19b+) :
    "sigma_export_stock": "reward.sigma_export_stock",
    # Miroir EXPORT-side (exp28+) : pénalise l'export de stock avant le pic export de l'horizon
    # (anti « déployer trop tôt », cf. base_microgrid_env._export_hold_cost) :
    "sigma_hold_export": "reward.sigma_hold_export",
    # Miroir CHARGE-side (exp29+) : pénalise la charge avant la fenêtre d'export min faisable de
    # l'horizon (anti « charger trop tôt », cf. base_microgrid_env._charge_hold_cost) :
    "sigma_charge_hold": "reward.sigma_charge_hold",
    # Features de timing v3 — 2 scalaires CHARGE-side (QUAND charger au moins cher, exp29+) :
    "timing_features_v3": "observation.timing_features_v3",
    # WS-1c — ancre MILP persistante / anti-washout (exp18+) :
    "bc_anchor_demo_buffer":  "training.bc_anchor_demo_buffer",
    "bc_anchor_demo_frac":    "training.bc_anchor_demo_frac",
    "bc_anchor_loss":         "training.bc_anchor_loss",
    "bc_anchor_lambda":       "training.bc_anchor_lambda",
    "bc_anchor_lambda_final": "training.bc_anchor_lambda_final",
    "bc_anchor_loss_batch":   "training.bc_anchor_loss_batch",
    # AWAC — terme BC pondéré par l'avantage critique (exp20+) :
    "bc_anchor_awac":         "training.bc_anchor_awac",
    "bc_anchor_awac_beta":    "training.bc_anchor_awac_beta",
    "bc_anchor_awac_wmax":    "training.bc_anchor_awac_wmax",
}

ROOT = Path(__file__).resolve().parents[2]


def _set_nested(d: dict, dot_path: str, value):
    keys = dot_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(s: str):
    """Cast a CLI token: bool → net_arch list ('256x256') → int → float → str.

    'net_arch' values are encoded with 'x' as separator so a single --params
    token stays comma-free, e.g. net_arch=256x256,400x400x300.
    """
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    parts = low.split("x")
    if len(parts) > 1 and all(p.isdigit() for p in parts):
        return [int(p) for p in parts]
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_params(raw: list[str]) -> dict[str, list]:
    """Parse ['gamma=0.99,0.999', 'lr=0.0003,0.001'] → {'gamma': [0.99, 0.999], ...}"""
    result = {}
    for item in raw:
        if "=" not in item:
            sys.exit(f"[generate_sweep] bad --params entry '{item}', expected key=v1,v2,...")
        key, vals = item.split("=", 1)
        if key not in PARAM_MAP:
            known = ", ".join(PARAM_MAP)
            sys.exit(f"[generate_sweep] unknown param '{key}'. Known: {known}")
        result[key] = [_parse_value(v) for v in vals.split(",")]
    return result


def main():
    ap = argparse.ArgumentParser(description="Generate sweep configs + SLURM manifest.")
    ap.add_argument("--base-config", required=True, help="Base YAML config path")
    ap.add_argument("--sweep-name", required=True, help="Sweep identifier (no spaces)")
    ap.add_argument("--params", nargs="+", default=[],
                    help="Params to sweep, e.g. gamma=0.99,0.999 lr=0.0003,0.001")
    ap.add_argument("--seeds", default="42",
                    help="Comma-separated seeds (default: 42)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print combinations without writing files")
    args = ap.parse_args()

    base_path = ROOT / args.base_config
    if not base_path.exists():
        sys.exit(f"[generate_sweep] base config not found: {base_path}")

    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    params = parse_params(args.params)
    seeds = [_parse_value(s) for s in args.seeds.split(",")]

    # Build cartesian product: seeds treated as another axis
    axes = list(params.items())
    if len(axes) == 0 and len(seeds) == 1:
        sys.exit("[generate_sweep] nothing to sweep: provide --params or multiple --seeds")

    # Combine params axes
    if axes:
        keys, value_lists = zip(*axes)
        combos = list(itertools.product(*value_lists))
    else:
        keys, combos = [], [()]

    runs = []
    for seed in seeds:
        for combo in combos:
            override = dict(zip(keys, combo))
            override["seed"] = seed
            runs.append(override)

    print(f"[generate_sweep] {len(runs)} runs  (params={list(keys)}, seeds={seeds})")

    if args.dry_run:
        for i, r in enumerate(runs):
            print(f"  run_{i:03d}: {r}")
        return

    # Write configs
    sweep_cfg_dir = ROOT / "configs" / "sweeps" / args.sweep_name
    sweep_cfg_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = ROOT / "cluster" / "sweep" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, override in enumerate(runs):
        cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy

        for param_key, val in override.items():
            if param_key == "timing_mode":
                # Axe composite exclusif : une seule valeur pilote les deux flags booléens
                # (le produit cartésien brut timing_feature × timing_features_v2 générerait
                # les combinaisons mixtes non voulues).
                if val not in ("none", "v1", "v2", "both"):
                    sys.exit(f"[generate_sweep] timing_mode={val!r} invalide "
                             "(attendu : none | v1 | v2 | both)")
                _set_nested(cfg, "observation.timing_feature", val in ("v1", "both"))
                _set_nested(cfg, "observation.timing_features_v2", val in ("v2", "both"))
                continue
            dot_path = PARAM_MAP[param_key]
            _set_nested(cfg, dot_path, val)

        run_name = f"{args.sweep_name}/run_{i:03d}"
        cfg["experiment"]["name"] = run_name

        cfg_path = sweep_cfg_dir / f"run_{i:03d}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        manifest.append({
            "index": i,
            "config": str(cfg_path.relative_to(ROOT)),
            "params_override": override,
        })

    manifest_path = manifest_dir / f"{args.sweep_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[generate_sweep] configs  → configs/sweeps/{args.sweep_name}/run_*.yaml")
    print(f"[generate_sweep] manifest → {manifest_path.relative_to(ROOT)}")
    print()
    print("Next step:")
    print(f"  ./cluster/sweep/launch_sweep.sh {args.sweep_name}")


if __name__ == "__main__":
    main()
