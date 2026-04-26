#!/usr/bin/env python3
"""
Main training script for MAPPO traffic signal control.

Supports both the baseline (standard MLP critic) and the BNTT experiment
variant (BatchNorm1d on the centralized critic) through a single entry-point.

Usage:
    # Baseline
    python scripts/train_mappo.py --seed 0
    python scripts/train_mappo.py --config configs/base_config.yaml --seed 0

    # BN-critic variant (BNTT experiment)
    python scripts/train_mappo.py --use-critic-bn --seed 0
    python scripts/train_mappo.py --config configs/critic_bn_config.yaml --seed 0

    # Other presets
    python scripts/train_mappo.py --debug
    python scripts/train_mappo.py --fast-test
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Auto-detect SUMO_HOME / PROJ_DATA from the installed sumo package so
# collaborators don't need to set shell environment variables manually.
def _configure_sumo_env() -> None:
    if not os.environ.get("SUMO_HOME"):
        try:
            import sumo as _sumo_pkg
            os.environ["SUMO_HOME"] = os.path.dirname(_sumo_pkg.__file__)
        except ImportError:
            pass
    if not os.environ.get("PROJ_DATA"):
        candidate = os.path.join(os.environ.get("SUMO_HOME", ""), "data", "proj")
        if os.path.isdir(candidate):
            os.environ["PROJ_DATA"] = candidate

_configure_sumo_env()

import yaml

from MAPPO.config import ExperimentConfig
from MAPPO.config.default_configs import get_default_config, get_debug_config, get_fast_test_config
from MAPPO.training import MAPPOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO for traffic signal control")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    # Preset configs
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug config (with GUI, short training)"
    )
    
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Use fast test config (no GUI, short episodes)"
    )

    # ── BNTT experiment flag ───────────────────────────────────────────────
    parser.add_argument(
        "--use-critic-bn",
        action="store_true",
        help=(
            "Enable BatchNorm1d on the centralized critic (BNTT experiment). "
            "Loads get_critic_bn_config() and uses MAPPOBNTTTrainer. "
            "All other hyperparameters stay identical to the baseline."
        ),
    )

    # Environment overrides
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Use SUMO GUI"
    )
    
    parser.add_argument(
        "--num-seconds",
        type=int,
        default=None,
        help="Simulation duration in seconds"
    )
    
    parser.add_argument(
        "--net-file",
        type=str,
        default=None,
        help="Path to SUMO network file"
    )
    
    parser.add_argument(
        "--route-file",
        type=str,
        default=None,
        help="Path to SUMO route file"
    )
    
    # Training overrides
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Maximum training epochs"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use (cpu, cuda, or mps)"
    )
    
    # Logging overrides
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging (disabled by default)"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Explicitly disable W&B logging"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Logging directory"
    )
    
    return parser.parse_args()


def _yaml_requests_critic_bn(yaml_path: str) -> bool:
    """Return True if the YAML file sets network.use_critic_bn to true."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    return bool(raw.get("network", {}).get("use_critic_bn", False))


def _load_config_from_yaml(yaml_path: str):
    """
    Load an ExperimentConfig from YAML, handling the BNTT extension.

    If the YAML sets network.use_critic_bn=true we must use BNNetworkConfig
    (from MAPPO_BNTT) instead of the baseline NetworkConfig, because
    NetworkConfig does not have that field and would raise a TypeError.

    Returns (config, use_critic_bn: bool).
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    use_critic_bn = bool(raw.get("network", {}).get("use_critic_bn", False))

    if use_critic_bn:
        from MAPPO_BNTT.config.default_configs import BNNetworkConfig
        from MAPPO.config.config import (
            SumoConfig, MAPPOConfig, TrainingConfig, LoggingConfig,
        )

        net_raw = raw.get("network", {})
        # Strip use_critic_bn before passing to BNNetworkConfig so we
        # don't double-set it; BNNetworkConfig will receive it explicitly.
        net_raw_clean = {k: v for k, v in net_raw.items() if k != "use_critic_bn"}

        config = ExperimentConfig(
            sumo=SumoConfig(**raw.get("sumo", {})),
            network=BNNetworkConfig(**net_raw_clean, use_critic_bn=True),
            mappo=MAPPOConfig(**raw.get("mappo", {})),
            training=TrainingConfig(**raw.get("training", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            seed=raw.get("seed", 42),
            device=raw.get("device", "cpu"),
        )
    else:
        config = ExperimentConfig.from_yaml(yaml_path)

    return config, use_critic_bn


def main():
    """Main training function."""
    args = parse_args()

    # ── Determine whether this is a BN-critic run ───────────────────────────
    # Priority: --use-critic-bn flag > YAML network.use_critic_bn > False
    use_critic_bn = args.use_critic_bn

    # ── Load configuration ───────────────────────────────────────────────────
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config, yaml_use_bn = _load_config_from_yaml(args.config)
        use_critic_bn = use_critic_bn or yaml_use_bn
    elif args.debug:
        print("Using debug configuration")
        config = get_debug_config()
    elif args.fast_test and use_critic_bn:
        # Both --fast-test and --use-critic-bn: use BN fast-test preset so that
        # the BN variant gets identical schedule settings (test_interval=1,
        # n_test_envs=5) as the baseline fast-test.  Without this, --fast-test
        # alone would silently load get_fast_test_config() (baseline NetworkConfig)
        # and the BN critic would never be created.
        from MAPPO_BNTT.config.default_configs import get_critic_bn_fast_test_config
        print("Using critic-BN fast-test configuration (BNTT experiment)")
        config = get_critic_bn_fast_test_config()
    elif args.fast_test:
        print("Using fast test configuration")
        config = get_fast_test_config()
    elif use_critic_bn:
        # --use-critic-bn with no explicit --config → load BN preset
        from MAPPO_BNTT.config.default_configs import get_critic_bn_config
        print("Using critic-BN configuration (BNTT experiment)")
        config = get_critic_bn_config()
    else:
        print("Using default configuration")
        config = get_default_config()

    # ── Apply command-line overrides ─────────────────────────────────────────
    if args.use_gui:
        config.sumo.use_gui = True
    
    if args.num_seconds is not None:
        config.sumo.num_seconds = args.num_seconds
    
    if args.net_file is not None:
        config.sumo.net_file = args.net_file
    
    if args.route_file is not None:
        config.sumo.route_file = args.route_file
    
    if args.max_epoch is not None:
        config.training.max_epoch = args.max_epoch
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.device is not None:
        config.device = args.device
    
    if args.wandb:
        config.logging.use_wandb = True
    
    if args.no_wandb:
        config.logging.use_wandb = False
    
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
    
    if args.log_dir is not None:
        config.logging.log_dir = args.log_dir

    # ── Select trainer class ─────────────────────────────────────────────────
    if use_critic_bn:
        from MAPPO_BNTT.training.trainer import MAPPOBNTTTrainer
        TrainerClass = MAPPOBNTTTrainer
        variant_label = "critic-BN (BNTT experiment)"
    else:
        TrainerClass = MAPPOTrainer
        variant_label = "baseline"

    # ── Print configuration ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MAPPO Training Configuration")
    print("=" * 70)
    print(f"Variant:             {variant_label}")
    print(f"Environment:         {config.sumo.net_file}")
    print(f"Simulation duration: {config.sumo.num_seconds}s")
    print(f"GUI:                 {config.sumo.use_gui}")
    print(f"Training epochs:     {config.training.max_epoch}")
    print(f"Device:              {config.device}")
    print(f"Random seed:         {config.seed}")
    print(f"W&B logging:         {config.logging.use_wandb}")
    print("=" * 70 + "\n")
    
    # ── Create trainer and run ───────────────────────────────────────────────
    trainer = TrainerClass(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if hasattr(trainer, 'train_env'):
            trainer.train_env.close()
        if hasattr(trainer, 'test_env'):
            trainer.test_env.close()
        print("\nEnvironments closed")


if __name__ == "__main__":
    main()
