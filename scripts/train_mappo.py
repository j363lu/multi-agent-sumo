#!/usr/bin/env python3
"""
Main training script for MAPPO traffic signal control.

Usage:
    python scripts/train_mappo.py --config configs/base_config.yaml
    python scripts/train_mappo.py --debug
    python scripts/train_mappo.py --use-gui --num-seconds 1000
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mappo_traffic.config import ExperimentConfig
from mappo_traffic.config.default_configs import get_default_config, get_debug_config, get_fast_test_config
from mappo_traffic.training import MAPPOTrainer


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
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (cpu or cuda)"
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


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config is not None:
        # Load from YAML file
        print(f"Loading config from: {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    elif args.debug:
        # Use debug config
        print("Using debug configuration")
        config = get_debug_config()
    elif args.fast_test:
        # Use fast test config
        print("Using fast test configuration")
        config = get_fast_test_config()
    else:
        # Use default config
        print("Using default configuration")
        config = get_default_config()
    
    # Apply command-line overrides
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
    
    # Print configuration
    print("\n" + "=" * 70)
    print("MAPPO Training Configuration")
    print("=" * 70)
    print(f"Environment: {config.sumo.net_file}")
    print(f"Simulation duration: {config.sumo.num_seconds}s")
    print(f"GUI: {config.sumo.use_gui}")
    print(f"Training epochs: {config.training.max_epoch}")
    print(f"Device: {config.device}")
    print(f"Random seed: {config.seed}")
    print(f"W&B logging: {config.logging.use_wandb}")
    print("=" * 70 + "\n")
    
    # Create trainer
    trainer = MAPPOTrainer(config)
    
    # Start training
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
        # Cleanup
        if hasattr(trainer, 'train_env'):
            trainer.train_env.close()
        if hasattr(trainer, 'test_env'):
            trainer.test_env.close()
        print("\nEnvironments closed")


if __name__ == "__main__":
    main()
