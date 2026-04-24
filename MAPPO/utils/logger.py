"""Logging utilities with WandB integration."""

import os
from typing import Dict, Any, Optional
import wandb
import numpy as np


class WandBLogger:
    """
    Weights & Biases logger wrapper.
    
    Args:
        project: W&B project name
        group: W&B group name
        tags: List of tags
        config: Configuration dictionary
        name: Run name (optional)
        use_wandb: Whether to actually use W&B (or just print)
    """
    
    def __init__(
        self,
        project: str,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        use_wandb: bool = True
    ):
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            print(f"[DEBUG] Attempting to initialize W&B...")
            print(f"[DEBUG] Project: {project}, Name: {name}")
            print(f"[DEBUG] Checking for W&B API key...")
            
            # Check if W&B is already logged in
            try:
                api_key = os.environ.get('WANDB_API_KEY')
                if api_key:
                    print(f"[DEBUG] WANDB_API_KEY found in environment")
                else:
                    print(f"[DEBUG] WARNING: No WANDB_API_KEY in environment")
                    print(f"[DEBUG] W&B will attempt interactive login or use cached credentials")
            except Exception as e:
                print(f"[DEBUG] Error checking API key: {e}")
            
            try:
                self.run = wandb.init(
                    project=project,
                    group=group,
                    tags=tags,
                    config=config,
                    name=name
                )
                print(f"[DEBUG] W&B initialized successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to initialize W&B: {type(e).__name__}: {e}")
                print(f"[INFO] You can:")
                print(f"  1. Run 'wandb login' in terminal to authenticate")
                print(f"  2. Set WANDB_API_KEY environment variable")
                print(f"  3. Use --no-wandb flag to disable W&B logging")
                raise
        else:
            self.run = None
            print(f"[Logger] Initialized (W&B disabled) - Project: {project}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log data to W&B."""
        if self.use_wandb and self.run is not None:
            wandb.log(data, step=step)
        else:
            print(f"[Step {step}] {data}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None):
        """Log metrics with optional prefix."""
        data = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        self.log(data, step=step)
    
    def finish(self):
        """Finish the W&B run."""
        if self.use_wandb and self.run is not None:
            wandb.finish()


def setup_logging(log_dir: str, experiment_name: str) -> str:
    """
    Setup logging directory.
    
    Args:
        log_dir: Base logging directory
        experiment_name: Name of experiment
        
    Returns:
        Path to experiment log directory
    """
    exp_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    
    return exp_dir


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """Format metrics dictionary for printing."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        elif isinstance(value, (list, np.ndarray)):
            if len(value) > 0:
                mean_val = np.mean(value)
                formatted.append(f"{key}: {mean_val:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)
