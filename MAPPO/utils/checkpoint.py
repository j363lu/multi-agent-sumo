"""Checkpoint saving and loading utilities."""

import os
import torch
from typing import Dict, Any, Optional
import json
from datetime import datetime


def save_checkpoint(
    policies: Dict[str, Any],
    critic: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    config: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None
) -> str:
    """
    Save training checkpoint.
    
    Args:
        policies: Dictionary of agent policies
        critic: Centralized critic network
        optimizers: Dictionary of optimizers
        epoch: Current epoch number
        metrics: Current metrics
        save_dir: Directory to save checkpoint
        config: Configuration dictionary (optional)
        filename: Custom filename (optional)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint_path = os.path.join(save_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'critic_state_dict': critic.state_dict(),
        'policies': {},
        'optimizers': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save each policy's state
    for agent_id, policy in policies.items():
        checkpoint['policies'][agent_id] = {
            'actor_state_dict': policy.actor.state_dict(),
        }
    
    # Save optimizer states
    for name, optimizer in optimizers.items():
        checkpoint['optimizers'][name] = optimizer.state_dict()
    
    # Add config if provided
    if config is not None:
        checkpoint['config'] = config
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Also save config separately as JSON for easy inspection
    if config is not None:
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    policies: Dict[str, Any],
    critic: torch.nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        policies: Dictionary of agent policies to load into
        critic: Centralized critic network to load into
        optimizers: Dictionary of optimizers to load into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint metadata (epoch, metrics, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load critic
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # Load policies
    for agent_id, policy in policies.items():
        if agent_id in checkpoint['policies']:
            policy.actor.load_state_dict(checkpoint['policies'][agent_id]['actor_state_dict'])
        else:
            print(f"Warning: No saved state for agent {agent_id}")
    
    # Load optimizers if provided
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            if name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][name])
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'config': checkpoint.get('config', None),
        'timestamp': checkpoint.get('timestamp', None)
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    
    if len(checkpoints) == 0:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    
    return os.path.join(checkpoint_dir, checkpoints[-1])
