"""Utility functions for neural networks."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def init_orthogonal(module: nn.Module, gain: float = np.sqrt(2)):
    """
    Initialize module with orthogonal initialization.
    
    Args:
        module: Neural network module
        gain: Gain factor for initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_xavier(module: nn.Module):
    """Initialize module with Xavier initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation module
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {activation}")


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0
) -> nn.Module:
    """
    Initialize a layer with orthogonal initialization.
    
    Args:
        layer: Layer to initialize
        std: Standard deviation for weight initialization
        bias_const: Constant value for bias initialization
        
    Returns:
        Initialized layer
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Rewards array of shape (num_steps, num_envs)
        values: Value estimates of shape (num_steps, num_envs)
        dones: Done flags of shape (num_steps, num_envs)
        next_values: Next value estimates of shape (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: Advantage estimates
        returns: Return estimates (advantages + values)
    """
    num_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
        
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantage tensor
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)
