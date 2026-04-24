"""Centralized critic network for MAPPO."""

import torch
import torch.nn as nn
from typing import List
import numpy as np


class CentralizedCritic(nn.Module):
    """
    Centralized critic that takes concatenated observations from all agents.
    
    This critic sees the global state (all agent observations) and outputs
    a value estimate for the joint state-action.
    
    Args:
        global_obs_dim: Dimension of concatenated observations from all agents
        hidden_dims: List of hidden layer dimensions
        activation: Activation function (default: ReLU)
        use_orthogonal_init: Whether to use orthogonal initialization
    """
    
    def __init__(
        self,
        global_obs_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        
        self.global_obs_dim = global_obs_dim
        
        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = global_obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value head outputs scalar value
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self._init_orthogonal()
    
    def _init_orthogonal(self):
        """Initialize network with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            global_obs: Concatenated observations from all agents
                       Shape: (batch_size, global_obs_dim)
            
        Returns:
            Value estimate of shape (batch_size, 1)
        """
        features = self.feature_extractor(global_obs)
        value = self.value_head(features)
        return value
    
    def get_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for global state.
        
        Args:
            global_obs: Concatenated observations from all agents
            
        Returns:
            Value tensor of shape (batch_size,)
        """
        value = self.forward(global_obs)
        return value.squeeze(-1)
