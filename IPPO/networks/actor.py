"""Actor network for discrete action spaces in traffic signal control."""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor network that outputs discrete action distributions.
    
    Takes local observation from a traffic signal and outputs
    a categorical distribution over available traffic phases.
    
    Args:
        obs_dim: Dimension of observation space
        action_dim: Number of discrete actions (traffic phases)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function (default: ReLU)
        use_orthogonal_init: Whether to use orthogonal initialization
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        use_orthogonal_init: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
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
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Action logits layer
        self.action_head = nn.Linear(prev_dim, action_dim)
        
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
        
        # Special initialization for last layer
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        features = self.feature_extractor(obs)
        logits = self.action_head(features)
        return logits
    
    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get categorical distribution over actions.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Categorical distribution
        """
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action from the policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action
            
        Returns:
            Action tensor
        """
        dist = self.get_action_distribution(obs)
        
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        
        return action
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            obs: Observation tensor
            actions: Actions to evaluate
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of the distribution
        """
        dist = self.get_action_distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
