"""
Centralized critic with BatchNorm1d applied to each hidden layer.

Motivation (from Kim & Panda 2021 — BNTT paper):
  Standard batch normalization is insufficient when the input distribution
  varies over time due to a mismatch between stored statistics and current
  inputs.  In online RL, the critic's input distribution shifts as the
  policy evolves (non-stationarity).  Inserting BatchNorm1d before each
  hidden-layer activation stabilises the internal covariate shift of the
  critic without requiring temporal decoupling (BNTT), which is the key
  architectural hypothesis of this experiment.

Architecture (with use_batch_norm=True):
    Input → Linear → BatchNorm1d → ReLU
          → Linear → BatchNorm1d → ReLU
          → ...
          → Linear (value head, no BN)

The baseline CentralizedCritic (use_batch_norm=False) is identical to
MAPPO/networks/critic.py; this class adds BN as the only modification.
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np

from MAPPO.networks.critic import CentralizedCritic


class CentralizedCriticBN(CentralizedCritic):
    """
    Centralized critic with optional BatchNorm1d on hidden layers.

    Subclasses CentralizedCritic from MAPPO/ so that all interface
    methods (forward, get_value) and orthogonal initialisation are
    inherited unchanged.  Only the feature_extractor is rebuilt when
    use_batch_norm=True.

    Args:
        global_obs_dim: Dimension of concatenated observations from all agents.
        hidden_dims: List of hidden layer sizes.
        activation: Activation function name ("relu", "tanh", "elu").
        use_orthogonal_init: Whether to apply orthogonal weight init.
        use_batch_norm: If True, insert BatchNorm1d before each activation.
    """

    def __init__(
        self,
        global_obs_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_orthogonal_init: bool = True,
        use_batch_norm: bool = True,
    ):
        # Call parent __init__ which builds the baseline feature_extractor and
        # value_head, then applies orthogonal init.  We will then replace
        # feature_extractor with the BN-augmented version if requested.
        super().__init__(
            global_obs_dim=global_obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_orthogonal_init=use_orthogonal_init,
        )

        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            # Rebuild feature_extractor as Linear → BatchNorm1d → Activation.
            # The value_head is kept as-is (no BN on the output layer).
            act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
            if activation not in act_map:
                raise ValueError(f"Unknown activation: {activation}")
            act_fn = act_map[activation]

            layers = []
            prev_dim = global_obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_fn())
                prev_dim = hidden_dim

            self.feature_extractor = nn.Sequential(*layers)

            # Re-apply orthogonal init to the new Linear layers if requested.
            # BatchNorm parameters are left at their PyTorch defaults
            # (weight=1, bias=0) which is standard BN practice.
            if use_orthogonal_init:
                for module in self.feature_extractor.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.0)

    def extra_repr(self) -> str:
        return f"use_batch_norm={self.use_batch_norm}"
