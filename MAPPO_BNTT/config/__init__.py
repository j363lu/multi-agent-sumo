"""Config package for MAPPO_BNTT."""

from MAPPO_BNTT.config.default_configs import (
    BNNetworkConfig,
    get_critic_bn_config,
)

__all__ = ["BNNetworkConfig", "get_critic_bn_config"]
