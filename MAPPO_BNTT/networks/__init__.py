"""Networks for MAPPO_BNTT — BN-augmented centralized critic + baseline actor."""

from MAPPO_BNTT.networks.critic import CentralizedCriticBN

# Re-export the unchanged actor from the baseline package
from MAPPO.networks.actor import ActorNetwork

__all__ = ["CentralizedCriticBN", "ActorNetwork"]
