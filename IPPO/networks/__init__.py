"""Neural network architectures for IPPO."""

from IPPO.networks.actor import ActorNetwork
from IPPO.networks.critic import LocalCritic
from IPPO.networks.utils import init_orthogonal

__all__ = ["ActorNetwork", "LocalCritic", "init_orthogonal"]
