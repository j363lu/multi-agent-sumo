"""
Configuration presets for the MAPPO_BNTT (BN-critic) experiment.

BNNetworkConfig extends MAPPO's NetworkConfig by adding a single
`use_critic_bn` field.  This keeps MAPPO/ untouched while providing a
typed, serialisable config that the MAPPOBNTTTrainer reads at runtime.

get_critic_bn_config() returns an ExperimentConfig whose .network field
is a BNNetworkConfig instance with use_critic_bn=True.  Every other
hyperparameter is inherited from get_default_config() (cologne3, same
LR, same epochs, same metrics) so the A/B comparison is fully controlled.
"""

from dataclasses import dataclass, field
from typing import List

from MAPPO.config.config import ExperimentConfig
from MAPPO.config.default_configs import get_default_config


@dataclass
class BNNetworkConfig:
    """
    Drop-in replacement for NetworkConfig that adds use_critic_bn.

    All fields mirror NetworkConfig so that MAPPOTrainer._create_policy_manager()
    can read actor_hidden, critic_hidden, activation, and use_orthogonal_init
    identically.  The extra use_critic_bn field is consumed exclusively by
    MAPPOBNTTTrainer.
    """

    actor_hidden: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_orthogonal_init: bool = True
    # ── BNTT experiment knob ────────────────────────────────────────────────
    # True  → CentralizedCriticBN (Linear → BatchNorm1d → Activation)
    # False → standard CentralizedCritic (identical to baseline MAPPO)
    use_critic_bn: bool = True


def get_critic_bn_config() -> ExperimentConfig:
    """
    Experiment config for the BN-on-critic variant.

    Starts from get_default_config() (cologne3, all tuned hyperparameters)
    and replaces only the network config with a BNNetworkConfig whose
    use_critic_bn=True.  This is the only structural difference from the
    baseline, ensuring a controlled comparison.

    Logging group is set to 'cologne3-critic-bn' so W&B / CSV files are
    automatically grouped separately from baseline runs.
    """
    config = get_default_config()

    # Replace the standard NetworkConfig with the BN-aware variant.
    # All other fields (hidden dims, activation, orthogonal init) are
    # copied from the default so architecture size is identical.
    default_net = config.network
    config.network = BNNetworkConfig(
        actor_hidden=list(default_net.actor_hidden),
        critic_hidden=list(default_net.critic_hidden),
        activation=default_net.activation,
        use_orthogonal_init=default_net.use_orthogonal_init,
        use_critic_bn=True,
    )

    # Separate logging group for easy filtering in W&B / CSV aggregation.
    config.logging.group = "cologne3-critic-bn"

    return config


def get_critic_bn_fast_test_config() -> ExperimentConfig:
    """
    Fast-test preset for the BN-on-critic variant.

    Starts from get_critic_bn_config() (BN network, tuned hyperparameters)
    and applies the same fast-test overrides as MAPPO's get_fast_test_config():
        - max_epoch=10
        - test_interval=1  (eval after every epoch → 10 eval rows in CSV)
        - save_interval=1
        - n_test_envs=5    (non-zero std_waiting_time shading in plots)

    Use this preset when running the sweep with --fast-test --use-critic-bn
    so that both conditions (baseline and BN) are trained under identical
    schedule settings, making epoch-to-epoch comparison valid.
    """
    config = get_critic_bn_config()
    config.training.max_epoch = 10
    config.training.step_per_epoch = 2000
    config.training.n_train_envs = 4
    config.training.test_interval = 1
    config.training.save_interval = 1
    config.logging.use_wandb = False
    return config
