"""Default configuration presets."""

import torch
from MAPPO.config.config import ExperimentConfig, SumoConfig, NetworkConfig, MAPPOConfig, TrainingConfig, LoggingConfig


def _best_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_config() -> ExperimentConfig:
    """Get default configuration for cologne3."""
    return ExperimentConfig(
        sumo=SumoConfig(
                net_file="RESCO/cologne3/cologne3.net.xml",
                route_file="RESCO/cologne3/cologne3.rou.xml",
                # cologne3 vehicles depart 23512s–28798s; begin_time skips the empty
                # warm-up without touching the route file, preserving benchmark validity.
                begin_time=23400,   # 100s before first departure (23512s)
                num_seconds=5600,   # episode runs 23400→29000s, past last departure (28798s)
                use_gui=False,
                delta_time=5
            ),
        network=NetworkConfig(
            actor_hidden=[128, 128],
            critic_hidden=[256, 256],
            activation="relu",
            use_orthogonal_init=True
        ),
        mappo=MAPPOConfig(
            # lr_actor lowered 3e-4 → 1e-4 to prevent large actor updates
            # from flipping the policy between the bimodal cologne3 traffic
            # attractors ("clear" vs "gridlock") in a single gradient step.
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            # value_clip re-enabled now that reward_normalization=True keeps
            # critic targets O(1).  Previously had to disable it because
            # unnormalized rewards made critic values O(100+), causing the
            # clamp range (±eps_clip=0.2) to saturate immediately and zero
            # all critic gradients.
            value_clip=True,
            # max_grad_norm lowered 0.5 → 0.3 as a second line of defence
            # against catastrophic policy jumps caused by noisy advantage
            # estimates (critic_loss 3–10 → RMS error 1.7–3.2 per step).
            max_grad_norm=0.3,
            # Normalize per-step rewards to zero-mean unit-variance using a
            # shared Welford running normalizer (clip ±10).  This reduces
            # critic targets from O(reward/(1-gamma)) ≈ O(100) to O(1),
            # which is the root-cause fix for the high critic loss (3–10)
            # and resulting inaccurate advantage estimates that caused
            # catastrophic policy flips between the cologne3 attractors.
            reward_normalization=True,
        ),
        training=TrainingConfig(
            max_epoch=100,
            step_per_epoch=10000,
            # 30 episodes averages out the bimodal gridlock/no-gridlock variance
            # that caused large training-reward swings with only 10 episodes.
            episode_per_collect=30,
            batch_size=256,
            # 2 passes reduce critic overfitting to a single stale collect batch,
            # smoothing the critic-loss spike seen between epochs.
            repeat_per_collect=2,
            n_train_envs=4,
            # n_test_envs raised 2 → 5: with only 2 fixed-seed eval episodes
            # over a bimodal env the reported mean can only take 3 values
            # (-326, -163, -0.9), making evaluation appear as binary jumps.
            # 5 episodes spread the mean across more of the distribution.
            n_test_envs=5
        ),
        logging=LoggingConfig(
            project="sumo-mappo-traffic",
            group="cologne3",
            use_wandb=False  # Disabled by default - use --wandb flag or set to True to enable
        ),
        seed=42,
        device=_best_device()
    )


def get_debug_config() -> ExperimentConfig:
    """Get debug configuration (faster, headless by default)."""
    config = get_default_config()
    config.sumo.use_gui = False  # Changed to False - GUI requires X11/XQuartz on macOS
    # begin_time and num_seconds inherited from get_default_config() (23400 / 5600)
    config.training.max_epoch = 5
    config.training.step_per_epoch = 1000
    config.training.n_train_envs = 1
    config.training.n_test_envs = 1
    config.logging.use_wandb = False
    return config


def get_fast_test_config() -> ExperimentConfig:
    """Get configuration for fast testing (no GUI, shorter but with vehicles)."""
    config = get_default_config()
    # begin_time=23400, num_seconds=5600 inherited from get_default_config()
    config.training.max_epoch = 10
    config.training.step_per_epoch = 2000
    config.training.n_train_envs = 4
    config.training.test_interval = 1
    config.training.save_interval = 1
    config.logging.use_wandb = False
    return config
