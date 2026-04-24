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
            num_seconds=26000,  # Extended to see vehicles (depart at ~23500s)
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
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            value_clip=True
        ),
        training=TrainingConfig(
            max_epoch=100,
            step_per_epoch=10000,
            episode_per_collect=10,
            batch_size=256,
            repeat_per_collect=4,
            n_train_envs=4,
            n_test_envs=2
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
    config.sumo.num_seconds = 25000  # Extended to see vehicles (depart at ~23500s)
    config.training.max_epoch = 5
    config.training.step_per_epoch = 1000
    config.training.n_train_envs = 1
    config.training.n_test_envs = 1
    config.logging.use_wandb = False
    return config


def get_fast_test_config() -> ExperimentConfig:
    """Get configuration for fast testing (no GUI, shorter but with vehicles)."""
    config = get_default_config()
    config.sumo.num_seconds = 24000  # Shortened but still long enough for vehicles
    config.training.max_epoch = 10
    config.training.step_per_epoch = 2000
    config.training.n_train_envs = 4
    config.training.test_interval = 1
    config.training.save_interval = 1
    config.logging.use_wandb = False
    return config
