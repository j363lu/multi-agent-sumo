"""
Configuration dataclasses for MAPPO training.

Provides structured configuration for all aspects of the training pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class SumoConfig:
    """Configuration for SUMO environment."""
    
    net_file: str = "RESCO/cologne3/cologne3.net.xml"
    route_file: str = "RESCO/cologne3/cologne3.rou.xml"
    num_seconds: int = 3600
    use_gui: bool = False
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50


@dataclass
class NetworkConfig:
    """Configuration for neural networks."""
    
    actor_hidden: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_orthogonal_init: bool = True


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO algorithm."""
    
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    value_clip: bool = True
    advantage_normalization: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_normalization: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    max_epoch: int = 100
    step_per_epoch: int = 10000
    episode_per_collect: int = 10
    batch_size: int = 256
    repeat_per_collect: int = 4
    n_train_envs: int = 4
    n_test_envs: int = 2
    test_interval: int = 5
    save_interval: int = 10
    log_interval: int = 10


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    project: str = "sumo-mappo-traffic"
    group: str = "cologne3"
    tags: List[str] = field(default_factory=lambda: ["mappo", "traffic-control"])
    use_wandb: bool = True
    log_dir: str = "logs"
    experiment_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    sumo: SumoConfig = field(default_factory=SumoConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            sumo=SumoConfig(**config_dict.get("sumo", {})),
            network=NetworkConfig(**config_dict.get("network", {})),
            mappo=MAPPOConfig(**config_dict.get("mappo", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda")
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
