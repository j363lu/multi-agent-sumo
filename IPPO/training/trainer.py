"""
IPPO Trainer for SUMO traffic control.

Main training loop implementation.
"""

import csv
import os
import random
import time
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.optim import Adam
from tianshou.data import Batch
from tqdm import tqdm

from IPPO.config import ExperimentConfig
from IPPO.envs import SumoTianshouEnv
from IPPO.networks import ActorNetwork, LocalCritic
from IPPO.agents import IPPOPolicy, MultiAgentPolicyManager
from IPPO.utils import WandBLogger, save_checkpoint, load_checkpoint, MetricsTracker, RewardNormalizer
from IPPO.training.evaluator import evaluate_policy


class IPPOTrainer:
    """
    IPPO Trainer for multi-agent traffic signal control.
    
    Args:
        config: Experiment configuration
    """
    
    def __init__(self, config: ExperimentConfig, resume_from: Optional[str] = None):
        self.config = config
        self.resume_from = resume_from
        
        # Set device — priority: cuda > mps > cpu
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create training environment
        print("Creating training environment...")
        self.train_env = self._create_env(config.sumo, num_envs=config.training.n_train_envs)
        
        # Create test environment
        print("Creating test environment...")
        self.test_env = self._create_env(config.sumo, num_envs=config.training.n_test_envs)
        
        # Get environment info
        self.agent_ids = self.train_env.agents
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.n
        
        print(f"Number of agents: {len(self.agent_ids)}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")

        # Shared reward normalizer — must be created BEFORE _create_policy_manager()
        # so it can be injected into each IPPOPolicy at construction time.
        # One instance for the whole experiment so all agents contribute to the same
        # running mean/variance.  None when reward_normalization=False.
        self._reward_normalizer: Optional[RewardNormalizer] = (
            RewardNormalizer() if config.ippo.reward_normalization else None
        )

        # Create networks and policies
        print("Creating networks and policies...")
        self.policy_manager = self._create_policy_manager()
        
        # Setup logging. Resumed runs keep writing into the checkpoint's run dir.
        if self.resume_from is not None:
            checkpoint_dir = os.path.dirname(os.path.abspath(self.resume_from))
            self.log_dir = os.path.dirname(checkpoint_dir)
            exp_name = os.path.basename(self.log_dir)
            config.logging.log_dir = os.path.dirname(self.log_dir)
            config.logging.experiment_name = exp_name
        else:
            exp_name = config.logging.experiment_name or f"ippo_{int(time.time())}"
            self.log_dir = os.path.join(config.logging.log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = WandBLogger(
            project=config.logging.project,
            group=config.logging.group,
            tags=config.logging.tags,
            config=config.to_dict(),
            name=exp_name,
            use_wandb=config.logging.use_wandb
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=100)
        
        # Training state
        self.current_epoch = 0
        self.start_epoch = 0
        self.total_steps = 0
        self._wallclock_offset = 0.0
        self.training_start_time: Optional[float] = None
        # Monotonically increasing episode counter used to derive per-episode
        # SUMO seeds: seed = config.seed * 10_000 + _global_episode_count.
        # Advancing across epochs ensures each epoch sees fresh traffic
        # realizations while the full sequence stays reproducible given the
        # same config.seed.
        self._global_episode_count: int = 0

        # CSV metrics file for offline plotting
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        append_metrics = self.resume_from is not None and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        if append_metrics:
            self._wallclock_offset = self._read_last_wallclock(csv_path)
        self._csv_file = open(csv_path, "a" if append_metrics else "w", newline="")
        self._csv_fields = [
            "epoch", "split", "wallclock_time_s",
            "episode_reward", "episode_length",
            "mean_waiting_time", "std_waiting_time",
            "mean_queue_length", "std_queue_length",
            "loss", "actor_loss", "critic_loss", "entropy", "clip_frac",
        ]
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_fields, restval=""
        )
        if not append_metrics:
            self._csv_writer.writeheader()

        if self.resume_from is not None:
            self._load_training_checkpoint(self.resume_from)
        
        print("Trainer initialized successfully!")

    def _read_last_wallclock(self, csv_path: str) -> float:
        """Return the last recorded wallclock time from an existing metrics CSV."""
        try:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            for row in reversed(rows):
                value = row.get("wallclock_time_s", "")
                if value != "":
                    return float(value)
        except (OSError, ValueError):
            pass
        return 0.0
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility.

        Network weight initialisation always uses a fixed seed (0) so model
        weights are identical across runs with different config.seed values.
        config.seed continues to control episode/environment randomness.
        """
        weight_init_seed = 0
        torch.manual_seed(weight_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(weight_init_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(weight_init_seed)

        np.random.seed(seed)
    
    def _create_env(self, sumo_config, num_envs: int = 1) -> SumoTianshouEnv:
        """Create SUMO environment."""
        # For simplicity, we create a single environment
        # In practice, you might want vectorized environments
        env = SumoTianshouEnv(
            net_file=sumo_config.net_file,
            route_file=sumo_config.route_file,
            use_gui=sumo_config.use_gui,
            num_seconds=sumo_config.num_seconds,
            begin_time=sumo_config.begin_time,
            delta_time=sumo_config.delta_time,
            yellow_time=sumo_config.yellow_time,
            min_green=sumo_config.min_green,
            max_green=sumo_config.max_green
        )
        return env
    
    def _create_policy_manager(self) -> MultiAgentPolicyManager:
        """Create IPPO policy manager with networks and optimizers."""
        net_config = self.config.network
        ippo_config = self.config.ippo
        
        # Get observation dimensions for each agent (may be different!)
        agent_obs_dims = {}
        for agent_id in self.agent_ids:
            obs_space = self.train_env.sumo_env.observation_space(agent_id)
            agent_obs_dims[agent_id] = obs_space.shape[0]
        
        print(f"Agent observation dimensions: {agent_obs_dims}")

        # Create independent actor/critic networks and policies for each agent.
        policies = []
        for agent_id in self.agent_ids:
            actor = ActorNetwork(
                obs_dim=agent_obs_dims[agent_id],
                action_dim=self.action_dim,
                hidden_dims=net_config.actor_hidden,
                activation=net_config.activation,
                use_orthogonal_init=net_config.use_orthogonal_init
            ).to(self.device)

            critic = LocalCritic(
                obs_dim=agent_obs_dims[agent_id],
                hidden_dims=net_config.critic_hidden,
                activation=net_config.activation,
                use_orthogonal_init=net_config.use_orthogonal_init
            ).to(self.device)
            
            actor_optimizer = Adam(actor.parameters(), lr=ippo_config.lr_actor)
            critic_optimizer = Adam(critic.parameters(), lr=ippo_config.lr_critic)
            
            policy = IPPOPolicy(
                actor=actor,
                critic=critic,
                optim_actor=actor_optimizer,
                optim_critic=critic_optimizer,
                gamma=ippo_config.gamma,
                gae_lambda=ippo_config.gae_lambda,
                eps_clip=ippo_config.eps_clip,
                value_clip=ippo_config.value_clip,
                dual_clip=ippo_config.dual_clip,
                advantage_normalization=ippo_config.advantage_normalization,
                vf_coef=ippo_config.vf_coef,
                ent_coef=ippo_config.ent_coef,
                max_grad_norm=ippo_config.max_grad_norm,
                reward_normalization=ippo_config.reward_normalization,
                # Shared normalizer: all agents contribute to the same running
                # mean/variance so normalization is consistent across agents.
                reward_normalizer=self._reward_normalizer,
            )
            
            policies.append(policy)
        
        policy_manager = MultiAgentPolicyManager(
            policies=policies,
            agent_ids=self.agent_ids
        )
        
        return policy_manager

    def _get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Return optimizer dict using the same keys as checkpoint files."""
        optimizers = {}
        for agent_id, policy in self.policy_manager.policies.items():
            optimizers[f'actor_{agent_id}'] = policy.optim_actor
            optimizers[f'critic_{agent_id}'] = policy.optim_critic
        return optimizers

    def _load_training_checkpoint(self, checkpoint_path: str):
        """Restore model/optimizer state and set the next epoch to train."""
        metadata = load_checkpoint(
            checkpoint_path=checkpoint_path,
            policies=self.policy_manager.policies,
            optimizers=self._get_optimizers(),
            device=str(self.device)
        )
        self.start_epoch = int(metadata.get('epoch', 0))
        self.current_epoch = self.start_epoch
        self._restore_extra_state(metadata.get('extra_state', {}))
        if self.config.training.use_fixed_episode_seeds and self._global_episode_count == 0:
            self._global_episode_count = self.start_epoch * self.config.training.episode_per_collect
        print(f"Resuming IPPO training from epoch {self.start_epoch + 1}")

    def _checkpoint_extra_state(self) -> Dict[str, Any]:
        """Collect non-module training state needed for faithful resume."""
        extra_state: Dict[str, Any] = {
            "global_episode_count": self._global_episode_count,
            "total_steps": self.total_steps,
            "metrics_tracker": self._metrics_tracker_state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": self._pack_numpy_rng_state(np.random.get_state()),
                "torch": torch.get_rng_state(),
            },
        }
        if torch.cuda.is_available():
            extra_state["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
        if self._reward_normalizer is not None:
            extra_state["reward_normalizer"] = self._reward_normalizer.state_dict()
        return extra_state

    def _pack_numpy_rng_state(self, state: tuple) -> tuple:
        """Convert NumPy RNG state to a checkpoint-friendly tuple."""
        name, keys, pos, has_gauss, cached_gaussian = state
        return (name, keys.tolist(), pos, has_gauss, cached_gaussian)

    def _unpack_numpy_rng_state(self, state: tuple) -> tuple:
        """Convert serialized NumPy RNG state back to NumPy's expected format."""
        name, keys, pos, has_gauss, cached_gaussian = state
        return (name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian)

    def _metrics_tracker_state_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.metrics_tracker.window_size,
            "metrics": {k: list(v) for k, v in self.metrics_tracker.metrics.items()},
            "episode_metrics": {
                k: list(v) for k, v in self.metrics_tracker.episode_metrics.items()
            },
        }

    def _load_metrics_tracker_state_dict(self, state: Dict[str, Any]):
        if not state:
            return
        window_size = int(state.get("window_size", self.metrics_tracker.window_size))
        self.metrics_tracker.window_size = window_size
        self.metrics_tracker.metrics = {
            k: deque(v, maxlen=window_size)
            for k, v in state.get("metrics", {}).items()
        }
        self.metrics_tracker.episode_metrics = {
            k: list(v) for k, v in state.get("episode_metrics", {}).items()
        }

    def _restore_extra_state(self, extra_state: Dict[str, Any]):
        """Restore optional trainer state from new-format checkpoints."""
        if not extra_state:
            print("Warning: checkpoint has no extra trainer state; resume may not be exact")
            return

        self._global_episode_count = int(extra_state.get("global_episode_count", self._global_episode_count))
        self.total_steps = int(extra_state.get("total_steps", self.total_steps))
        self._load_metrics_tracker_state_dict(extra_state.get("metrics_tracker", {}))

        normalizer_state = extra_state.get("reward_normalizer")
        if normalizer_state is not None and self._reward_normalizer is not None:
            self._reward_normalizer.load_state_dict(normalizer_state)

        rng_state = extra_state.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(self._unpack_numpy_rng_state(rng_state["numpy"]))
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].detach().cpu())
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([
                state.detach().cpu() for state in rng_state["cuda"]
            ])
    
    def train(self):
        """Main training loop."""
        self.training_start_time = time.time()

        print("\n" + "=" * 60)
        print("Starting IPPO Training")
        print("=" * 60)
        
        config = self.config.training
        
        if self.start_epoch >= config.max_epoch:
            print(
                f"Checkpoint is already at epoch {self.start_epoch}; "
                f"max_epoch is {config.max_epoch}, so there is nothing to train."
            )

        epoch_pbar = tqdm(range(self.start_epoch, config.max_epoch), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Collect training data
            print(f"\nEpoch {epoch + 1}/{config.max_epoch}")
            print("-" * 60)
            
            collect_result = self._collect_episodes(
                self.train_env,
                n_episode=config.episode_per_collect
            )
            
            # Training update
            train_result = self._update_policy(
                collect_result,
                batch_size=config.batch_size,
                repeat=config.repeat_per_collect
            )
            
            # Log training metrics
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch + 1, collect_result, train_result, epoch_time)

            epoch_pbar.set_postfix({
                "rew": f"{collect_result['episode_reward']:.2f}",
                "loss": f"{train_result['loss']:.4f}",
            })
            
            # Evaluation
            if (epoch + 1) % config.test_interval == 0:
                eval_result = self._evaluate()
                self._log_evaluation(epoch + 1, eval_result)
            
            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(self.metrics_tracker.summary())
        
        self.logger.finish()
        self._csv_file.close()
    
    def _collect_episodes(self, env, n_episode: int) -> Dict[str, Any]:
        """
        Collect episodes and store full transitions for PPO training.

        Returns a dict with per-agent trajectory data plus summary stats.
        """
        # Per-agent trajectory storage
        agent_obs:       Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_next_obs:  Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_actions:   Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_log_probs: Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_rewards:   Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_term:      Dict[str, List] = {aid: [] for aid in self.agent_ids}
        agent_trunc:     Dict[str, List] = {aid: [] for aid in self.agent_ids}
        episode_rewards = []
        episode_lengths = []

        for _ in tqdm(range(n_episode), desc="Collecting", unit="ep", leave=False):
            # Deterministic seed cycling: each episode gets a unique seed derived
            # from config.seed so results are reproducible across runs with the
            # same seed, but different epochs always sample new traffic realizations.
            if self.config.training.use_fixed_episode_seeds:
                episode_seed = self.config.seed * 10_000 + self._global_episode_count
                obs_dict, _ = env.reset(seed=episode_seed)
            else:
                obs_dict, _ = env.reset()
            self._global_episode_count += 1
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Get actions and log-probs from each agent's policy
                actions = {}
                log_probs = {}
                for agent_id in self.agent_ids:
                    obs_t = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0).to(self.device)
                    policy = self.policy_manager.policies[agent_id]
                    with torch.no_grad():
                        dist = policy.actor.get_action_distribution(obs_t)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    actions[agent_id]   = action.item()
                    log_probs[agent_id] = log_prob.item()

                # Step environment
                next_obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(actions)

                # Store transition for every agent
                for aid in self.agent_ids:
                    agent_obs[aid].append(obs_dict[aid])
                    agent_next_obs[aid].append(next_obs_dict[aid])
                    agent_actions[aid].append(actions[aid])
                    agent_log_probs[aid].append(log_probs[aid])
                    agent_rewards[aid].append(reward_dict.get(aid, 0.0))
                    agent_term[aid].append(term_dict.get(aid, False))
                    agent_trunc[aid].append(trunc_dict.get(aid, False))

                episode_reward += sum(reward_dict.values()) / max(len(reward_dict), 1)
                episode_length += 1

                done = any(term_dict.values()) or any(trunc_dict.values())
                obs_dict = next_obs_dict

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            'episode_reward': float(np.mean(episode_rewards)),
            'episode_length': float(np.mean(episode_lengths)),
            'n_episodes': n_episode,
            # Per-agent trajectories (used by _update_policy)
            'agent_obs':       {aid: np.array(agent_obs[aid],       dtype=np.float32) for aid in self.agent_ids},
            'agent_next_obs':  {aid: np.array(agent_next_obs[aid],  dtype=np.float32) for aid in self.agent_ids},
            'agent_actions':   {aid: np.array(agent_actions[aid],   dtype=np.int64)   for aid in self.agent_ids},
            'agent_log_probs': {aid: np.array(agent_log_probs[aid], dtype=np.float32) for aid in self.agent_ids},
            'agent_rewards':   {aid: np.array(agent_rewards[aid],   dtype=np.float32) for aid in self.agent_ids},
            'agent_term':      {aid: np.array(agent_term[aid],      dtype=np.bool_)   for aid in self.agent_ids},
            'agent_trunc':     {aid: np.array(agent_trunc[aid],     dtype=np.bool_)   for aid in self.agent_ids},
        }

    def _update_policy(self, collect_result: Dict, batch_size: int, repeat: int) -> Dict[str, Any]:
        """
        Run PPO update on each agent's policy using the collected trajectory data.

        Calls process_fn (GAE computation) then learn (PPO gradient steps) for
        every agent, aggregates losses and returns summary stats.
        """
        all_losses        = []
        all_actor_losses  = []
        all_critic_losses = []
        all_entropies     = []
        all_clip_fracs    = []

        for agent_id in self.agent_ids:
            policy = self.policy_manager.policies[agent_id]

            # Build Tianshou Batch for this agent
            batch = Batch(
                obs         = collect_result['agent_obs'][agent_id],        # (T, obs_dim)
                obs_next    = collect_result['agent_next_obs'][agent_id],   # (T, obs_dim)
                act         = collect_result['agent_actions'][agent_id],    # (T,)
                rew         = collect_result['agent_rewards'][agent_id],    # (T,)
                terminated  = collect_result['agent_term'][agent_id],       # (T,)
                truncated   = collect_result['agent_trunc'][agent_id],      # (T,)
                # logp_old will be added by process_fn; store here as initial value
                logp_old    = collect_result['agent_log_probs'][agent_id],  # (T,)
            )

            # Compute GAE advantages and value targets (process_fn also recomputes logp_old)
            batch = policy.process_fn(batch, buffer=None, indices=None)

            stats = policy.learn(batch, batch_size=batch_size, repeat=repeat)

            # Accumulate stats
            all_losses.extend(stats.get('loss', []))
            all_actor_losses.extend(stats.get('loss/actor', []))
            all_critic_losses.extend(stats.get('loss/critic', []))
            all_entropies.extend(stats.get('loss/entropy', []))
            all_clip_fracs.extend(stats.get('clip_frac', []))

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return {
            'loss':        _safe_mean(all_losses),
            'actor_loss':  _safe_mean(all_actor_losses),
            'critic_loss': _safe_mean(all_critic_losses),
            'entropy':     _safe_mean(all_entropies),
            'clip_frac':   _safe_mean(all_clip_fracs),
        }
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        # Eval episodes use a fixed seed block separated from training seeds.
        # Seed block: config.seed * 10_000 + 900_000 + episode_idx.
        # The same n_test_envs traffic realizations are used at every eval
        # checkpoint, so epoch-to-epoch eval comparison is not confounded by
        # env randomness.
        eval_seed_base = (
            self.config.seed * 10_000 + 900_000
            if self.config.training.use_fixed_episode_seeds
            else None
        )
        return evaluate_policy(
            self.policy_manager,
            self.test_env,
            n_episode=self.config.training.n_test_envs,
            device=self.device,
            eval_seed_base=eval_seed_base
        )
    
    def _log_epoch(self, epoch: int, collect_result: Dict, train_result: Dict, epoch_time: float):
        """Log epoch metrics."""
        wallclock_time = self._wallclock_offset + time.time() - self.training_start_time

        metrics = {
            'epoch': epoch,
            'train/episode_reward': collect_result['episode_reward'],
            'train/episode_length': collect_result['episode_length'],
            'train/epoch_time': epoch_time,
            'train/wallclock_time': wallclock_time,
        }
        
        # Add training losses if available
        for key, value in train_result.items():
            if isinstance(value, list) and len(value) > 0:
                metrics[f'train/{key}'] = np.mean(value)
            elif isinstance(value, (int, float)):
                metrics[f'train/{key}'] = value
        
        self.logger.log(metrics, step=epoch)
        self.metrics_tracker.update_episode(metrics)

        # Write CSV row for this training epoch
        self._csv_writer.writerow({
            "epoch":            epoch,
            "split":            "train",
            "wallclock_time_s": round(wallclock_time, 3),
            "episode_reward":   collect_result['episode_reward'],
            "episode_length":   collect_result['episode_length'],
            "loss":             train_result.get('loss', ''),
            "actor_loss":       train_result.get('actor_loss', ''),
            "critic_loss":      train_result.get('critic_loss', ''),
            "entropy":          train_result.get('entropy', ''),
            "clip_frac":        train_result.get('clip_frac', ''),
        })
        self._csv_file.flush()

        print(f"  Episode Reward: {collect_result['episode_reward']:.2f}")
        print(f"  Episode Length: {collect_result['episode_length']:.0f}")
        print(f"  Epoch Time:     {epoch_time:.2f}s")
        print(f"  Wallclock Time: {wallclock_time:.2f}s")
    
    def _log_evaluation(self, epoch: int, eval_result: Dict):
        """Log evaluation metrics."""
        wallclock_time = self._wallclock_offset + time.time() - self.training_start_time

        metrics = {f'eval/{k}': v for k, v in eval_result.items()}
        metrics['eval/wallclock_time'] = wallclock_time
        self.logger.log(metrics, step=epoch)

        # Write CSV row for this evaluation checkpoint
        self._csv_writer.writerow({
            "epoch":              epoch,
            "split":              "eval",
            "wallclock_time_s":   round(wallclock_time, 3),
            "episode_reward":     eval_result.get('mean_reward', ''),
            "mean_waiting_time":  eval_result.get('mean_waiting_time', ''),
            "std_waiting_time":   eval_result.get('std_waiting_time', ''),
            "mean_queue_length":  eval_result.get('mean_queue_length', ''),
            "std_queue_length":   eval_result.get('std_queue_length', ''),
        })
        self._csv_file.flush()

        print(f"  Eval Reward: {eval_result.get('mean_reward', 0):.2f}")
        print(f"  Eval Waiting Time: {eval_result.get('mean_waiting_time', 0):.2f}")
        print(f"  Eval Wallclock Time: {wallclock_time:.2f}s")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        
        save_checkpoint(
            policies=self.policy_manager.policies,
            optimizers=self._get_optimizers(),
            epoch=epoch + 1,
            metrics=self.metrics_tracker.get_all_stats(),
            save_dir=checkpoint_dir,
            config=self.config.to_dict(),
            extra_state=self._checkpoint_extra_state()
        )
