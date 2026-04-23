"""
MAPPO Trainer for SUMO traffic control.

Main training loop implementation.
"""

import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.optim import Adam
from tianshou.data import Batch

from mappo_traffic.config import ExperimentConfig
from mappo_traffic.envs import SumoTianshouEnv
from mappo_traffic.networks import ActorNetwork, CentralizedCritic
from mappo_traffic.agents import MAPPOPolicy, MultiAgentPolicyManager
from mappo_traffic.utils import WandBLogger, save_checkpoint, MetricsTracker
from mappo_traffic.training.evaluator import evaluate_policy


class MAPPOTrainer:
    """
    MAPPO Trainer for multi-agent traffic signal control.
    
    Args:
        config: Experiment configuration
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Set device
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
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
        
        # Create networks and policies
        print("Creating networks and policies...")
        self.policy_manager = self._create_policy_manager()
        
        # Setup logging
        exp_name = config.logging.experiment_name or f"mappo_{int(time.time())}"
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
        self.total_steps = 0
        
        print("Trainer initialized successfully!")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _create_env(self, sumo_config, num_envs: int = 1) -> SumoTianshouEnv:
        """Create SUMO environment."""
        # For simplicity, we create a single environment
        # In practice, you might want vectorized environments
        env = SumoTianshouEnv(
            net_file=sumo_config.net_file,
            route_file=sumo_config.route_file,
            use_gui=sumo_config.use_gui,
            num_seconds=sumo_config.num_seconds,
            delta_time=sumo_config.delta_time,
            yellow_time=sumo_config.yellow_time,
            min_green=sumo_config.min_green,
            max_green=sumo_config.max_green
        )
        return env
    
    def _create_policy_manager(self) -> MultiAgentPolicyManager:
        """Create MAPPO policy manager with networks and optimizers."""
        net_config = self.config.network
        mappo_config = self.config.mappo
        
        # Get observation dimensions for each agent (may be different!)
        agent_obs_dims = {}
        total_obs_dim = 0
        for agent_id in self.agent_ids:
            obs_space = self.train_env.sumo_env.observation_space(agent_id)
            agent_obs_dims[agent_id] = obs_space.shape[0]
            total_obs_dim += obs_space.shape[0]
        
        print(f"Agent observation dimensions: {agent_obs_dims}")
        
        # Create centralized critic with total observation dimension
        critic = CentralizedCritic(
            global_obs_dim=total_obs_dim,
            hidden_dims=net_config.critic_hidden,
            activation=net_config.activation,
            use_orthogonal_init=net_config.use_orthogonal_init
        ).to(self.device)
        
        # Create optimizer for critic
        critic_optimizer = Adam(critic.parameters(), lr=mappo_config.lr_critic)
        
        # Create actor networks and policies for each agent (with agent-specific obs dims!)
        policies = []
        for agent_id in self.agent_ids:
            # Create actor with agent-specific observation dimension
            actor = ActorNetwork(
                obs_dim=agent_obs_dims[agent_id],  # Agent-specific!
                action_dim=self.action_dim,
                hidden_dims=net_config.actor_hidden,
                activation=net_config.activation,
                use_orthogonal_init=net_config.use_orthogonal_init
            ).to(self.device)
            
            # Create optimizer for actor
            actor_optimizer = Adam(actor.parameters(), lr=mappo_config.lr_actor)
            
            # Create MAPPO policy
            policy = MAPPOPolicy(
                actor=actor,
                critic=critic,  # Shared critic
                optim_actor=actor_optimizer,
                optim_critic=critic_optimizer,
                gamma=mappo_config.gamma,
                gae_lambda=mappo_config.gae_lambda,
                eps_clip=mappo_config.eps_clip,
                value_clip=mappo_config.value_clip,
                dual_clip=mappo_config.dual_clip,
                advantage_normalization=mappo_config.advantage_normalization,
                vf_coef=mappo_config.vf_coef,
                ent_coef=mappo_config.ent_coef,
                max_grad_norm=mappo_config.max_grad_norm,
                reward_normalization=mappo_config.reward_normalization
            )
            
            policies.append(policy)
        
        # Create policy manager
        policy_manager = MultiAgentPolicyManager(
            policies=policies,
            critic=critic,
            agent_ids=self.agent_ids
        )
        
        return policy_manager
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting MAPPO Training")
        print("=" * 60)
        
        config = self.config.training
        
        for epoch in range(config.max_epoch):
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
            self._log_epoch(epoch, collect_result, train_result, epoch_time)
            
            # Evaluation
            if (epoch + 1) % config.test_interval == 0:
                eval_result = self._evaluate()
                self._log_evaluation(epoch, eval_result)
            
            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(self.metrics_tracker.summary())
        
        self.logger.finish()
    
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
        # Global critic inputs – concatenation of all agents' obs in fixed order
        global_obs:      List = []
        global_obs_next: List = []

        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episode):
            obs_dict, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Build global critic input from current obs
                global_inp = np.concatenate(
                    [obs_dict[aid] for aid in self.agent_ids], axis=-1
                )

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

                # Build global critic input from next obs
                global_inp_next = np.concatenate(
                    [next_obs_dict[aid] for aid in self.agent_ids], axis=-1
                )

                # Store transition for every agent
                for aid in self.agent_ids:
                    agent_obs[aid].append(obs_dict[aid])
                    agent_next_obs[aid].append(next_obs_dict[aid])
                    agent_actions[aid].append(actions[aid])
                    agent_log_probs[aid].append(log_probs[aid])
                    agent_rewards[aid].append(reward_dict.get(aid, 0.0))
                    agent_term[aid].append(term_dict.get(aid, False))
                    agent_trunc[aid].append(trunc_dict.get(aid, False))

                global_obs.append(global_inp)
                global_obs_next.append(global_inp_next)

                episode_reward += sum(reward_dict.values()) / max(len(reward_dict), 1)
                episode_length += 1

                done = any(term_dict.values()) or any(trunc_dict.values())
                obs_dict = next_obs_dict

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Convert global obs lists to arrays once
        global_obs_arr      = np.array(global_obs,      dtype=np.float32)
        global_obs_next_arr = np.array(global_obs_next, dtype=np.float32)

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
            'global_obs':      global_obs_arr,
            'global_obs_next': global_obs_next_arr,
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

        global_obs      = collect_result['global_obs']       # (T, global_obs_dim)
        global_obs_next = collect_result['global_obs_next']  # (T, global_obs_dim)

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
                critic_inp      = global_obs,       # (T, global_obs_dim)
                critic_inp_next = global_obs_next,  # (T, global_obs_dim)
                # logp_old will be added by process_fn; store here as initial value
                logp_old    = collect_result['agent_log_probs'][agent_id],  # (T,)
            )

            # Compute GAE advantages and value targets (process_fn also recomputes logp_old)
            batch = policy.process_fn(batch, buffer=None, indices=None)

            # Run PPO gradient updates (repeat passes over mini-batches)
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
        return evaluate_policy(
            self.policy_manager,
            self.test_env,
            n_episode=self.config.training.n_test_envs,
            device=self.device
        )
    
    def _log_epoch(self, epoch: int, collect_result: Dict, train_result: Dict, epoch_time: float):
        """Log epoch metrics."""
        metrics = {
            'epoch': epoch,
            'train/episode_reward': collect_result['episode_reward'],
            'train/episode_length': collect_result['episode_length'],
            'train/epoch_time': epoch_time,
        }
        
        # Add training losses if available
        for key, value in train_result.items():
            if isinstance(value, list) and len(value) > 0:
                metrics[f'train/{key}'] = np.mean(value)
            elif isinstance(value, (int, float)):
                metrics[f'train/{key}'] = value
        
        self.logger.log(metrics, step=epoch)
        self.metrics_tracker.update_episode(metrics)
        
        print(f"  Episode Reward: {collect_result['episode_reward']:.2f}")
        print(f"  Episode Length: {collect_result['episode_length']:.0f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
    
    def _log_evaluation(self, epoch: int, eval_result: Dict):
        """Log evaluation metrics."""
        metrics = {f'eval/{k}': v for k, v in eval_result.items()}
        self.logger.log(metrics, step=epoch)
        
        print(f"  Eval Reward: {eval_result.get('mean_reward', 0):.2f}")
        print(f"  Eval Waiting Time: {eval_result.get('mean_waiting_time', 0):.2f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        
        # Get all optimizers
        optimizers = {
            'critic': list(self.policy_manager.policies.values())[0].optim_critic
        }
        for i, (agent_id, policy) in enumerate(self.policy_manager.policies.items()):
            optimizers[f'actor_{agent_id}'] = policy.optim_actor
        
        save_checkpoint(
            policies=self.policy_manager.policies,
            critic=self.policy_manager.critic,
            optimizers=optimizers,
            epoch=epoch,
            metrics=self.metrics_tracker.get_all_stats(),
            save_dir=checkpoint_dir,
            config=self.config.to_dict()
        )
