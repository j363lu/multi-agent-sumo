#!/usr/bin/env python3
"""
Evaluation script for trained MAPPO policies.

Usage:
    python scripts/evaluate.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt
    python scripts/evaluate.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt --use-gui
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from mappo_traffic.config import ExperimentConfig
from mappo_traffic.envs import SumoTianshouEnv
from mappo_traffic.networks import ActorNetwork, CentralizedCritic
from mappo_traffic.agents import MAPPOPolicy, MultiAgentPolicyManager
from mappo_traffic.training.evaluator import evaluate_policy, evaluate_baseline
from mappo_traffic.utils import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO policy")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    
    parser.add_argument(
        "--n-episode",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Use SUMO GUI for visualization"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use"
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate random baseline"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load config from checkpoint directory
    checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
    else:
        print("Warning: Config not found, using default")
        from mappo_traffic.config.default_configs import get_default_config
        config = get_default_config()
    
    # Override GUI setting
    if args.use_gui:
        config.sumo.use_gui = True
    
    # Create environment
    print("Creating evaluation environment...")
    env = SumoTianshouEnv(
        net_file=config.sumo.net_file,
        route_file=config.sumo.route_file,
        use_gui=config.sumo.use_gui,
        num_seconds=config.sumo.num_seconds,
        delta_time=config.sumo.delta_time
    )
    
    # Get environment info
    agent_ids = env.agents
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Agents: {len(agent_ids)}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    
    # Create networks
    device = torch.device(args.device)
    
    global_obs_dim = obs_dim * len(agent_ids)
    critic = CentralizedCritic(
        global_obs_dim=global_obs_dim,
        hidden_dims=config.network.critic_hidden
    ).to(device)
    
    policies = {}
    for agent_id in agent_ids:
        actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.network.actor_hidden
        ).to(device)
        
        policy = MAPPOPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters())
        )
        policies[agent_id] = policy
    
    # Create policy manager
    policy_manager = MultiAgentPolicyManager(
        policies=list(policies.values()),
        critic=critic,
        agent_ids=agent_ids
    )
    
    # Load checkpoint
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        policies=policies,
        critic=critic,
        device=args.device
    )
    
    # Evaluate policy
    print(f"\nEvaluating policy for {args.n_episode} episodes...")
    results = evaluate_policy(
        policy_manager=policy_manager,
        env=env,
        n_episode=args.n_episode,
        device=args.device,
        deterministic=True
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key:30s}: {value:.4f}")
    print("=" * 60)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        print("\nEvaluating random baseline...")
        baseline_results = evaluate_baseline(
            env=env,
            n_episode=args.n_episode,
            policy_type="random"
        )
        
        print("\n" + "=" * 60)
        print("Baseline Results (Random Policy)")
        print("=" * 60)
        for key, value in baseline_results.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        print("=" * 60)
        
        # Comparison
        if 'mean_reward' in results and 'mean_reward' in baseline_results:
            improvement = (results['mean_reward'] - baseline_results['mean_reward']) / abs(baseline_results['mean_reward']) * 100
            print(f"\nReward improvement: {improvement:+.2f}%")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
