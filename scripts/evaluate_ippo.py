#!/usr/bin/env python3
"""
Evaluation script for trained IPPO policies.

Usage:
    python scripts/evaluate_ippo.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt
    python scripts/evaluate_ippo.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt --use-gui
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Auto-detect SUMO_HOME / PROJ_DATA from the installed sumo package so
# collaborators don't need to set shell environment variables manually.
def _configure_sumo_env() -> None:
    if not os.environ.get("SUMO_HOME"):
        try:
            import sumo as _sumo_pkg
            os.environ["SUMO_HOME"] = os.path.dirname(_sumo_pkg.__file__)
        except ImportError:
            pass
    if not os.environ.get("PROJ_DATA"):
        candidate = os.path.join(os.environ.get("SUMO_HOME", ""), "data", "proj")
        if os.path.isdir(candidate):
            os.environ["PROJ_DATA"] = candidate

_configure_sumo_env()

import torch
from IPPO.config import ExperimentConfig
from IPPO.envs import SumoTianshouEnv
from IPPO.networks import ActorNetwork, LocalCritic
from IPPO.agents import IPPOPolicy, MultiAgentPolicyManager
from IPPO.training.evaluator import evaluate_policy, evaluate_baseline
from IPPO.utils import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO policy")
    
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
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use"
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate random baseline"
    )
    
    return parser.parse_args()


def _load_config(checkpoint_path: str) -> ExperimentConfig:
    """Load the training config saved with a checkpoint."""
    checkpoint_file_dir = os.path.dirname(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_file_dir)
    config_candidates = [
        os.path.join(checkpoint_file_dir, "config.json"),
        os.path.join(run_dir, "config.json"),
    ]

    for config_path in config_candidates:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return ExperimentConfig.from_dict(json.load(f))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") is not None:
        return ExperimentConfig.from_dict(checkpoint["config"])

    print("Warning: Config not found, using default")
    from IPPO.config.default_configs import get_default_config
    return get_default_config()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    
    config = _load_config(args.checkpoint)
    
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
        begin_time=config.sumo.begin_time,
        delta_time=config.sumo.delta_time,
        yellow_time=config.sumo.yellow_time,
        min_green=config.sumo.min_green,
        max_green=config.sumo.max_green
    )
    
    # Get environment info
    agent_ids = env.agents
    action_dim = env.action_space.n
    
    print(f"Agents: {len(agent_ids)}")
    print(f"Action dim: {action_dim}")
    
    # Create networks
    device = torch.device(args.device)
    
    policies = {}
    for agent_id in agent_ids:
        obs_dim = env.sumo_env.observation_space(agent_id).shape[0]
        print(f"Observation dim [{agent_id}]: {obs_dim}")

        actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.network.actor_hidden
        ).to(device)

        critic = LocalCritic(
            obs_dim=obs_dim,
            hidden_dims=config.network.critic_hidden
        ).to(device)
        
        policy = IPPOPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters())
        )
        policies[agent_id] = policy
    
    # Create policy manager
    policy_manager = MultiAgentPolicyManager(
        policies=list(policies.values()),
        agent_ids=agent_ids
    )
    
    # Load checkpoint
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        policies=policies,
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
