#!/usr/bin/env python3
"""
Evaluation script for trained IPPO policies.

Usage:
    python scripts/evaluate_ippo.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt
    python scripts/evaluate_ippo.py --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt --use-gui
"""

import argparse
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


def _infer_first_layer_input_dim(state_dict, preferred_key="feature_extractor.0.weight"):
    """Infer network input dimension from a checkpoint state_dict."""
    if preferred_key in state_dict:
        return state_dict[preferred_key].shape[1]

    for key, value in state_dict.items():
        if key.endswith(".weight") and getattr(value, "ndim", 0) == 2:
            return value.shape[1]

    raise KeyError("Could not infer input dimension from checkpoint state_dict")


def _infer_last_layer_output_dim(state_dict, fallback):
    """Infer actor action dimension from the last Linear layer, falling back to env action dim."""
    for key, value in reversed(list(state_dict.items())):
        if key.endswith(".weight") and getattr(value, "ndim", 0) == 2:
            return value.shape[0]
    return fallback


def _get_agent_critic_state_dict(checkpoint_data, agent_id):
    """Return the local critic state_dict for an IPPO agent checkpoint."""
    agent_checkpoint = checkpoint_data["policies"][agent_id]

    if "critic_state_dict" in agent_checkpoint:
        return agent_checkpoint["critic_state_dict"]

    # Fallback for alternate checkpoint layouts.
    if "critics" in checkpoint_data and agent_id in checkpoint_data["critics"]:
        critic_entry = checkpoint_data["critics"][agent_id]
        if isinstance(critic_entry, dict) and "critic_state_dict" in critic_entry:
            return critic_entry["critic_state_dict"]
        return critic_entry

    raise KeyError(
        f"Could not find local critic_state_dict for agent {agent_id}. "
        "IPPO evaluation needs one local critic per agent."
    )


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load checkpoint first so evaluation can rebuild networks with the exact saved sizes.
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_data = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    # Load config from checkpoint directory, or from checkpoint itself.
    checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, "config.json")

    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
    elif "config" in checkpoint_data:
        print("Config file not found; using config stored inside checkpoint")
        config = ExperimentConfig.from_dict(checkpoint_data["config"])
    else:
        print("Warning: Config not found, using default")
        from IPPO.config.default_configs import get_default_config
        config = get_default_config()

    # Override GUI setting.
    if args.use_gui:
        config.sumo.use_gui = True

    # Create environment with the same SUMO timing settings used during training.
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
        max_green=config.sumo.max_green,
    )

    # Get environment info.
    agent_ids = env.agents
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Agents: {len(agent_ids)}")
    print(f"Default env observation dim: {obs_dim}")
    print(f"Default env action dim: {action_dim}")

    device = torch.device(args.device)

    if "policies" not in checkpoint_data:
        raise KeyError("Checkpoint has no 'policies' key; cannot rebuild IPPO policies.")

    checkpoint_agent_ids = list(checkpoint_data["policies"].keys())
    print("Env agent IDs:", agent_ids)
    print("Checkpoint policy IDs:", checkpoint_agent_ids)

    missing_agents = [agent_id for agent_id in agent_ids if agent_id not in checkpoint_data["policies"]]
    if missing_agents:
        raise KeyError(
            f"These env agents are missing from the checkpoint: {missing_agents}. "
            "Make sure this checkpoint was trained on the same scenario."
        )

    # IPPO uses one local actor and one local critic per agent.
    policies = {}

    for agent_id in agent_ids:
        agent_checkpoint = checkpoint_data["policies"][agent_id]
        actor_state_dict = agent_checkpoint["actor_state_dict"]
        critic_state_dict = _get_agent_critic_state_dict(checkpoint_data, agent_id)

        actor_obs_dim = _infer_first_layer_input_dim(actor_state_dict)
        critic_obs_dim = _infer_first_layer_input_dim(critic_state_dict)
        actor_action_dim = _infer_last_layer_output_dim(actor_state_dict, fallback=action_dim)

        print(
            f"Creating IPPO networks for {agent_id}: "
            f"actor_obs_dim={actor_obs_dim}, "
            f"critic_obs_dim={critic_obs_dim}, "
            f"action_dim={actor_action_dim}"
        )

        actor = ActorNetwork(
            obs_dim=actor_obs_dim,
            action_dim=actor_action_dim,
            hidden_dims=config.network.actor_hidden
        ).to(device)

        critic = LocalCritic(
            obs_dim=critic_obs_dim,
            hidden_dims=config.network.critic_hidden
        ).to(device)

        # Load directly from checkpoint. Optimizer state is not needed for evaluation.
        actor.load_state_dict(actor_state_dict)
        critic.load_state_dict(critic_state_dict)
        actor.eval()
        critic.eval()

        policy = IPPOPolicy(
            actor=actor,
            critic=critic,
            optim_actor=torch.optim.Adam(actor.parameters()),
            optim_critic=torch.optim.Adam(critic.parameters())
        )
        policies[agent_id] = policy

    # Create policy manager. IPPO managers usually do not need a centralized critic.
    try:
        policy_manager = MultiAgentPolicyManager(
            policies=list(policies.values()),
            agent_ids=agent_ids
        )
    except TypeError:
        # Fallback in case your manager signature still has a critic keyword.
        policy_manager = MultiAgentPolicyManager(
            policies=list(policies.values()),
            critic=None,
            agent_ids=agent_ids
        )

    # Evaluate policy.
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

    # Compare with baseline if requested.
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

        # Comparison.
        if 'mean_reward' in results and 'mean_reward' in baseline_results:
            improvement = (
                (results['mean_reward'] - baseline_results['mean_reward'])
                / abs(baseline_results['mean_reward'])
                * 100
            )
            print(f"\nReward improvement: {improvement:+.2f}%")

    # Cleanup.
    env.close()


if __name__ == "__main__":
    main()
