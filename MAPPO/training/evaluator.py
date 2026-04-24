"""Policy evaluation utilities."""

from typing import Dict, Any
import numpy as np
import torch

from MAPPO.envs import SumoTianshouEnv
from MAPPO.agents import MultiAgentPolicyManager
from MAPPO.envs.env_utils import compute_traffic_metrics


def evaluate_policy(
    policy_manager: MultiAgentPolicyManager,
    env: SumoTianshouEnv,
    n_episode: int = 10,
    device: str = "cpu",
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate a policy on an environment.
    
    Args:
        policy_manager: Multi-agent policy manager
        env: SUMO environment
        n_episode: Number of episodes to evaluate
        device: Device to run evaluation on
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_waiting_times = []
    episode_queue_lengths = []
    
    for ep in range(n_episode):
        obs_dict, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        episode_traffic_metrics = []
        
        while not done:
            # Get actions from policies
            actions = {}
            for agent_id in env.agents:
                obs_tensor = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0).to(device)
                policy = policy_manager.policies[agent_id]
                with torch.no_grad():
                    action = policy.actor.get_action(obs_tensor, deterministic=deterministic)
                actions[agent_id] = action.item()
            
            # Step environment
            next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
            
            # Accumulate metrics
            episode_reward += sum(reward_dict.values()) / len(reward_dict)
            episode_length += 1
            
            # Compute traffic metrics
            traffic_metrics = compute_traffic_metrics(info_dict)
            episode_traffic_metrics.append(traffic_metrics)
            
            # Check if done
            done = any(term_dict.values()) or any(trunc_dict.values())
            obs_dict = next_obs_dict
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Aggregate traffic metrics for episode
        if episode_traffic_metrics:
            avg_waiting = np.mean([m.get('avg_waiting_time', 0) for m in episode_traffic_metrics])
            avg_queue = np.mean([m.get('avg_queue_length', 0) for m in episode_traffic_metrics])
            episode_waiting_times.append(avg_waiting)
            episode_queue_lengths.append(avg_queue)
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }
    
    if episode_waiting_times:
        results['mean_waiting_time'] = np.mean(episode_waiting_times)
        results['std_waiting_time'] = np.std(episode_waiting_times)
    
    if episode_queue_lengths:
        results['mean_queue_length'] = np.mean(episode_queue_lengths)
        results['std_queue_length'] = np.std(episode_queue_lengths)
    
    return results


def evaluate_baseline(
    env: SumoTianshouEnv,
    n_episode: int = 10,
    policy_type: str = "random"
) -> Dict[str, float]:
    """
    Evaluate a baseline policy.
    
    Args:
        env: SUMO environment
        n_episode: Number of episodes
        policy_type: Type of baseline ("random" or "fixed")
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_waiting_times = []
    
    for ep in range(n_episode):
        obs_dict, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        episode_traffic_metrics = []
        
        while not done:
            # Get baseline actions
            if policy_type == "random":
                actions = {agent: env.action_space.sample() for agent in env.agents}
            elif policy_type == "fixed":
                actions = {agent: 0 for agent in env.agents}  # Always choose first phase
            else:
                raise ValueError(f"Unknown baseline type: {policy_type}")
            
            # Step environment
            next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
            
            # Accumulate metrics
            episode_reward += sum(reward_dict.values()) / len(reward_dict)
            episode_length += 1
            
            # Compute traffic metrics
            traffic_metrics = compute_traffic_metrics(info_dict)
            episode_traffic_metrics.append(traffic_metrics)
            
            # Check if done
            done = any(term_dict.values()) or any(trunc_dict.values())
            obs_dict = next_obs_dict
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Aggregate traffic metrics
        if episode_traffic_metrics:
            avg_waiting = np.mean([m.get('avg_waiting_time', 0) for m in episode_traffic_metrics])
            episode_waiting_times.append(avg_waiting)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'policy_type': policy_type
    }
    
    if episode_waiting_times:
        results['mean_waiting_time'] = np.mean(episode_waiting_times)
    
    return results
