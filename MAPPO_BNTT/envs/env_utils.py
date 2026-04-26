"""Utility functions for environment handling."""

import numpy as np
from typing import Dict, List, Any
from gymnasium import spaces


def normalize_observation(obs: np.ndarray, obs_space: spaces.Space) -> np.ndarray:
    """
    Normalize observations to [0, 1] or [-1, 1] range.
    
    Args:
        obs: Raw observation
        obs_space: Observation space
        
    Returns:
        Normalized observation
    """
    if isinstance(obs_space, spaces.Box):
        # Normalize Box spaces
        low = obs_space.low
        high = obs_space.high
        
        # Handle infinite bounds
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)
        
        # Avoid division by zero
        range_val = high - low
        range_val = np.where(range_val == 0, 1.0, range_val)
        
        normalized = (obs - low) / range_val
        return normalized
    
    return obs


def flatten_dict_observations(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten a dictionary of observations into a single array.
    
    Args:
        obs_dict: Dictionary of observations
        
    Returns:
        Flattened observation array
    """
    obs_list = []
    for key in sorted(obs_dict.keys()):
        obs = obs_dict[key]
        if isinstance(obs, np.ndarray):
            obs_list.append(obs.flatten())
        else:
            obs_list.append(np.array([obs]).flatten())
    
    return np.concatenate(obs_list)


def get_global_state(observations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate all agent observations into a global state.
    
    Args:
        observations: Dict mapping agent_id to observation
        
    Returns:
        Concatenated global state
    """
    # Sort by agent ID for consistency
    sorted_agents = sorted(observations.keys())
    obs_list = [observations[agent] for agent in sorted_agents]
    
    return np.concatenate(obs_list, axis=-1)


def compute_traffic_metrics(infos: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregate traffic metrics from environment info.

    SUMO-RL (with add_system_info=True, add_per_agent_info=True) populates
    each agent's info dict with keys such as:
      - 'system_mean_waiting_time'            (float, same for all agents)
      - 'system_total_waiting_time'           (float)
      - 'system_mean_speed'                   (float)
      - '{agent_id}_stopped'                  (int)
      - '{agent_id}_accumulated_waiting_time' (float)
      - '{agent_id}_average_speed'            (float)

    Args:
        infos: Info dictionaries from all agents  {agent_id: {key: value}}

    Returns:
        Dictionary of traffic metrics with keys:
            'avg_waiting_time', 'avg_queue_length', 'total_stopped', 'num_agents'
    """
    metrics = {
        'avg_waiting_time': 0.0,
        'avg_queue_length': 0.0,
        'total_stopped': 0,
        'num_agents': len(infos)
    }

    if len(infos) == 0:
        return metrics

    # System-level mean waiting time — present in every agent's info dict,
    # same value for all; just read it once from the first entry.
    first_info = next(iter(infos.values()))
    if 'system_mean_waiting_time' in first_info:
        metrics['avg_waiting_time'] = float(first_info['system_mean_waiting_time'])

    # Per-agent stopped vehicle counts (queue proxy)
    stopped_list = []
    for agent_id, info in infos.items():
        stopped_key = f"{agent_id}_stopped"
        if stopped_key in info:
            stopped_list.append(int(info[stopped_key]))

    if stopped_list:
        metrics['total_stopped'] = int(np.sum(stopped_list))
        metrics['avg_queue_length'] = float(np.mean(stopped_list))

    return metrics
