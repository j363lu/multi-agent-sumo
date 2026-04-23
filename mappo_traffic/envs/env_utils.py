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
    
    Args:
        infos: Info dictionaries from all agents
        
    Returns:
        Dictionary of traffic metrics
    """
    metrics = {
        'avg_waiting_time': 0.0,
        'avg_queue_length': 0.0,
        'total_stopped': 0,
        'num_agents': len(infos)
    }
    
    if len(infos) == 0:
        return metrics
    
    waiting_times = []
    queue_lengths = []
    stopped_vehicles = []
    
    for agent_id, info in infos.items():
        if 'waiting_time' in info:
            waiting_times.append(info['waiting_time'])
        if 'queue' in info:
            queue_lengths.append(info['queue'])
        if 'stopped' in info:
            stopped_vehicles.append(info['stopped'])
    
    if waiting_times:
        metrics['avg_waiting_time'] = np.mean(waiting_times)
    if queue_lengths:
        metrics['avg_queue_length'] = np.mean(queue_lengths)
    if stopped_vehicles:
        metrics['total_stopped'] = np.sum(stopped_vehicles)
    
    return metrics
