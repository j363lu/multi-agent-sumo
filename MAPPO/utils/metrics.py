"""Traffic metrics computation and tracking."""

from typing import Dict, List, Any
import numpy as np
from collections import deque


def compute_traffic_metrics(infos: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute traffic metrics from environment info dictionaries.
    
    Args:
        infos: Dictionary mapping agent_id to info dict
        
    Returns:
        Dictionary of computed metrics
    """
    if len(infos) == 0:
        return {}
    
    metrics = {}
    
    # Aggregate metrics across agents
    waiting_times = []
    queue_lengths = []
    speeds = []
    
    for agent_id, info in infos.items():
        # Extract metrics (keys may vary by environment)
        if 'waiting_time' in info:
            waiting_times.append(info['waiting_time'])
        if 'queue' in info:
            queue_lengths.append(info['queue'])
        if 'speed' in info:
            speeds.append(info['speed'])
    
    # Compute aggregate statistics
    if waiting_times:
        metrics['avg_waiting_time'] = np.mean(waiting_times)
        metrics['max_waiting_time'] = np.max(waiting_times)
        metrics['total_waiting_time'] = np.sum(waiting_times)
    
    if queue_lengths:
        metrics['avg_queue_length'] = np.mean(queue_lengths)
        metrics['max_queue_length'] = np.max(queue_lengths)
        metrics['total_queue_length'] = np.sum(queue_lengths)
    
    if speeds:
        metrics['avg_speed'] = np.mean(speeds)
    
    metrics['num_agents'] = len(infos)
    
    return metrics


class MetricsTracker:
    """
    Track and aggregate metrics over time.
    
    Args:
        window_size: Size of rolling window for statistics
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.episode_metrics: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update tracker with new metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
            
            if isinstance(value, (list, np.ndarray)):
                # For list/array values, store all elements
                self.metrics[key].extend(value)
            else:
                self.metrics[key].append(value)
    
    def update_episode(self, metrics: Dict[str, float]):
        """Update episode-level metrics."""
        for key, value in metrics.items():
            if key not in self.episode_metrics:
                self.episode_metrics[key] = []
            self.episode_metrics[key].append(value)
    
    def get_stats(self, key: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            key: Metric key
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return {}
        
        values = np.array(self.metrics[key])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def get_recent_mean(self, key: str, n: int = 10) -> float:
        """Get mean of recent n values."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        
        recent = list(self.metrics[key])[-n:]
        return np.mean(recent)
    
    def get_episode_stats(self, key: str) -> Dict[str, float]:
        """Get statistics for episode-level metric."""
        if key not in self.episode_metrics or len(self.episode_metrics[key]) == 0:
            return {}
        
        values = np.array(self.episode_metrics[key])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked metrics."""
        stats = {}
        for key in self.metrics.keys():
            stats[key] = self.get_stats(key)
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.episode_metrics.clear()
    
    def summary(self) -> str:
        """Get a summary string of current statistics."""
        lines = ["Metrics Summary:"]
        lines.append("-" * 50)
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                stats = self.get_stats(key)
                lines.append(
                    f"{key:30s}: "
                    f"mean={stats['mean']:8.4f}, "
                    f"std={stats['std']:8.4f}"
                )
        
        if self.episode_metrics:
            lines.append("\nEpisode Metrics:")
            lines.append("-" * 50)
            for key, values in self.episode_metrics.items():
                if len(values) > 0:
                    stats = self.get_episode_stats(key)
                    lines.append(
                        f"{key:30s}: "
                        f"mean={stats['mean']:8.4f}, "
                        f"std={stats['std']:8.4f}"
                    )
        
        return "\n".join(lines)
