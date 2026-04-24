"""
SUMO-RL Environment Wrapper for Tianshou Compatibility

This module provides a wrapper that adapts SUMO-RL's parallel environment
to work with Tianshou's multi-agent policy framework.
"""

import os

# Auto-detect SUMO_HOME and PROJ_DATA if not set, so collaborators don't need
# to configure environment variables manually as long as the sumo Python package
# is installed (pip install sumo or sumo-rl installs it automatically).
def _configure_sumo_env() -> None:
    """Set SUMO_HOME and PROJ_DATA from the installed sumo package if missing."""
    if not os.environ.get("SUMO_HOME"):
        try:
            import sumo as _sumo_pkg
            os.environ["SUMO_HOME"] = os.path.dirname(_sumo_pkg.__file__)
        except ImportError:
            pass  # sumo package not available; SUMO_HOME must be set manually

    if not os.environ.get("PROJ_DATA"):
        sumo_home = os.environ.get("SUMO_HOME", "")
        candidate = os.path.join(sumo_home, "data", "proj")
        if os.path.isdir(candidate):
            os.environ["PROJ_DATA"] = candidate

_configure_sumo_env()

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sumo_rl


class SumoTianshouEnv:
    """
    Wrapper for SUMO-RL parallel environments to make them compatible with Tianshou.
    
    This wrapper:
    - Converts SUMO-RL observations to the format expected by MAPPO
    - Handles agent_id tracking for multi-agent scenarios
    - Manages episode termination and truncation
    - Supports both GUI and headless modes
    
    Args:
        net_file: Path to SUMO network file (.net.xml)
        route_file: Path to SUMO route file (.rou.xml)
        use_gui: Whether to use SUMO GUI (default: False)
        num_seconds: Simulation duration in seconds (default: 3600)
        delta_time: Seconds between actions (default: 5)
        yellow_time: Yellow phase duration in seconds (default: 2)
        min_green: Minimum green phase duration (default: 5)
        max_green: Maximum green phase duration (default: 50)
        **kwargs: Additional arguments passed to sumo_rl.parallel_env
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        **kwargs
    ):
        # Create SUMO-RL environment
        # add_system_info=True  → populates 'system_mean_waiting_time' etc. in each agent's info
        # add_per_agent_info=True → populates '{agent_id}_accumulated_waiting_time' etc.
        self.sumo_env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            add_system_info=True,
            add_per_agent_info=True,
            **kwargs
        )
        
        # Store configuration
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        
        # Initialize environment to get agent information
        self.sumo_env.reset()
        
        # Store agent information (now it's just a regular attribute, not a property)
        self.agents = self.sumo_env.agents
        self.possible_agents = self.sumo_env.possible_agents
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}
        
        # Get spaces from first agent
        if len(self.agents) > 0:
            first_agent = self.agents[0]
            self.observation_space = self.sumo_env.observation_space(first_agent)
            self.action_space = self.sumo_env.action_space(first_agent)
        else:
            self.observation_space = None
            self.action_space = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observations: Dict of observations for each agent
            infos: Dict of info for each agent
        """
        # Reset SUMO environment
        if seed is not None:
            observations, infos = self.sumo_env.reset(seed=seed, options=options)
        else:
            observations, infos = self.sumo_env.reset()
        
        # Update agent lists
        self.agents = self.sumo_env.agents
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}
        
        return observations, infos
    
    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Step the environment with actions for all agents.
        
        Args:
            actions: Dict mapping agent_id to action
            
        Returns:
            observations: Next observations for each agent
            rewards: Rewards for each agent
            terminations: Termination flags for each agent
            truncations: Truncation flags for each agent
            infos: Additional info for each agent
        """
        observations, rewards, terminations, truncations, infos = self.sumo_env.step(actions)
        
        return observations, rewards, terminations, truncations, infos
    
    def close(self):
        """Close the environment."""
        if hasattr(self.sumo_env, 'close'):
            self.sumo_env.close()
    
    def render(self):
        """Render the environment (only works if use_gui=True)."""
        if hasattr(self.sumo_env, 'render'):
            return self.sumo_env.render()
        return None
    
    def observation_space_dict(self) -> Dict[str, spaces.Space]:
        """Get observation spaces for all agents."""
        return {agent: self.sumo_env.observation_space(agent) for agent in self.agents}
    
    def action_space_dict(self) -> Dict[str, spaces.Space]:
        """Get action spaces for all agents."""
        return {agent: self.sumo_env.action_space(agent) for agent in self.agents}
    
    def seed(self, seed: int):
        """Set random seed for the environment."""
        # SUMO-RL handles seeding through reset
        pass


def create_sumo_env(
    net_file: str,
    route_file: str,
    use_gui: bool = False,
    num_seconds: int = 3600,
    **kwargs
) -> SumoTianshouEnv:
    """
    Convenience function to create a SUMO environment.
    
    Args:
        net_file: Path to SUMO network file
        route_file: Path to SUMO route file
        use_gui: Whether to use SUMO GUI
        num_seconds: Simulation duration
        **kwargs: Additional arguments
        
    Returns:
        Wrapped SUMO environment compatible with Tianshou
    """
    return SumoTianshouEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        **kwargs
    )
