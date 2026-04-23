"""
Multi-Agent Policy Manager for MAPPO.

Coordinates multiple agent policies with a centralized critic.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Batch, ReplayBuffer

from mappo_traffic.agents.mappo_policy import MAPPOPolicy
from mappo_traffic.networks import CentralizedCritic


class MultiAgentPolicyManager(nn.Module):
    """
    Multi-agent policy manager for MAPPO.
    
    Manages multiple MAPPO policies (one per agent) and a shared centralized critic.
    Dispatches observations to policies and aggregates global state for the critic.
    
    Args:
        policies: List of MAPPO policies (one per agent)
        critic: Shared centralized critic
        agent_ids: List of agent IDs
    """
    
    def __init__(
        self,
        policies: List[MAPPOPolicy],
        critic: CentralizedCritic,
        agent_ids: List[str],
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        
        assert len(policies) == len(agent_ids), "One policy per agent required"
        
        # Store policies and agent info
        self.policies = {agent_id: policy for agent_id, policy in zip(agent_ids, policies)}
        self.agent_ids = agent_ids
        self.agent_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}
        
        # Set centralized critic for all policies
        for agent_id, policy in self.policies.items():
            policy.set_agent_id(agent_id)
            policy.set_critic(critic)
        
        self.critic = critic
    
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any
    ) -> Batch:
        """
        Forward pass for all agents.
        
        Args:
            batch: Batch of observations for all agents
            state: Optional state information
            
        Returns:
            Batch with actions for all agents
        """
        results = []
        
        for agent_id, policy in self.policies.items():
            # Get observations for this agent
            if hasattr(batch.obs, 'agent_id'):
                # Multi-agent batch format
                agent_mask = batch.obs.agent_id == agent_id
                if not agent_mask.any():
                    continue
                agent_batch = batch[agent_mask]
            else:
                # Assume batch is already for this agent
                agent_batch = batch
            
            # Get action from policy
            agent_state = None if state is None else state.get(agent_id, None)
            out = policy(agent_batch, state=agent_state, **kwargs)
            
            results.append({
                'agent_id': agent_id,
                'act': out.act,
                'out': out
            })
        
        # Aggregate results
        if len(results) == 0:
            return Batch()
        
        # Combine actions
        acts = {r['agent_id']: r['act'] for r in results}
        outs = {r['agent_id']: r['out'] for r in results}
        
        return Batch(act=acts, out=outs)
    
    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray
    ) -> Batch:
        """
        Process collected data for all agents.
        
        Computes global state and distributes to each agent's policy.
        
        Args:
            batch: Batch of collected data
            buffer: Replay buffer
            indices: Batch indices
            
        Returns:
            Processed batch
        """
        # Compute global state (concatenate all agent observations)
        critic_inp, critic_inp_next = self._get_global_state(batch)
        
        results = {}
        
        for agent_id, policy in self.policies.items():
            # Get data for this agent
            if hasattr(batch.obs, 'agent_id'):
                agent_mask = batch.obs.agent_id == agent_id
                if not agent_mask.any():
                    results[agent_id] = Batch()
                    continue
                agent_indices = indices[agent_mask]
                agent_batch = batch[agent_mask]
            else:
                agent_batch = batch
                agent_indices = indices
            
            # Add global state information
            agent_batch.critic_inp = critic_inp
            agent_batch.critic_inp_next = critic_inp_next
            
            # Process with agent's policy
            processed = policy.process_fn(agent_batch, buffer, agent_indices)
            results[agent_id] = processed
        
        return Batch(results)
    
    def learn(
        self,
        batch: Batch,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Learn from collected data for all agents.
        
        Args:
            batch: Batch of processed data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training statistics for each agent
        """
        results = {}
        
        # Get global state
        global_obs = []
        global_obs_next = []
        
        for agent_id in self.agent_ids:
            if agent_id not in batch or batch[agent_id].is_empty():
                continue
            
            agent_data = batch[agent_id]
            global_obs.append(agent_data.obs)
            global_obs_next.append(agent_data.obs_next)
        
        # Concatenate global state
        if len(global_obs) > 0:
            critic_inp = np.concatenate(global_obs, axis=-1)
            critic_inp_next = np.concatenate(global_obs_next, axis=-1)
        else:
            return results
        
        # Update each agent's policy
        for agent_id, policy in self.policies.items():
            if agent_id not in batch or batch[agent_id].is_empty():
                continue
            
            agent_data = batch[agent_id]
            
            # Add global state
            agent_data.critic_inp = critic_inp
            agent_data.critic_inp_next = critic_inp_next
            
            # Learn
            learn_result = policy.learn(agent_data, **kwargs)
            
            # Store results with agent prefix
            for key, value in learn_result.items():
                results[f"{agent_id}/{key}"] = value
        
        return results
    
    def _get_global_state(self, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract global state from batch.
        
        Concatenates observations from all agents.
        
        Args:
            batch: Batch with multi-agent data
            
        Returns:
            Tuple of (current_global_state, next_global_state)
        """
        obs_list = []
        obs_next_list = []
        
        for agent_id in self.agent_ids:
            if hasattr(batch.obs, 'agent_id'):
                agent_mask = batch.obs.agent_id == agent_id
                if agent_mask.any():
                    obs_list.append(batch.obs[agent_mask])
                    obs_next_list.append(batch.obs_next[agent_mask])
            else:
                # Fallback: assume batch structure
                obs_list.append(batch.obs)
                obs_next_list.append(batch.obs_next)
        
        if len(obs_list) == 0:
            return np.array([]), np.array([])
        
        global_obs = np.concatenate(obs_list, axis=-1)
        global_obs_next = np.concatenate(obs_next_list, axis=-1)
        
        return global_obs, global_obs_next
