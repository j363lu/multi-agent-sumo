"""
Multi-Agent Policy Manager for IPPO.

Coordinates multiple independent agent policies.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Batch, ReplayBuffer

from IPPO.agents.ippo_policy import IPPOPolicy


class MultiAgentPolicyManager(nn.Module):
    """
    Multi-agent policy manager for IPPO.
    
    Manages multiple IPPO policies (one per agent) and dispatches observations.
    
    Args:
        policies: List of IPPO policies (one per agent)
        agent_ids: List of agent IDs
    """
    
    def __init__(
        self,
        policies: List[IPPOPolicy],
        agent_ids: List[str],
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        
        assert len(policies) == len(agent_ids), "One policy per agent required"
        
        # Store policies and agent info
        self.policies = {agent_id: policy for agent_id, policy in zip(agent_ids, policies)}
        self.agent_ids = agent_ids
        self.agent_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}
        
        # Set agent identifiers for all policies.
        for agent_id, policy in self.policies.items():
            policy.set_agent_id(agent_id)
    
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
        
        Distributes per-agent data to each agent's policy.
        
        Args:
            batch: Batch of collected data
            buffer: Replay buffer
            indices: Batch indices
            
        Returns:
            Processed batch
        """
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
        
        # Update each agent's policy
        for agent_id, policy in self.policies.items():
            if agent_id not in batch or batch[agent_id].is_empty():
                continue
            
            agent_data = batch[agent_id]
            
            # Learn
            learn_result = policy.learn(agent_data, **kwargs)
            
            # Store results with agent prefix
            for key, value in learn_result.items():
                results[f"{agent_id}/{key}"] = value
        
        return results
    
