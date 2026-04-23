"""
MAPPO Policy implementation for discrete action spaces.

Adapted from the reference MAPPO implementation to work with
discrete action spaces in SUMO traffic control.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from tianshou.data import Batch, ReplayBuffer

from mappo_traffic.networks import ActorNetwork, CentralizedCritic
from mappo_traffic.networks.utils import compute_gae, normalize_advantages


class MAPPOPolicy(nn.Module):
    """
    Multi-Agent PPO Policy for discrete action spaces.
    
    This implements the PPO algorithm adapted for multi-agent settings
    with a centralized critic.
    
    Args:
        actor: Actor network
        critic: Centralized critic network (shared across agents)
        optim_actor: Optimizer for actor
        optim_critic: Optimizer for critic
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        eps_clip: PPO clipping parameter
        value_clip: Whether to use value clipping
        dual_clip: Dual clipping parameter (optional)
        advantage_normalization: Whether to normalize advantages
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        reward_normalization: Whether to normalize rewards
    """
    
    def __init__(
        self,
        actor: ActorNetwork,
        critic: CentralizedCritic,
        optim_actor: Optimizer,
        optim_critic: Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        value_clip: bool = True,
        dual_clip: Optional[float] = None,
        advantage_normalization: bool = True,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        reward_normalization: bool = False,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        
        self.actor = actor
        self.critic = critic
        self.optim_actor = optim_actor
        self.optim_critic = optim_critic
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.dual_clip = dual_clip
        self.advantage_normalization = advantage_normalization
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.reward_normalization = reward_normalization
        
        # Agent identifier
        self._agent_id = None
        
    def set_agent_id(self, agent_id: str):
        """Set the agent ID for this policy."""
        self._agent_id = agent_id
    
    def set_critic(self, critic: CentralizedCritic):
        """Set the centralized critic (shared across agents)."""
        self.critic = critic
    
    def forward(
        self,
        batch: Batch,
        state: Optional[Any] = None,
        **kwargs: Any
    ) -> Batch:
        """
        Forward pass to get action from observation.
        
        Args:
            batch: Batch of observations
            state: Hidden state (not used)
            
        Returns:
            Batch containing actions and other info
        """
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.actor.action_head.weight.device)
        
        with torch.no_grad():
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            log_prob = dist.log_prob(act)
        
        return Batch(act=act.cpu().numpy(), logp=log_prob.cpu().numpy(), dist=dist)
    
    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray
    ) -> Batch:
        """
        Process collected data before learning.
        
        Computes advantages using GAE and value targets.
        
        Args:
            batch: Batch of collected transitions
            buffer: Replay buffer
            indices: Indices of the batch in buffer
            
        Returns:
            Processed batch with advantages and returns
        """
        # Get observations as tensors
        device = self.actor.action_head.weight.device
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        obs_next = torch.as_tensor(batch.obs_next, dtype=torch.float32, device=device)
        
        # Compute values using centralized critic
        # batch should have critic_inp and critic_inp_next from manager
        if hasattr(batch, 'critic_inp'):
            critic_inp = torch.as_tensor(batch.critic_inp, dtype=torch.float32, device=device)
            critic_inp_next = torch.as_tensor(batch.critic_inp_next, dtype=torch.float32, device=device)
        else:
            # Fallback to local observations if global state not available
            critic_inp = obs
            critic_inp_next = obs_next
        
        with torch.no_grad():
            values = self.critic(critic_inp).squeeze(-1).cpu().numpy()
            next_values = self.critic(critic_inp_next).squeeze(-1).cpu().numpy()
        
        # Compute advantages using GAE
        rewards = batch.rew
        dones = np.logical_or(batch.terminated, batch.truncated)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] * (1.0 - dones[t])
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + values[t]
        
        batch.adv = advantages
        batch.ret = returns
        batch.v_s = values
        
        # Store old log probs for PPO ratio
        act = torch.as_tensor(batch.act, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            batch.logp_old = dist.log_prob(act).cpu().numpy()
        
        return batch
    
    def learn(
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Update policy using PPO algorithm.
        
        Args:
            batch: Batch of transitions
            batch_size: Mini-batch size for updates
            repeat: Number of epochs to repeat
            
        Returns:
            Dictionary of training statistics
        """
        device = self.actor.action_head.weight.device
        
        losses = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        clip_fracs = []
        
        for _ in range(repeat):
            # Sample mini-batches
            for mini_batch in batch.split(batch_size, merge_last=True):
                # Convert to tensors
                obs = torch.as_tensor(mini_batch.obs, dtype=torch.float32, device=device)
                act = torch.as_tensor(mini_batch.act, dtype=torch.long, device=device)
                adv = torch.as_tensor(mini_batch.adv, dtype=torch.float32, device=device)
                ret = torch.as_tensor(mini_batch.ret, dtype=torch.float32, device=device)
                logp_old = torch.as_tensor(mini_batch.logp_old, dtype=torch.float32, device=device)
                v_s = torch.as_tensor(mini_batch.v_s, dtype=torch.float32, device=device)
                
                # Get critic input
                if hasattr(mini_batch, 'critic_inp'):
                    critic_inp = torch.as_tensor(mini_batch.critic_inp, dtype=torch.float32, device=device)
                else:
                    critic_inp = obs
                
                # Normalize advantages
                if self.advantage_normalization:
                    adv = normalize_advantages(adv)
                
                # Actor loss (PPO clipped objective)
                logits = self.actor(obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act)
                
                ratio = torch.exp(logp - logp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
                
                if self.dual_clip is not None:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * adv)
                    actor_loss = -torch.where(adv < 0, clip2, clip1).mean()
                else:
                    actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                value = self.critic(critic_inp).squeeze(-1)
                
                if self.value_clip:
                    v_clipped = v_s + torch.clamp(value - v_s, -self.eps_clip, self.eps_clip)
                    vf1 = (ret - value).pow(2)
                    vf2 = (ret - v_clipped).pow(2)
                    critic_loss = torch.max(vf1, vf2).mean()
                else:
                    critic_loss = (ret - value).pow(2).mean()
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                
                # Update networks
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward()
                
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.optim_actor.step()
                self.optim_critic.step()
                
                # Track statistics
                losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())
                
                # Track clipping fraction
                clip_frac = ((ratio - 1.0).abs() > self.eps_clip).float().mean().item()
                clip_fracs.append(clip_frac)
        
        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/critic": critic_losses,
            "loss/entropy": entropy_losses,
            "clip_frac": clip_fracs
        }
