"""
MAPPOBNTTTrainer — MAPPO trainer with BatchNorm-augmented centralized critic.

This class subclasses MAPPOTrainer from MAPPO/ and overrides only
_create_policy_manager() to substitute CentralizedCriticBN for the standard
CentralizedCritic when config.network.use_critic_bn is True.

All other training logic (data collection, GAE computation, PPO update,
evaluation, logging, checkpointing) is inherited from MAPPOTrainer unchanged.
This ensures the A/B comparison between baseline and BN-critic is controlled —
only the critic architecture differs.
"""

from torch.optim import Adam

from MAPPO.config import ExperimentConfig
from MAPPO.agents import MAPPOPolicy, MultiAgentPolicyManager
from MAPPO.networks.actor import ActorNetwork
from MAPPO.training.trainer import MAPPOTrainer
from MAPPO.utils.reward_normalizer import RewardNormalizer

from MAPPO_BNTT.networks.critic import CentralizedCriticBN


class MAPPOBNTTTrainer(MAPPOTrainer):
    """
    MAPPO trainer whose centralized critic may include BatchNorm1d layers.

    The config.network field is expected to be a BNNetworkConfig instance
    (from MAPPO_BNTT.config) that carries a use_critic_bn boolean.  If that
    attribute is absent (e.g. a plain NetworkConfig is passed), the trainer
    falls back to use_critic_bn=False and behaves identically to the baseline.

    Args:
        config: Experiment configuration.  The .network field should be a
                BNNetworkConfig with use_critic_bn=True for the BN variant.
    """

    def __init__(self, config: ExperimentConfig):
        # Delegate entirely to MAPPOTrainer.__init__, which calls
        # self._create_policy_manager() — our override is picked up
        # automatically via Python's method resolution order.
        super().__init__(config)

    # ------------------------------------------------------------------
    # Override: build centralized critic with optional BatchNorm1d
    # ------------------------------------------------------------------
    def _create_policy_manager(self) -> MultiAgentPolicyManager:
        """
        Build actor networks and (BN-augmented) centralized critic.

        Mirrors MAPPOTrainer._create_policy_manager() exactly, with the
        single change that CentralizedCriticBN is instantiated instead of
        CentralizedCritic.  The use_critic_bn flag is read from
        config.network (defaults to False if the attribute is missing so
        this trainer degrades gracefully to the baseline).
        """
        net_config = self.config.network
        mappo_config = self.config.mappo

        # Read use_critic_bn safely — plain NetworkConfig won't have it.
        use_bn = getattr(net_config, "use_critic_bn", False)

        # ── Observation dimensions ─────────────────────────────────────
        agent_obs_dims = {}
        total_obs_dim = 0
        for agent_id in self.agent_ids:
            obs_space = self.train_env.sumo_env.observation_space(agent_id)
            agent_obs_dims[agent_id] = obs_space.shape[0]
            total_obs_dim += obs_space.shape[0]

        print(f"[BNTT] Agent observation dimensions: {agent_obs_dims}")
        print(f"[BNTT] use_critic_bn = {use_bn}")

        # ── Centralized critic (BN variant) ────────────────────────────
        critic = CentralizedCriticBN(
            global_obs_dim=total_obs_dim,
            hidden_dims=net_config.critic_hidden,
            activation=net_config.activation,
            use_orthogonal_init=net_config.use_orthogonal_init,
            use_batch_norm=use_bn,
        ).to(self.device)

        critic_optimizer = Adam(critic.parameters(), lr=mappo_config.lr_critic)

        # ── Per-agent actor networks (unchanged from baseline) ─────────
        policies = []
        for agent_id in self.agent_ids:
            actor = ActorNetwork(
                obs_dim=agent_obs_dims[agent_id],
                action_dim=self.action_dim,
                hidden_dims=net_config.actor_hidden,
                activation=net_config.activation,
                use_orthogonal_init=net_config.use_orthogonal_init,
            ).to(self.device)

            actor_optimizer = Adam(actor.parameters(), lr=mappo_config.lr_actor)

            policy = MAPPOPolicy(
                actor=actor,
                critic=critic,          # shared BN critic
                optim_actor=actor_optimizer,
                optim_critic=critic_optimizer,
                gamma=mappo_config.gamma,
                gae_lambda=mappo_config.gae_lambda,
                eps_clip=mappo_config.eps_clip,
                value_clip=mappo_config.value_clip,
                dual_clip=mappo_config.dual_clip,
                advantage_normalization=mappo_config.advantage_normalization,
                vf_coef=mappo_config.vf_coef,
                ent_coef=mappo_config.ent_coef,
                max_grad_norm=mappo_config.max_grad_norm,
                reward_normalization=mappo_config.reward_normalization,
                reward_normalizer=self._reward_normalizer,
            )
            policies.append(policy)

        return MultiAgentPolicyManager(
            policies=policies,
            critic=critic,
            agent_ids=self.agent_ids,
        )
