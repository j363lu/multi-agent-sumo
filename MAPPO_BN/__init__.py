"""
MAPPO_BNTT — Batch-Normalization variant of the centralized critic.

This package is a self-contained extension of MAPPO/.  It imports from
MAPPO/ but never modifies it.  Only the pieces that differ from the baseline
are re-implemented here:

  • networks/critic.py  — CentralizedCriticBN (adds BatchNorm1d to hidden layers)
  • config/             — BNNetworkConfig + get_critic_bn_config()
  • training/trainer.py — MAPPOBNTTTrainer (overrides _create_policy_manager)

Everything else (actor, policy, env wrapper, evaluator, utils) is inherited
directly from MAPPO/.
"""
