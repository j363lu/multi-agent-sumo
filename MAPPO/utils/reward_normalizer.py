"""
Running reward normalization using Welford's online algorithm.

Normalizing rewards to zero-mean unit-variance reduces critic targets from
O(reward/(1-gamma)) ≈ O(100) to O(1), which is required for value_clip to
function correctly and greatly improves critic accuracy.
"""

import numpy as np


class RewardNormalizer:
    """
    Online running mean/variance normalization for rewards.

    Uses Welford's online algorithm for numerically stable single-pass
    computation of mean and variance across all steps and episodes seen
    so far.  A single shared instance is used across all agents so the
    normalizer sees the full reward distribution, not one agent's slice.

    Args:
        epsilon: Small constant added to std to prevent division by zero.
        clip: Clip normalized rewards to [-clip, clip].  10.0 keeps rare
            extreme events from dominating the gradient while preserving
            the sign of the advantage.  Set to 0 or inf to disable.
    """

    def __init__(self, epsilon: float = 1e-8, clip: float = 10.0):
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0          # sum of squared deviations from mean
        self.epsilon = epsilon
        self.clip = clip

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a flat array of reward values."""
        for xi in np.asarray(x, dtype=np.float64).flat:
            self._count += 1
            delta = xi - self._mean
            self._mean += delta / self._count
            delta2 = xi - self._mean
            self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        """Population std dev; returns 1.0 until at least 2 samples seen."""
        if self._count < 2:
            return 1.0
        return float(np.sqrt(self._M2 / self._count)) + self.epsilon

    @property
    def count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize x to (x - mean) / std and optionally update running stats.

        Args:
            x: Reward array of any shape.
            update: If True, update running statistics with the values in x
                before normalizing.  Pass False during evaluation so the
                running stats are not polluted by deterministic eval episodes.

        Returns:
            Normalized array of the same shape and dtype as x, clipped to
            [-clip, clip] if clip > 0.
        """
        x = np.asarray(x, dtype=np.float32)
        if update:
            self.update(x)
        normalized = (x - self.mean) / self.std
        if self.clip > 0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # Serialisation (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "count": self._count,
            "mean":  self._mean,
            "M2":    self._M2,
            "epsilon": self.epsilon,
            "clip":    self.clip,
        }

    def load_state_dict(self, d: dict) -> None:
        self._count   = d["count"]
        self._mean    = d["mean"]
        self._M2      = d["M2"]
        self.epsilon  = d.get("epsilon", self.epsilon)
        self.clip     = d.get("clip",    self.clip)

    def __repr__(self) -> str:
        return (
            f"RewardNormalizer(count={self._count}, "
            f"mean={self._mean:.4f}, std={self.std:.4f})"
        )
