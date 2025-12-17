"""
Joint sampler for combined task-aware and curriculum-aware sampling.

This module provides a unified sampler that combines:
1. Task-aware oversampling: Rebalances sampling across tasks with different
   label counts using inverse-power weighting.
2. Curriculum-aware sampling: Adjusts sampling based on data quality labels
   that change with curriculum phase progression.

The two strategies are combined via multiplicative weight composition:
    w_joint[i] = w_task[i] × w_quality[i]

For multi-task samples, the "primary" task (used for weight computation)
is the rarest task among those the sample has labels for.

Examples
--------
>>> from admet.model.chemprop.joint_sampler import JointSampler
>>> from admet.model.chemprop.curriculum import CurriculumState
>>>
>>> # Create joint sampler with both strategies
>>> sampler = JointSampler(
...     targets=target_array,           # (N, T) array with NaN for missing
...     quality_labels=quality_list,    # ["high", "medium", "low", ...]
...     curriculum_state=curr_state,    # CurriculumState object
...     task_alpha=0.3,                 # Task rebalancing strength
...     num_samples=1000,
...     seed=42,
...     increment_seed_per_epoch=True,  # Vary sampling across epochs
... )
>>>
>>> # Use with DataLoader
>>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from admet.model.chemprop.curriculum import CurriculumState

logger = logging.getLogger("admet.model.chemprop.joint_sampler")


class JointSampler(Sampler[int]):
    """
    Unified sampler combining task-aware oversampling and curriculum learning.

    Computes joint sampling weights via multiplicative composition of:
    1. Task weights: w_task[i] ∝ (count[primary_task[i]])^(-α)
    2. Curriculum weights: w_curriculum[i] from CurriculumState

    Final weight: w_joint[i] = w_task[i] × w_curriculum[i]

    For samples with multiple task labels, the "primary task" is the rarest
    task (smallest label count) among those the sample has labels for.

    Parameters
    ----------
    targets : np.ndarray
        Target matrix of shape (N, T) where N is number of samples and
        T is number of tasks. Missing labels should be NaN.
    quality_labels : Sequence[str] | None
        Quality label for each sample. If None, curriculum weights are uniform.
    curriculum_state : CurriculumState | None
        Curriculum state object for quality-aware weights. If None, curriculum
        weights are uniform.
    task_alpha : float, default=0.0
        Exponent for inverse-power task sampling schedule [0, 1].
        - alpha=0: Uniform task weighting (no rebalancing)
        - alpha=0.5: Moderate rebalancing
        - alpha=1: Full inverse-proportional (rare tasks heavily favored)
    num_samples : int | None, default=None
        Number of samples per epoch. If None, uses len(targets).
    seed : int, default=42
        Base random seed for reproducibility.
    increment_seed_per_epoch : bool, default=True
        If True, increments seed each epoch for sampling variety.
        If False, uses same seed each epoch.
    log_weight_stats : bool, default=True
        Whether to log weight statistics (min, max, entropy, effective samples).

    Attributes
    ----------
    _current_epoch : int
        Current epoch counter for seed incrementing.
    _last_phase : str | None
        Last observed curriculum phase for logging changes.

    Examples
    --------
    >>> # Task oversampling only (no curriculum)
    >>> sampler = JointSampler(
    ...     targets=targets,
    ...     quality_labels=None,
    ...     curriculum_state=None,
    ...     task_alpha=0.5,
    ... )
    >>>
    >>> # Curriculum only (no task oversampling)
    >>> sampler = JointSampler(
    ...     targets=targets,
    ...     quality_labels=quality_labels,
    ...     curriculum_state=state,
    ...     task_alpha=0.0,
    ... )
    >>>
    >>> # Both strategies combined
    >>> sampler = JointSampler(
    ...     targets=targets,
    ...     quality_labels=quality_labels,
    ...     curriculum_state=state,
    ...     task_alpha=0.3,
    ... )
    """

    def __init__(
        self,
        targets: np.ndarray,
        quality_labels: Sequence[str] | None = None,
        curriculum_state: "CurriculumState | None" = None,
        task_alpha: float = 0.0,
        num_samples: int | None = None,
        seed: int = 42,
        increment_seed_per_epoch: bool = True,
        log_weight_stats: bool = True,
    ) -> None:
        super().__init__(None)  # type: ignore
        self.targets = targets
        self.quality_labels = list(quality_labels) if quality_labels else None
        self.curriculum_state = curriculum_state
        self.task_alpha = task_alpha
        self._num_samples = num_samples or len(targets)
        self.seed = seed
        self.increment_seed_per_epoch = increment_seed_per_epoch
        self.log_weight_stats = log_weight_stats

        # Epoch tracking for seed incrementation
        self._current_epoch = 0
        self._last_phase: str | None = None

        # Validate alpha and warn if outside recommended range
        if task_alpha < 0 or task_alpha > 1:
            logger.warning(
                "task_alpha=%.2f outside recommended range [0, 1]. "
                "Values outside this range may produce unexpected behavior.",
                task_alpha,
            )

        # Precompute task information
        self.num_tasks = targets.shape[1]
        self.task_counts = np.sum(~np.isnan(targets), axis=0).astype(float)

        # Precompute primary task for each sample (rarest task)
        self._primary_tasks = self._compute_primary_tasks()

        logger.info(
            "JointSampler initialized: task_alpha=%.2f, curriculum=%s, " "increment_seed=%s, num_samples=%d",
            task_alpha,
            curriculum_state is not None,
            increment_seed_per_epoch,
            self._num_samples,
        )
        logger.info("Task counts: %s", self.task_counts)

    def _compute_primary_tasks(self) -> np.ndarray:
        """
        Compute primary task (rarest) for each sample.

        For samples with multiple task labels, selects the task with the
        smallest label count. This ensures rare tasks influence sampling.

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing primary task index for each sample.
        """
        primary_tasks = np.zeros(len(self.targets), dtype=int)

        for i in range(len(self.targets)):
            valid_tasks = ~np.isnan(self.targets[i])
            if not valid_tasks.any():
                # No valid tasks for this sample - use first task as fallback
                primary_tasks[i] = 0
                continue

            # Select task with minimum count among valid tasks
            valid_counts = self.task_counts.copy()
            valid_counts[~valid_tasks] = np.inf
            primary_tasks[i] = np.argmin(valid_counts)

        return primary_tasks

    def _compute_task_weights(self) -> np.ndarray:
        """
        Compute task-aware weights using inverse-power scheduling.

        Returns per-sample weights based on the primary task label count:
            w_task[i] ∝ (count[primary_task[i]])^(-α)

        Returns
        -------
        np.ndarray
            Unnormalized task weights of shape (N,).
        """
        if self.task_alpha == 0.0:
            # Uniform task weighting
            return np.ones(len(self.targets), dtype=np.float64)

        # Get count for each sample's primary task
        sample_task_counts = self.task_counts[self._primary_tasks]

        # Compute weights: w ∝ count^(-α)
        # Add epsilon to avoid division by zero
        weights = np.power(sample_task_counts + 1e-6, -self.task_alpha)

        return weights

    def _compute_curriculum_weights(self) -> np.ndarray:
        """
        Compute curriculum-aware weights from current phase.

        Returns per-sample weights based on quality labels and curriculum phase:
            w_curriculum[i] = phase_prob[quality[i]]

        Returns
        -------
        np.ndarray
            Unnormalized curriculum weights of shape (N,).
        """
        if self.quality_labels is None or self.curriculum_state is None:
            # Uniform curriculum weighting
            return np.ones(len(self.targets), dtype=np.float64)

        # Get current phase probabilities
        probs = self.curriculum_state.sampling_probs()

        # Assign weight to each sample based on quality
        weights = np.array([probs.get(label, 0.0) for label in self.quality_labels])

        # Handle all-zero weights
        if weights.sum() == 0:
            logger.warning(
                "All curriculum weights are zero in phase %s. Using uniform weights.",
                self.curriculum_state.phase,
            )
            weights = np.ones(len(self.targets), dtype=np.float64)

        return weights

    def _compute_joint_weights(self) -> np.ndarray:
        """
        Compute joint sampling weights via multiplicative composition.

        Combines task and curriculum weights:
            w_joint[i] = w_task[i] × w_curriculum[i]

        Then normalizes to valid probability distribution.

        Returns
        -------
        np.ndarray
            Normalized joint weights of shape (N,) summing to 1.
        """
        task_weights = self._compute_task_weights()
        curriculum_weights = self._compute_curriculum_weights()

        # Multiplicative composition
        joint_weights = task_weights * curriculum_weights

        # Normalize to probability distribution
        total = joint_weights.sum()
        if total == 0:
            logger.warning("All joint weights are zero. Using uniform sampling.")
            joint_weights = np.ones(len(self.targets), dtype=np.float64)
            total = len(self.targets)

        return joint_weights / total

    def _log_weight_statistics(self, weights: np.ndarray) -> None:
        """Log weight distribution statistics for monitoring."""
        if not self.log_weight_stats:
            return

        # Basic statistics
        min_weight = weights.min()
        max_weight = weights.max()
        mean_weight = weights.mean()

        # Entropy (measure of uniformity)
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy = -np.sum(weights * np.log(weights + eps))

        # Effective number of samples (inverse of sum of squared weights)
        # Higher = more uniform, lower = more concentrated
        effective_samples = 1.0 / np.sum(weights**2)

        logger.info(
            "Weight stats: min=%.6f, max=%.6f, mean=%.6f, " "entropy=%.3f, effective_samples=%.1f",
            min_weight,
            max_weight,
            mean_weight,
            entropy,
            effective_samples,
        )

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices based on joint weights.

        Computes joint weights on each iteration to capture curriculum
        phase transitions. Uses epoch-specific seed if increment_seed_per_epoch
        is enabled.

        Yields
        ------
        int
            Sample indices drawn according to joint weights.
        """
        # Log phase changes
        if self.curriculum_state is not None:
            current_phase = self.curriculum_state.phase
            if self._last_phase is not None and current_phase != self._last_phase:
                logger.info(
                    "JointSampler: curriculum phase changed %s -> %s",
                    self._last_phase,
                    current_phase,
                )
            self._last_phase = current_phase

        # Compute current joint weights
        weights = self._compute_joint_weights()
        self._log_weight_statistics(weights)

        # Determine seed for this epoch
        if self.increment_seed_per_epoch:
            epoch_seed = self.seed + self._current_epoch
            self._current_epoch += 1
        else:
            epoch_seed = self.seed

        # Sample indices according to weights
        rng = np.random.default_rng(epoch_seed)
        indices = rng.choice(
            len(self.targets),
            size=self._num_samples,
            replace=True,
            p=weights,
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self._num_samples
