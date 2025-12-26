"""
Joint sampler for combined task-aware and curriculum-aware sampling.

This module provides a unified sampler that combines:
1. Task-aware oversampling: Two-stage sampling that first picks a task according
   to inverse-power probabilities, then samples a molecule with that task's label.
2. Curriculum-aware sampling: Adjusts sampling based on data quality labels
   that change with curriculum phase progression.

The two strategies are combined via two-stage sampling with curriculum weighting:
1. Sample task t according to task probabilities: p_t ∝ count_t^(-α)
2. Sample molecule from task t's molecules, weighted by curriculum: p_i ∝ w_curriculum[i]

This preserves the original TaskAwareSampler behavior when curriculum is disabled.

.. warning::
    **num_workers Limitation**: When using this sampler with `num_workers > 0` in
    DataLoader, each worker gets its own copy of the sampler. The internal
    `_current_epoch` counter and curriculum phase state will not be synchronized
    across workers, potentially causing inconsistent sampling behavior. For reliable
    curriculum learning, use `num_workers=0`.

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

    Uses two-stage sampling that preserves the original TaskAwareSampler behavior:
    1. Sample task t with probability p_t ∝ count_t^(-α)
    2. Sample molecule from task t's valid molecules, weighted by curriculum

    When curriculum is disabled, this reduces to uniform sampling within each
    task (identical to TaskAwareSampler). When task_alpha=0, all tasks are
    equally likely and curriculum weights control within-task sampling.

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
        If True, increments seed each epoch (seed + epoch_number) for sampling variety.
        This means different samples are drawn each epoch, which is generally desired
        for training. If False, uses same seed each epoch, resulting in identical
        sampling every epoch (deterministic but may limit generalization).
        **Note**: When True, training is NOT fully reproducible across runs even with
        the same base seed, unless you also track and restore the epoch counter.
    log_weight_stats : bool, default=True
        Whether to log weight statistics (min, max, entropy, effective samples).

    Attributes
    ----------
    _current_epoch : int
        Current epoch counter for seed incrementing.
    _last_phase : str | None
        Last observed curriculum phase for logging changes.
    task_indices : list[np.ndarray]
        Valid molecule indices for each task.
    task_probs : np.ndarray
        Sampling probability for each task.

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

        # Store last computed weights and stats for callback access
        self._last_weights: np.ndarray | None = None
        self._last_weight_stats: dict[str, float] | None = None

        # Validate alpha and warn if outside recommended range
        if task_alpha < 0 or task_alpha > 1:
            logger.warning(
                "task_alpha=%.2f outside recommended range [0, 1]. "
                "Values outside this range may produce unexpected behavior.",
                task_alpha,
            )

        # Precompute task information (like original TaskAwareSampler)
        self.num_tasks = targets.shape[1]
        self.task_indices: list[np.ndarray] = []
        task_counts = []

        for t in range(self.num_tasks):
            valid_mask = ~np.isnan(targets[:, t])
            indices = np.where(valid_mask)[0]
            self.task_indices.append(indices)
            task_counts.append(len(indices))

        self.task_counts = np.array(task_counts, dtype=float)

        # Calculate task sampling probabilities: p_t ∝ count_t^(-α)
        weights = np.power(self.task_counts + 1e-6, -self.task_alpha)
        self.task_probs = weights / np.sum(weights)

        logger.info(
            "JointSampler initialized: task_alpha=%.2f, curriculum=%s, " "increment_seed=%s, num_samples=%d",
            task_alpha,
            curriculum_state is not None,
            increment_seed_per_epoch,
            self._num_samples,
        )
        logger.info("Task counts: %s", self.task_counts)
        logger.info("Task probabilities: %s", np.round(self.task_probs, 4))

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

    def _sample_from_task(
        self,
        task_idx: int,
        curriculum_weights: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample a molecule from the given task's valid indices.

        Parameters
        ----------
        task_idx : int
            Task index to sample from.
        curriculum_weights : np.ndarray
            Curriculum weights for all samples.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        int
            Sampled molecule index.
        """
        valid_indices = self.task_indices[task_idx]

        if len(valid_indices) == 0:
            # Fallback: sample uniformly from all samples
            return rng.integers(0, len(self.targets))

        # Get curriculum weights for valid indices
        task_curriculum_weights = curriculum_weights[valid_indices]

        # Normalize weights within this task
        total = task_curriculum_weights.sum()
        if total == 0:
            # Uniform sampling within task
            probs = np.ones(len(valid_indices)) / len(valid_indices)
        else:
            probs = task_curriculum_weights / total

        # Sample molecule index within task
        local_idx = rng.choice(len(valid_indices), p=probs)
        return valid_indices[local_idx]

    def _log_weight_statistics(self, weights: np.ndarray) -> None:
        """Log weight distribution statistics for monitoring."""
        if not self.log_weight_stats:
            return

        # Basic statistics
        min_weight = weights.min()
        max_weight = weights.max()
        mean_weight = weights.mean()

        # Entropy (measure of uniformity)
        eps = 1e-10
        entropy = -np.sum(weights * np.log(weights + eps))

        # Effective number of samples (inverse of sum of squared weights)
        # Higher = more uniform, lower = more concentrated
        effective_samples = 1.0 / np.sum(weights**2)

        stats = {
            "min": min_weight,
            "max": max_weight,
            "mean": mean_weight,
            "entropy": entropy,
            "effective_samples": effective_samples,
        }

        logger.info(
            "Weight stats: min=%.6f, max=%.6f, mean=%.6f, " "entropy=%.3f, effective_samples=%.1f",
            min_weight,
            max_weight,
            mean_weight,
            entropy,
            effective_samples,
        )

        # Store for potential MLflow logging by callback
        self._last_weight_stats = stats

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices using two-stage sampling.

        Stage 1: Sample task t with probability p_t ∝ count_t^(-α)
        Stage 2: Sample molecule from task t's valid indices, weighted by curriculum

        This preserves the original TaskAwareSampler behavior while integrating
        curriculum learning.

        Yields
        ------
        int
            Sample indices drawn according to two-stage sampling.
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

        # Compute current curriculum weights
        curriculum_weights = self._compute_curriculum_weights()

        # Normalize for logging
        weights = curriculum_weights / curriculum_weights.sum()
        self._log_weight_statistics(weights)

        # Determine seed for this epoch
        if self.increment_seed_per_epoch:
            epoch_seed = self.seed + self._current_epoch
            self._current_epoch += 1
        else:
            epoch_seed = self.seed

        rng = np.random.default_rng(epoch_seed)

        # Two-stage sampling
        indices: list[int] = []
        for _ in range(self._num_samples):
            # Stage 1: Sample task according to task probabilities
            task_idx = rng.choice(self.num_tasks, p=self.task_probs)

            # Stage 2: Sample molecule from task, weighted by curriculum
            mol_idx = self._sample_from_task(task_idx, curriculum_weights, rng)
            indices.append(mol_idx)

        return iter(indices)

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self._num_samples
