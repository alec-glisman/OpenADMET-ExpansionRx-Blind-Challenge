"""
Curriculum-aware sampling for quality-weighted training data.

This module provides weighted samplers that adjust sampling probabilities
based on the current curriculum phase. The samplers implement sampling-based
curriculum learning rather than loss weighting.

Two sampler types are available:

1. **DynamicCurriculumSampler**: Recalculates weights on each iteration,
   responding to phase changes during training. This is the recommended sampler.

2. **build_curriculum_sampler**: Creates a static WeightedRandomSampler with
   fixed weights from the curriculum state at construction time. Useful for
   debugging or when phase changes are not expected.

Examples
--------
>>> from admet.model.chemprop.curriculum_sampler import DynamicCurriculumSampler
>>> from admet.model.chemprop.curriculum import CurriculumState
>>>
>>> # Build dynamic sampler that responds to phase changes
>>> quality_labels = df["Quality"].tolist()
>>> state = CurriculumState(qualities=["high", "medium", "low"])
>>> sampler = DynamicCurriculumSampler(quality_labels, state, seed=42)
>>>
>>> # Use with DataLoader
>>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler

if TYPE_CHECKING:
    from admet.model.chemprop.curriculum import CurriculumState

logger = logging.getLogger("admet.model.chemprop.curriculum_sampler")


class DynamicCurriculumSampler(Sampler[int]):
    """
    Dynamic sampler that recalculates weights based on curriculum phase.

    Unlike WeightedRandomSampler which uses fixed weights, this sampler
    reads from the CurriculumState on each iteration, allowing weights
    to update when phases change during training.

    Count Normalization
    -------------------
    When `count_normalize=True` in the CurriculumState config, the phase weights
    are interpreted as TARGET proportions and automatically adjusted for dataset
    size imbalance. This ensures that the specified proportions (e.g., 80% high-quality)
    actually reflect what appears in training batches, regardless of raw dataset sizes.

    For example, with High=5k, Medium=100k, Low=15k and target [0.8, 0.15, 0.05]:
    - Without count normalization: High gets ~4% of batches (dominated by Medium)
    - With count normalization: High gets ~80% of batches (as intended)

    Parameters
    ----------
    quality_labels : Sequence[str]
        Quality label for each sample in the dataset.
    curriculum_state : CurriculumState
        Curriculum state object (weights are read dynamically).
    num_samples : int, optional
        Number of samples per epoch. If None, uses len(quality_labels).
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _last_phase : str
        Last observed phase, used to log phase changes.
    _quality_counts : dict[str, int]
        Number of samples per quality level (for count normalization).

    Examples
    --------
    >>> state = CurriculumState(qualities=["high", "medium", "low"])
    >>> sampler = DynamicCurriculumSampler(quality_labels, state)
    >>> # Sampler will automatically use updated weights when state.phase changes
    """

    def __init__(
        self,
        quality_labels: Sequence[str],
        curriculum_state: "CurriculumState",
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not quality_labels:
            raise ValueError("quality_labels cannot be empty")

        self.quality_labels = list(quality_labels)
        self.curriculum_state = curriculum_state
        self._num_samples = num_samples or len(quality_labels)
        self.seed = seed

        # Track phase for logging changes
        self._last_phase: str | None = None

        # Pre-compute label-to-index mapping for efficiency
        self._label_indices: dict[str, list[int]] = {}
        for i, label in enumerate(self.quality_labels):
            if label not in self._label_indices:
                self._label_indices[label] = []
            self._label_indices[label].append(i)

        # Compute quality counts for count normalization
        self._quality_counts: dict[str, int] = {}
        for label in self.quality_labels:
            self._quality_counts[label] = self._quality_counts.get(label, 0) + 1

        # Log dataset composition
        total = len(self.quality_labels)
        composition = {q: f"{c} ({100*c/total:.1f}%)" for q, c in self._quality_counts.items()}
        logger.info("DynamicCurriculumSampler initialized with dataset composition: %s", composition)

        # Warn about unknown quality labels
        present = set(self.quality_labels)
        defined = set(curriculum_state.qualities)
        unknown = present - defined
        if unknown:
            import warnings

            warnings.warn(
                f"Quality labels {unknown} not in curriculum qualities "
                f"{curriculum_state.qualities}. These samples will never be sampled.",
                UserWarning,
                stacklevel=2,
            )

    def _compute_weights(self) -> np.ndarray:
        """Compute sample weights from current curriculum state.

        When count_normalize=True in the curriculum config, the target proportions
        are converted to per-sample weights that achieve those proportions regardless
        of dataset size imbalance.

        Count Normalization Formula
        ---------------------------
        To achieve target proportion P for quality Q with count C_Q:
            per_sample_weight_Q = P / C_Q

        This ensures samples from quality Q appear in ~P fraction of batches.
        """
        target_probs = self.curriculum_state.sampling_probs()
        config = self.curriculum_state.config
        count_normalize = getattr(config, "count_normalize", True)

        weights = np.zeros(len(self.quality_labels), dtype=np.float64)

        if count_normalize:
            # Count-normalized: target proportions become actual batch proportions
            for i, label in enumerate(self.quality_labels):
                target_prop = target_probs.get(label, 0.0)
                count = self._quality_counts.get(label, 1)
                # Per-sample weight = target_proportion / count
                # This gives each quality level its target share of batches
                weights[i] = target_prop / count if count > 0 else 0.0
        else:
            # Legacy behavior: apply target proportions as direct weights
            for i, label in enumerate(self.quality_labels):
                weights[i] = target_probs.get(label, 0.0)

        # Handle all-zero weights
        if weights.sum() == 0:
            logger.warning("All sample weights are zero. Using uniform sampling.")
            weights = np.ones(len(self.quality_labels), dtype=np.float64)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

    def _log_effective_proportions(self) -> None:
        """Log the effective sampling proportions for debugging."""
        weights = self._compute_weights()
        effective_props = {}
        for quality in self.curriculum_state.qualities:
            indices = self._label_indices.get(quality, [])
            if indices:
                effective_props[quality] = float(np.sum(weights[indices]))
        logger.debug(
            "Phase %s: target=%s, effective=%s",
            self.curriculum_state.phase,
            self.curriculum_state.sampling_probs(),
            effective_props,
        )

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices based on current curriculum weights."""
        # Log phase changes
        current_phase = self.curriculum_state.phase
        if self._last_phase is not None and current_phase != self._last_phase:
            logger.info(
                "DynamicCurriculumSampler: phase changed %s -> %s, target_probs=%s",
                self._last_phase,
                current_phase,
                self.curriculum_state.sampling_probs(),
            )
            self._log_effective_proportions()
        self._last_phase = current_phase

        # Compute weights based on current phase
        weights = self._compute_weights()

        # Create generator with optional seed
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()

        # Sample indices according to weights
        indices = rng.choice(
            len(self.quality_labels),
            size=self._num_samples,
            replace=True,
            p=weights,
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self._num_samples


def build_curriculum_sampler(
    quality_labels: Sequence[str],
    curriculum_state: CurriculumState,
    num_samples: int | None = None,
    seed: int | None = None,
) -> WeightedRandomSampler:
    """
    Build a weighted sampler based on curriculum phase.

    Creates a WeightedRandomSampler that samples training data according to
    the quality weights defined by the current curriculum phase. This enables
    curriculum learning by controlling which samples appear in each epoch.

    Parameters
    ----------
    quality_labels : Sequence[str]
        Quality label for each sample in the dataset (e.g., ["high", "low", "medium", ...]).
        Labels must match qualities defined in curriculum_state.
    curriculum_state : CurriculumState
        Current curriculum state containing phase and quality weights.
    num_samples : int, optional
        Number of samples to draw per epoch. If None, uses len(quality_labels)
        for one full epoch's worth of samples.
    seed : int, optional
        Random seed for reproducible sampling. If None, sampling is non-deterministic.

    Returns
    -------
    WeightedRandomSampler
        A sampler that weights samples according to curriculum phase probabilities.

    Raises
    ------
    ValueError
        If quality_labels is empty or contains unknown quality labels.

    Notes
    -----
    The sampler handles edge cases gracefully:
    - Unknown quality labels: Assigned weight of 0 (never sampled)
    - Missing qualities in data: Redistributes weights among present qualities
    - All zero weights: Falls back to uniform sampling

    Examples
    --------
    >>> quality_labels = ["high", "high", "medium", "low", "high"]
    >>> state = CurriculumState(qualities=["high", "medium", "low"])
    >>> state.phase = "warmup"  # weights: high=0.9, medium=0.1, low=0.0
    >>> sampler = build_curriculum_sampler(quality_labels, state, seed=42)
    >>> # High-quality samples will be sampled much more frequently
    """
    if not quality_labels:
        raise ValueError("quality_labels cannot be empty")

    n_samples = len(quality_labels)
    num_samples = num_samples or n_samples

    # Get sampling probabilities from curriculum state
    probs = curriculum_state.sampling_probs()

    # Identify which qualities are actually present in the data
    present_qualities = set(quality_labels)
    defined_qualities = set(curriculum_state.qualities)

    # Warn about unknown qualities (they get 0 weight)
    unknown = present_qualities - defined_qualities
    if unknown:
        import warnings

        warnings.warn(
            f"Quality labels {unknown} not in curriculum qualities {curriculum_state.qualities}. "
            "These samples will have zero weight (never sampled).",
            UserWarning,
            stacklevel=2,
        )

    # Compute sample weights
    weights = np.zeros(n_samples, dtype=np.float64)
    for i, label in enumerate(quality_labels):
        weights[i] = probs.get(label, 0.0)

    # Handle edge case: all zero weights -> uniform sampling
    if weights.sum() == 0:
        import warnings

        warnings.warn(
            "All sample weights are zero. Falling back to uniform sampling.",
            UserWarning,
            stacklevel=2,
        )
        weights = np.ones(n_samples, dtype=np.float64)

    # Normalize weights (not strictly necessary for WeightedRandomSampler,
    # but helpful for debugging)
    weights = weights / weights.sum()

    # Set random seed for reproducibility
    if seed is not None:
        # WeightedRandomSampler uses torch random state, so we need to
        # use a torch generator
        import torch

        torch_generator = torch.Generator()
        torch_generator.manual_seed(seed)
    else:
        torch_generator = None

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=num_samples,
        replacement=True,
        generator=torch_generator,
    )


def get_quality_indices(
    quality_labels: Sequence[str],
    qualities: Sequence[str],
) -> dict[str, list[int]]:
    """
    Get indices for each quality level in the dataset.

    Useful for computing per-quality metrics by splitting predictions/targets
    by quality level.

    Parameters
    ----------
    quality_labels : Sequence[str]
        Quality label for each sample in the dataset.
    qualities : Sequence[str]
        List of quality levels to extract indices for.

    Returns
    -------
    dict[str, list[int]]
        Mapping from quality level to list of sample indices.

    Examples
    --------
    >>> quality_labels = ["high", "medium", "high", "low", "medium"]
    >>> indices = get_quality_indices(quality_labels, ["high", "medium", "low"])
    >>> indices["high"]
    [0, 2]
    >>> indices["medium"]
    [1, 4]
    >>> indices["low"]
    [3]
    """
    indices: dict[str, list[int]] = {q: [] for q in qualities}
    for i, label in enumerate(quality_labels):
        if label in indices:
            indices[label].append(i)
    return indices


def compute_per_quality_weights(
    quality_labels: Sequence[str],
    curriculum_state: CurriculumState,
) -> dict[str, float]:
    """
    Compute the effective weight per quality level for the current phase.

    This is useful for understanding how much each quality level contributes
    to training in the current curriculum phase.

    Parameters
    ----------
    quality_labels : Sequence[str]
        Quality label for each sample in the dataset.
    curriculum_state : CurriculumState
        Current curriculum state.

    Returns
    -------
    dict[str, float]
        Mapping from quality level to its effective sampling weight.
        Accounts for both the phase weights and the count of samples
        at each quality level.

    Examples
    --------
    >>> quality_labels = ["high"] * 100 + ["medium"] * 200 + ["low"] * 300
    >>> state = CurriculumState()
    >>> state.phase = "warmup"
    >>> weights = compute_per_quality_weights(quality_labels, state)
    >>> # Even with more low-quality samples, high gets higher weight
    """
    probs = curriculum_state.sampling_probs()

    # Count samples per quality
    counts: dict[str, int] = {}
    for label in quality_labels:
        counts[label] = counts.get(label, 0) + 1

    # Compute effective weight: phase_prob / count (normalized)
    effective = {}
    for q in curriculum_state.qualities:
        count = counts.get(q, 0)
        if count > 0:
            # Per-sample weight from this quality
            effective[q] = probs.get(q, 0.0) * count
        else:
            effective[q] = 0.0

    # Normalize
    total = sum(effective.values())
    if total > 0:
        effective = {k: v / total for k, v in effective.items()}

    return effective
