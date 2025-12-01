"""
Curriculum-aware sampling for quality-weighted training data.

This module provides a weighted sampler that adjusts sampling probabilities
based on the current curriculum phase. The sampler implements sampling-based
curriculum learning rather than loss weighting.

Examples
--------
>>> from admet.model.chemprop.curriculum_sampler import build_curriculum_sampler
>>> from admet.model.chemprop.curriculum import CurriculumState
>>>
>>> # Build sampler for training data
>>> quality_labels = df["Quality"].tolist()
>>> state = CurriculumState(qualities=["high", "medium", "low"])
>>> sampler = build_curriculum_sampler(quality_labels, state, seed=42)
>>>
>>> # Use with DataLoader
>>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from admet.model.chemprop.curriculum import CurriculumState


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
