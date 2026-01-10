"""
Task-aware sampling for imbalanced multi-task learning.

This module provides a sampler that balances training by sampling tasks
according to an inverse-power schedule of their frequencies, then selecting
molecules that have labels for the chosen task.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger("admet.model.chemprop.task_sampler")


class TaskAwareSampler(Sampler[int]):
    """
    Task-guided sampler for imbalanced multi-task learning.

    Samples a task i with probability p_i, then samples a molecule that has
    a label for task i. This ensures that tasks with fewer labels are
    sampled more frequently than they would be in uniform sampling.

    Sampling probability schedule p_i:
        p_i ∝ n_i^(-α)

    where n_i is the number of labels for task i, and α ∈ [0, 1] controls
    the strength of the rebalancing.
    - α = 0: Uniform sampling across tasks (each task equally likely to be chosen).
    - α = 1: Inverse proportional sampling (rare tasks sampled much more).

    Parameters
    ----------
    targets : np.ndarray
        Target matrix of shape (N, T) where N is number of molecules and
        T is number of tasks. Missing labels should be NaN.
    alpha : float, default=0.5
        Exponent for inverse-power sampling schedule.
    num_samples : int, optional
        Number of samples to draw per epoch. If None, uses len(targets).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        targets: np.ndarray,
        alpha: float = 0.5,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(None)  # type: ignore
        self.targets = targets
        self.alpha = alpha
        self.num_samples = num_samples or len(targets)
        self.seed = seed

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

        # 1. Identify valid indices for each task
        self.num_tasks = targets.shape[1]
        self.task_indices: list[np.ndarray] = []
        task_counts = []

        for t in range(self.num_tasks):
            # Check if targets are floats or objects. Usually floats with NaN.
            # We assume standard Chemprop format where missing is NaN
            valid_mask = ~np.isnan(targets[:, t])
            indices = np.where(valid_mask)[0]
            self.task_indices.append(indices)
            task_counts.append(len(indices))

        self.task_counts = np.array(task_counts)

        # 2. Calculate task sampling probabilities
        # p_i ∝ n_i^(-α)
        # Add epsilon to avoid division by zero if a task has 0 samples
        # (though such tasks should probably be excluded or warned about)
        weights = np.power(self.task_counts + 1e-6, -self.alpha)
        self.task_probs = weights / np.sum(weights)

        logger.info("Task-aware sampler initialized with alpha=%.2f", alpha)
        logger.info("Task counts: %s", self.task_counts)
        logger.info("Task probabilities: %s", np.round(self.task_probs, 4))

    def __iter__(self) -> Iterator[int]:
        # Vectorized two-stage sampling:
        # Stage 1: Sample all tasks at once
        sampled_tasks = self.rng.choice(
            self.num_tasks,
            size=self.num_samples,
            p=self.task_probs,
            replace=True,
        )

        # Stage 2: Vectorized within-task sampling using bincount grouping
        # Count how many samples needed per task
        task_sample_counts = np.bincount(sampled_tasks, minlength=self.num_tasks)

        # Pre-allocate output array
        indices = np.empty(self.num_samples, dtype=np.int64)

        # Track position for each task's samples in the output
        task_positions = np.zeros(self.num_tasks, dtype=np.int64)
        current_pos = np.zeros(self.num_tasks, dtype=np.int64)

        # Create position mapping: for each task, where do its samples go?
        for i, task_idx in enumerate(sampled_tasks):
            if task_positions[task_idx] == 0 and current_pos[task_idx] == 0:
                task_positions[task_idx] = i
            current_pos[task_idx] += 1

        # Batch sample for each task that has samples to draw
        for task_idx in range(self.num_tasks):
            count = task_sample_counts[task_idx]
            if count == 0:
                continue

            valid_indices = self.task_indices[task_idx]
            if len(valid_indices) > 0:
                # Batch sample all molecules for this task at once
                sampled_mols = self.rng.choice(valid_indices, size=count, replace=True)
            else:
                # Fallback: sample random molecules from whole dataset
                sampled_mols = self.rng.integers(0, len(self.targets), size=count)

            # Place samples at their correct positions (where sampled_tasks == task_idx)
            task_mask = sampled_tasks == task_idx
            indices[task_mask] = sampled_mols

        yield from indices.tolist()

    def __len__(self) -> int:
        return self.num_samples
