"""
Tests for TaskAwareSampler with vectorization regression tests.

This module verifies that the optimized vectorized implementation
produces statistically equivalent results to a reference non-vectorized
implementation.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np
import pytest

from admet.model.chemprop.task_sampler import TaskAwareSampler


class ReferenceTaskAwareSamplerNonVectorized:
    """
    Non-vectorized reference implementation for regression testing.

    This implements the original per-sample loop algorithm that the optimized
    TaskAwareSampler replaces. Used to verify the vectorized implementation
    produces statistically equivalent results.
    """

    def __init__(
        self,
        targets: np.ndarray,
        alpha: float = 0.5,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.targets = targets
        self.alpha = alpha
        self.num_samples = num_samples or len(targets)
        self.seed = seed

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

        self.num_tasks = targets.shape[1]
        self.task_indices: list[np.ndarray] = []
        task_counts = []

        for t in range(self.num_tasks):
            valid_mask = ~np.isnan(targets[:, t])
            indices = np.where(valid_mask)[0]
            self.task_indices.append(indices)
            task_counts.append(len(indices))

        self.task_counts = np.array(task_counts)

        weights = np.power(self.task_counts + 1e-6, -self.alpha)
        self.task_probs = weights / np.sum(weights)

    def __iter__(self) -> Iterator[int]:
        """Original per-sample loop implementation."""
        sampled_tasks = self.rng.choice(
            self.num_tasks,
            size=self.num_samples,
            p=self.task_probs,
            replace=True,
        )

        for task_idx in sampled_tasks:
            valid_indices = self.task_indices[task_idx]
            if len(valid_indices) > 0:
                mol_idx = self.rng.choice(valid_indices)
                yield int(mol_idx)
            else:
                yield int(self.rng.integers(0, len(self.targets)))

    def __len__(self) -> int:
        return self.num_samples


@pytest.fixture
def multi_task_targets():
    """Create multi-task target matrix with varying label availability."""
    np.random.seed(42)
    n_samples = 200
    n_tasks = 4

    targets = np.full((n_samples, n_tasks), np.nan)
    # Task 0: 80% coverage (common task)
    mask0 = np.random.rand(n_samples) < 0.8
    targets[mask0, 0] = np.random.randn(mask0.sum())
    # Task 1: 50% coverage (medium task)
    mask1 = np.random.rand(n_samples) < 0.5
    targets[mask1, 1] = np.random.randn(mask1.sum())
    # Task 2: 20% coverage (rare task)
    mask2 = np.random.rand(n_samples) < 0.2
    targets[mask2, 2] = np.random.randn(mask2.sum())
    # Task 3: 10% coverage (very rare task)
    mask3 = np.random.rand(n_samples) < 0.1
    targets[mask3, 3] = np.random.randn(mask3.sum())

    return targets


@pytest.fixture
def large_imbalanced_targets():
    """Create larger dataset with highly imbalanced task coverage."""
    np.random.seed(123)
    n_samples = 1000
    n_tasks = 5

    targets = np.full((n_samples, n_tasks), np.nan)
    coverages = [0.9, 0.5, 0.2, 0.05, 0.01]

    for t, cov in enumerate(coverages):
        mask = np.random.rand(n_samples) < cov
        targets[mask, t] = np.random.randn(mask.sum())

    return targets


class TestTaskAwareSamplerBasic:
    """Basic functionality tests for TaskAwareSampler."""

    def test_initialization(self, multi_task_targets):
        """Test sampler initializes correctly."""
        sampler = TaskAwareSampler(multi_task_targets, alpha=0.5, seed=42)

        assert sampler.num_tasks == 4
        assert len(sampler) == len(multi_task_targets)
        assert sampler.alpha == 0.5
        assert np.isclose(np.sum(sampler.task_probs), 1.0)

    def test_yields_correct_count(self, multi_task_targets):
        """Test sampler yields exactly num_samples indices."""
        num_samples = 500
        sampler = TaskAwareSampler(multi_task_targets, num_samples=num_samples, seed=42)

        indices = list(sampler)

        assert len(indices) == num_samples

    def test_indices_in_valid_range(self, multi_task_targets):
        """Test all sampled indices are within valid range."""
        sampler = TaskAwareSampler(multi_task_targets, num_samples=1000, seed=42)

        indices = list(sampler)

        assert all(0 <= idx < len(multi_task_targets) for idx in indices)

    def test_seed_reproducibility(self, multi_task_targets):
        """Test same seed produces identical samples."""
        sampler1 = TaskAwareSampler(multi_task_targets, num_samples=500, seed=42)
        sampler2 = TaskAwareSampler(multi_task_targets, num_samples=500, seed=42)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_different_seeds_produce_different_samples(self, multi_task_targets):
        """Test different seeds produce different samples."""
        sampler1 = TaskAwareSampler(multi_task_targets, num_samples=500, seed=42)
        sampler2 = TaskAwareSampler(multi_task_targets, num_samples=500, seed=123)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 != indices2


class TestTaskAwareSamplerAlpha:
    """Tests for alpha parameter effect on sampling."""

    def test_alpha_zero_uniform_tasks(self, multi_task_targets):
        """Test alpha=0 gives uniform task sampling."""
        sampler = TaskAwareSampler(multi_task_targets, alpha=0.0, seed=42)

        # All task probabilities should be equal
        expected_prob = 1.0 / sampler.num_tasks
        assert np.allclose(sampler.task_probs, expected_prob, atol=1e-6)

    def test_alpha_one_inverse_proportional(self, large_imbalanced_targets):
        """Test alpha=1 gives inverse proportional sampling."""
        sampler = TaskAwareSampler(large_imbalanced_targets, alpha=1.0, seed=42)

        # Rare tasks should have higher sampling probability
        # Task with fewer samples should have higher probability
        for i in range(len(sampler.task_counts) - 1):
            if sampler.task_counts[i] > sampler.task_counts[i + 1]:
                assert sampler.task_probs[i] < sampler.task_probs[i + 1]

    def test_alpha_affects_task_distribution(self, large_imbalanced_targets):
        """Test different alpha values produce different task distributions."""
        num_samples = 10000

        sampler_0 = TaskAwareSampler(large_imbalanced_targets, alpha=0.0, num_samples=num_samples, seed=42)
        sampler_1 = TaskAwareSampler(large_imbalanced_targets, alpha=1.0, num_samples=num_samples, seed=42)

        # These should have different task probability distributions
        assert not np.allclose(sampler_0.task_probs, sampler_1.task_probs)


class TestTaskAwareSamplerVectorizationRegression:
    """
    Regression tests verifying vectorized implementation matches reference.

    These tests ensure the optimized vectorized implementation produces
    statistically equivalent results to the original per-sample loop.
    """

    @pytest.fixture
    def regression_test_data(self):
        """Create test data for regression tests."""
        np.random.seed(42)
        n_samples = 500
        n_tasks = 4

        targets = np.full((n_samples, n_tasks), np.nan)
        coverages = [0.8, 0.5, 0.2, 0.1]

        for t, cov in enumerate(coverages):
            mask = np.random.rand(n_samples) < cov
            targets[mask, t] = np.random.randn(mask.sum())

        return targets

    def test_task_distribution_matches_reference(self, regression_test_data):
        """Verify task sampling distribution matches reference implementation."""
        targets = regression_test_data
        num_samples = 50000
        seed = 42

        optimized = TaskAwareSampler(targets, num_samples=num_samples, seed=seed)
        reference = ReferenceTaskAwareSamplerNonVectorized(targets, num_samples=num_samples, seed=seed)

        opt_indices = list(optimized)
        ref_indices = list(reference)

        # Count which task each sample belongs to
        def count_tasks(indices, targets):
            task_counts = Counter()
            for idx in indices:
                for t in range(targets.shape[1]):
                    if not np.isnan(targets[idx, t]):
                        task_counts[t] += 1
                        break
            return task_counts

        opt_task_counts = count_tasks(opt_indices, targets)
        ref_task_counts = count_tasks(ref_indices, targets)

        # Both should have similar task distributions (within 2% tolerance)
        for t in range(targets.shape[1]):
            opt_prop = opt_task_counts.get(t, 0) / num_samples
            ref_prop = ref_task_counts.get(t, 0) / num_samples
            assert abs(opt_prop - ref_prop) < 0.02, (
                f"Task {t} distribution differs: optimized={opt_prop:.4f}, " f"reference={ref_prop:.4f}"
            )

    def test_sample_range_valid(self, regression_test_data):
        """Verify all sampled indices are valid."""
        targets = regression_test_data
        num_samples = 10000

        sampler = TaskAwareSampler(targets, num_samples=num_samples, seed=42)
        indices = list(sampler)

        assert len(indices) == num_samples
        assert all(0 <= idx < len(targets) for idx in indices)

    def test_task_probs_match_reference(self, regression_test_data):
        """Verify task probabilities match reference implementation."""
        targets = regression_test_data

        optimized = TaskAwareSampler(targets, seed=42)
        reference = ReferenceTaskAwareSamplerNonVectorized(targets, seed=42)

        assert np.allclose(optimized.task_probs, reference.task_probs)
        assert np.array_equal(optimized.task_counts, reference.task_counts)

    def test_task_indices_match_reference(self, regression_test_data):
        """Verify task indices match reference implementation."""
        targets = regression_test_data

        optimized = TaskAwareSampler(targets, seed=42)
        reference = ReferenceTaskAwareSamplerNonVectorized(targets, seed=42)

        for t in range(targets.shape[1]):
            assert np.array_equal(optimized.task_indices[t], reference.task_indices[t])

    def test_within_task_uniform_sampling(self, regression_test_data):
        """Verify within-task sampling is approximately uniform."""
        targets = regression_test_data
        num_samples = 100000

        # Use alpha=0 so all tasks are equally likely
        sampler = TaskAwareSampler(targets, alpha=0.0, num_samples=num_samples, seed=42)
        indices = list(sampler)

        # For each task, check if its molecules are sampled uniformly
        for t in range(targets.shape[1]):
            task_valid_indices = set(sampler.task_indices[t])
            if len(task_valid_indices) < 10:
                continue

            # Count samples that fall in this task's valid indices
            task_samples = [idx for idx in indices if idx in task_valid_indices]

            if len(task_samples) > 500:
                sample_counts = Counter(task_samples)
                counts = list(sample_counts.values())

                # Check coefficient of variation is reasonable (< 1.0 for uniform with replacement)
                # Uniform sampling with replacement has CV ≈ 1/sqrt(n) for n samples per item
                if len(counts) > 1:
                    cv = np.std(counts) / np.mean(counts)
                    assert cv < 1.0, f"Task {t} within-task sampling not uniform: CV={cv}"

    def test_determinism_with_same_seed(self, regression_test_data):
        """Verify same seed produces identical results."""
        targets = regression_test_data
        num_samples = 1000

        sampler1 = TaskAwareSampler(targets, num_samples=num_samples, seed=42)
        sampler2 = TaskAwareSampler(targets, num_samples=num_samples, seed=42)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2, "Same seed should produce identical samples"


class TestTaskAwareSamplerPerformance:
    """Performance benchmarks for TaskAwareSampler."""

    @pytest.mark.benchmark
    def test_large_scale_iteration_time(self, large_imbalanced_targets):
        """Benchmark iteration time for large number of samples."""
        import time

        num_samples = 100000
        sampler = TaskAwareSampler(large_imbalanced_targets, num_samples=num_samples, seed=42)

        start = time.perf_counter()
        indices = list(sampler)
        elapsed = time.perf_counter() - start

        assert len(indices) == num_samples
        # Should complete in < 0.5 seconds for 100k samples
        assert elapsed < 0.5, f"Iteration took {elapsed:.3f}s, expected < 0.5s"

    @pytest.mark.benchmark
    def test_vectorized_vs_reference_speedup(self, large_imbalanced_targets):
        """Verify vectorized implementation is faster than reference."""
        import time

        num_samples = 50000
        targets = large_imbalanced_targets

        # Time optimized implementation
        optimized = TaskAwareSampler(targets, num_samples=num_samples, seed=42)
        start = time.perf_counter()
        _ = list(optimized)
        opt_time = time.perf_counter() - start

        # Time reference implementation
        reference = ReferenceTaskAwareSamplerNonVectorized(targets, num_samples=num_samples, seed=42)
        start = time.perf_counter()
        _ = list(reference)
        ref_time = time.perf_counter() - start

        # Optimized should be at least 2x faster
        speedup = ref_time / opt_time if opt_time > 0 else float("inf")
        assert speedup > 2.0, f"Speedup only {speedup:.2f}x (opt={opt_time:.4f}s, ref={ref_time:.4f}s)"
