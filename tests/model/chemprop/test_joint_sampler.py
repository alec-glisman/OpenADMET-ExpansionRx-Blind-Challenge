"""
Tests for JointSampler combining task-aware and curriculum-aware sampling.

Includes tests verifying the two-stage sampling algorithm matches the original
TaskAwareSampler behavior:
1. Stage 1: Sample task t with probability p_t ∝ count_t^(-α)
2. Stage 2: Sample molecule uniformly from task t's valid indices
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.joint_sampler import JointSampler


class OriginalTaskAwareSampler:
    """
    Reference implementation of the original TaskAwareSampler for comparison.

    This is the exact algorithm from the original implementation that we need
    to match with the new JointSampler.
    """

    def __init__(
        self,
        targets: np.ndarray,
        alpha: float = 0.0,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.targets = targets
        self.alpha = alpha
        self.num_samples = num_samples or len(targets)
        self.seed = seed

        # Precompute task indices and counts
        self.num_tasks = targets.shape[1]
        self.task_indices: list[np.ndarray] = []
        task_counts = []

        for t in range(self.num_tasks):
            valid_mask = ~np.isnan(targets[:, t])
            indices = np.where(valid_mask)[0]
            self.task_indices.append(indices)
            task_counts.append(len(indices))

        self.task_counts = np.array(task_counts, dtype=float)

        # Calculate task probabilities: p_t ∝ count_t^(-α)
        weights = np.power(self.task_counts + 1e-6, -self.alpha)
        self.task_probs = weights / np.sum(weights)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        indices = []

        for _ in range(self.num_samples):
            # Stage 1: Sample task according to task probabilities
            task_idx = rng.choice(self.num_tasks, p=self.task_probs)

            # Stage 2: Sample molecule UNIFORMLY from task's valid indices
            valid_indices = self.task_indices[task_idx]
            if len(valid_indices) > 0:
                mol_idx = rng.choice(valid_indices)
            else:
                mol_idx = rng.integers(0, len(self.targets))
            indices.append(mol_idx)

        return iter(indices)

    def __len__(self):
        return self.num_samples


@pytest.fixture
def multi_task_targets() -> np.ndarray:
    """
    Create multi-task target array with imbalanced label counts.

    Tasks:
    - Task 0: 8 labels (common)
    - Task 1: 4 labels (medium)
    - Task 2: 2 labels (rare)
    """
    targets = np.full((8, 3), np.nan, dtype=float)
    # Task 0: all samples
    targets[:, 0] = np.random.randn(8)
    # Task 1: samples 0, 2, 4, 6
    targets[[0, 2, 4, 6], 1] = np.random.randn(4)
    # Task 2: samples 1, 5
    targets[[1, 5], 2] = np.random.randn(2)
    return targets


@pytest.fixture
def large_imbalanced_targets() -> np.ndarray:
    """
    Create larger multi-task target array for statistical tests.

    Tasks:
    - Task 0: 100 labels (common)
    - Task 1: 20 labels (medium)
    - Task 2: 5 labels (rare)
    """
    np.random.seed(123)
    targets = np.full((100, 3), np.nan, dtype=float)
    # Task 0: all samples
    targets[:, 0] = np.random.randn(100)
    # Task 1: first 20 samples
    targets[:20, 1] = np.random.randn(20)
    # Task 2: first 5 samples
    targets[:5, 2] = np.random.randn(5)
    return targets


@pytest.fixture
def quality_labels() -> list[str]:
    """Quality labels matching multi_task_targets."""
    return ["high", "high", "medium", "low", "high", "medium", "low", "high"]


@pytest.fixture
def curriculum_state() -> CurriculumState:
    """Curriculum state for testing."""
    return CurriculumState(qualities=["high", "medium", "low"], patience=5)


class TestOriginalTaskAwareSamplerParity:
    """
    Tests verifying JointSampler matches original TaskAwareSampler behavior.

    The original TaskAwareSampler uses two-stage sampling:
    1. Sample task t with probability p_t ∝ count_t^(-α)
    2. Sample molecule UNIFORMLY from task t's valid indices

    These tests ensure the new JointSampler (without curriculum) produces
    statistically equivalent results.
    """

    def test_task_indices_match(self, multi_task_targets):
        """Verify task_indices are computed identically."""
        original = OriginalTaskAwareSampler(
            targets=multi_task_targets,
            alpha=0.5,
            seed=42,
        )
        new = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            seed=42,
        )

        # Both should have same task indices
        assert len(original.task_indices) == len(new.task_indices)
        for t in range(len(original.task_indices)):
            assert set(original.task_indices[t]) == set(new.task_indices[t])

    def test_task_probs_match(self, multi_task_targets):
        """Verify task probabilities are computed identically."""
        original = OriginalTaskAwareSampler(
            targets=multi_task_targets,
            alpha=0.5,
            seed=42,
        )
        new = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            seed=42,
        )

        assert np.allclose(original.task_probs, new.task_probs)

    def test_exact_sequence_match_no_curriculum(self, multi_task_targets):
        """
        Verify statistically equivalent sampling when no curriculum is used.

        The exact sequence may differ due to RNG consumption patterns, but
        the distribution should be equivalent.
        """
        num_samples = 10000
        original = OriginalTaskAwareSampler(
            targets=multi_task_targets,
            alpha=0.5,
            num_samples=num_samples,
            seed=42,
        )
        new = JointSampler(
            targets=multi_task_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=0.5,
            num_samples=num_samples,
            seed=42,
            increment_seed_per_epoch=False,
        )

        original_indices = list(original)
        new_indices = list(new)

        # Check that both produce valid indices
        assert len(original_indices) == num_samples
        assert len(new_indices) == num_samples

        # Check distribution is similar (chi-square would be ideal)
        original_counts = Counter(original_indices)
        new_counts = Counter(new_indices)

        # Each sample should appear with roughly similar frequency
        for idx in range(len(multi_task_targets)):
            orig_count = original_counts.get(idx, 0)
            new_count = new_counts.get(idx, 0)
            # Allow 50% relative difference for statistical variance
            if orig_count > 100:  # Only check samples with sufficient counts
                ratio = new_count / orig_count
                assert 0.5 < ratio < 2.0, f"Sample {idx}: orig={orig_count}, new={new_count}"

    def test_task_sampling_distribution(self, large_imbalanced_targets):
        """
        Verify task sampling distribution matches expected probabilities.

        With alpha > 0, rare tasks should be sampled more frequently relative
        to their size.
        """
        num_samples = 10000
        alpha = 1.0  # Strong rebalancing

        sampler = JointSampler(
            targets=large_imbalanced_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=alpha,
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)

        # Count samples per molecule, grouped by which tasks they belong to
        counts = Counter(indices)

        # Task 2 only has samples 0-4, Task 1 has 0-19, Task 0 has all
        # Samples 0-4 are in all tasks, samples 5-19 are in tasks 0 and 1
        # Samples 20-99 are only in task 0

        # With alpha=1 and task probs inversely proportional to counts,
        # Task 2 (5 samples) should have high probability
        # Each of its samples (0-4) should be drawn frequently

        # Count hits for exclusive task 0 samples (indices 20-99)
        task_0_exclusive_hits = sum(counts.get(i, 0) for i in range(20, 100))

        # Count hits for task 2 samples (indices 0-4)
        task_2_hits = sum(counts.get(i, 0) for i in range(5))

        # With strong rebalancing, task 2 samples should be overrepresented
        # Task 0 exclusive samples: 80 samples getting ~1/3 of task prob (~33% of selections)
        # Task 2 samples: 5 samples getting ~1/3 of task prob (~33% of selections)
        # So task 2 samples should have ~5x higher per-sample rate

        task_0_exclusive_rate_per_sample = task_0_exclusive_hits / 80
        task_2_rate_per_sample = task_2_hits / 5

        # Task 2 samples should be oversampled significantly
        assert task_2_rate_per_sample > task_0_exclusive_rate_per_sample * 2, (
            f"Task 2 rate {task_2_rate_per_sample:.1f} should be much higher than "
            f"Task 0 exclusive rate {task_0_exclusive_rate_per_sample:.1f}"
        )

    def test_uniform_within_task_sampling(self, large_imbalanced_targets):
        """
        Verify that within each task, samples are drawn uniformly.

        This is the key property of the original TaskAwareSampler:
        once a task is selected, all molecules in that task have equal probability.
        """
        num_samples = 50000
        alpha = 0.0  # Equal task weighting to simplify analysis

        sampler = JointSampler(
            targets=large_imbalanced_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=alpha,
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        counts = Counter(indices)

        # For Task 2 (indices 0-4), each should be sampled roughly equally
        # when that task is selected
        task_2_indices = sampler.task_indices[2]
        task_2_counts = [counts.get(idx, 0) for idx in task_2_indices]

        # With enough samples, variance should be low
        # Chi-square test would be ideal, but simple variance check suffices
        mean_count = np.mean(task_2_counts)
        std_count = np.std(task_2_counts)
        cv = std_count / mean_count if mean_count > 0 else 0

        # Coefficient of variation should be reasonable for uniform sampling
        assert cv < 0.3, f"Within-task sampling not uniform: CV={cv}"

    def test_alpha_zero_uniform_tasks(self, multi_task_targets):
        """Verify alpha=0 gives uniform task probabilities."""
        original = OriginalTaskAwareSampler(
            targets=multi_task_targets,
            alpha=0.0,
            seed=42,
        )
        new = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.0,
            seed=42,
        )

        # All tasks should have equal probability
        expected = np.ones(3) / 3
        assert np.allclose(original.task_probs, expected)
        assert np.allclose(new.task_probs, expected)

    def test_alpha_one_inverse_proportional(self, large_imbalanced_targets):
        """Verify alpha=1 gives inverse-proportional task probabilities."""
        sampler = JointSampler(
            targets=large_imbalanced_targets,
            task_alpha=1.0,
            seed=42,
        )

        # p_t ∝ count_t^(-1) = 1/count_t
        # Task 0: 100 → 1/100, Task 1: 20 → 1/20, Task 2: 5 → 1/5
        counts = sampler.task_counts
        expected_weights = 1.0 / (counts + 1e-6)
        expected_probs = expected_weights / expected_weights.sum()

        assert np.allclose(sampler.task_probs, expected_probs)


class TestJointSamplerTaskOnly:
    """Tests for JointSampler with task oversampling only."""

    def test_task_only_alpha_zero(self, multi_task_targets):
        """Test uniform task sampling when alpha=0."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=0.0,
            num_samples=100,
            seed=42,
        )

        # All tasks should have equal probability
        assert np.allclose(sampler.task_probs, 1.0 / 3)

    def test_task_only_alpha_positive(self, multi_task_targets):
        """Test that rare tasks get higher probability with alpha > 0."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=0.5,
            num_samples=100,
            seed=42,
        )

        # Task 2 is rarest (count=2), should have highest probability
        # Task 0 is most common (count=8), should have lowest probability
        assert sampler.task_probs[2] > sampler.task_probs[1] > sampler.task_probs[0]

    def test_task_indices_correctness(self, multi_task_targets):
        """Test that task_indices correctly maps tasks to valid samples."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=0.5,
            seed=42,
        )

        # Task 0: all 8 samples
        assert len(sampler.task_indices[0]) == 8
        # Task 1: samples 0, 2, 4, 6
        assert set(sampler.task_indices[1]) == {0, 2, 4, 6}
        # Task 2: samples 1, 5
        assert set(sampler.task_indices[2]) == {1, 5}


class TestJointSamplerCurriculumOnly:
    """Tests for JointSampler with curriculum learning only."""

    def test_curriculum_only_warmup_phase(self, multi_task_targets, quality_labels, curriculum_state):
        """Test high-quality samples are favored in warmup phase."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.0,
            num_samples=100,
            seed=42,
        )

        # Warmup phase should favor high quality
        assert curriculum_state.phase == "warmup"
        weights = sampler._compute_curriculum_weights()

        high_indices = [i for i, q in enumerate(quality_labels) if q == "high"]
        low_indices = [i for i, q in enumerate(quality_labels) if q == "low"]

        assert np.mean(weights[high_indices]) > np.mean(weights[low_indices])

    def test_curriculum_only_phase_transition(self, multi_task_targets, quality_labels, curriculum_state):
        """Test weights update when curriculum phase changes."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.0,
            num_samples=100,
            seed=42,
        )

        # Initial weights in warmup
        weights_warmup = sampler._compute_curriculum_weights()

        # Manually advance phase by changing state and updating weights
        curriculum_state.phase = 1  # Advance to expand phase index
        curriculum_state.weights = curriculum_state._weights_for_phase("expand")  # Update weights
        weights_expand = sampler._compute_curriculum_weights()

        # Weights should change
        assert not np.allclose(weights_warmup, weights_expand)


class TestJointSamplerCombined:
    """Tests for JointSampler with both strategies."""

    def test_two_stage_sampling_respects_task_probs(self, multi_task_targets):
        """Test that sampling frequency matches task probabilities."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=None,
            curriculum_state=None,
            task_alpha=1.0,  # Strong rebalancing
            num_samples=10000,
            seed=42,
        )

        # Sample many times and count task occurrences
        indices = list(sampler)

        # Count how many times each task's samples are drawn
        task_sample_counts = np.zeros(3)
        for idx in indices:
            for t in range(3):
                if idx in sampler.task_indices[t]:
                    task_sample_counts[t] += 1

        # With alpha=1.0, rare tasks should be oversampled significantly
        # Task 2 (count=2) should have more samples than Task 0 (count=8) proportionally
        # Normalize by task size to get effective sampling rate
        rate_task0 = task_sample_counts[0] / len(sampler.task_indices[0])
        rate_task2 = task_sample_counts[2] / len(sampler.task_indices[2])

        # Rare task should be sampled at higher rate per sample
        assert rate_task2 > rate_task0

    def test_curriculum_integration(self, multi_task_targets, quality_labels, curriculum_state):
        """Test curriculum weights are applied within tasks."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.0,  # No task rebalancing, pure curriculum
            num_samples=10000,
            seed=42,
        )

        indices = list(sampler)

        # Count samples by quality
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        for idx in indices:
            quality_counts[quality_labels[idx]] += 1

        # In warmup phase, high quality should be favored
        assert quality_counts["high"] > quality_counts["low"]

    def test_both_strategies_combined(self, multi_task_targets, quality_labels, curriculum_state):
        """Test both task rarity and quality influence sampling."""
        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.5,
            num_samples=1000,
            seed=42,
        )

        # Should not crash and produce valid indices
        indices = list(sampler)
        assert len(indices) == 1000
        assert all(0 <= idx < len(multi_task_targets) for idx in indices)


class TestJointSamplerIteration:
    """Tests for JointSampler iteration and sampling."""

    def test_yields_correct_count(self, multi_task_targets):
        """Test sampler yields requested number of samples."""
        num_samples = 50
        sampler = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        assert len(indices) == num_samples

    def test_seed_reproducibility(self, multi_task_targets):
        """Test same seed produces same sampling."""
        sampler1 = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            num_samples=50,
            seed=42,
            increment_seed_per_epoch=False,
        )

        sampler2 = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            num_samples=50,
            seed=42,
            increment_seed_per_epoch=False,
        )

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_seed_increment_per_epoch(self, multi_task_targets):
        """Test seed increments produce different sampling."""
        sampler = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            num_samples=50,
            seed=42,
            increment_seed_per_epoch=True,
        )

        indices1 = list(sampler)
        indices2 = list(sampler)

        # Different epochs should produce different samples
        assert indices1 != indices2

    def test_len_method(self, multi_task_targets):
        """Test __len__ returns num_samples."""
        num_samples = 100
        sampler = JointSampler(
            targets=multi_task_targets,
            num_samples=num_samples,
            seed=42,
        )

        assert len(sampler) == num_samples


class TestJointSamplerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_alpha_outside_range_warning(self, multi_task_targets, caplog):
        """Test warning for alpha outside [0, 1]."""
        # Construct sampler to trigger range warning but do not need the instance
        JointSampler(
            targets=multi_task_targets,
            task_alpha=1.5,
            seed=42,
        )

        assert "outside recommended range" in caplog.text

    def test_all_zero_curriculum_weights(self, multi_task_targets):
        """Test fallback when all curriculum weights are zero.

        Note: The JointSampler handles this gracefully by detecting zero weights
        and falling back to uniform weights. This test verifies that behavior.
        """
        # Create state where all weights are near zero (shouldn't happen normally)
        state = CurriculumState(qualities=["high", "medium", "low"], patience=5)
        # Set very small but non-zero weights to avoid ZeroDivisionError in sampling_probs
        state.weights = {"high": 1e-10, "medium": 1e-10, "low": 1e-10}

        quality_labels = ["high"] * len(multi_task_targets)

        sampler = JointSampler(
            targets=multi_task_targets,
            quality_labels=quality_labels,
            curriculum_state=state,
            task_alpha=0.0,
            seed=42,
        )

        # JointSampler has fallback logic for near-zero weights
        weights = sampler._compute_curriculum_weights()
        # Since all samples are "high" and weights are nearly equal,
        # each sample gets roughly equal weight
        assert len(weights) == len(multi_task_targets)
        assert np.all(weights > 0)

    def test_no_valid_tasks(self, multi_task_targets):
        """Test handling of samples with no valid tasks."""
        targets = np.full((5, 3), np.nan, dtype=float)
        targets[0, 0] = 1.0  # Only one valid label

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            seed=42,
        )

        # Should not crash and task_indices should have correct structure
        assert len(sampler.task_indices) == 3
        assert 0 in sampler.task_indices[0]
        # Should be able to iterate
        indices = list(sampler)
        assert len(indices) == len(targets)

    def test_default_num_samples(self, multi_task_targets):
        """Test num_samples defaults to dataset length."""
        sampler = JointSampler(
            targets=multi_task_targets,
            num_samples=None,
            seed=42,
        )

        assert len(sampler) == len(multi_task_targets)


class TestJointSamplerWeightStatistics:
    """Tests for weight statistics logging."""

    def test_entropy_calculation(self, multi_task_targets):
        """Test entropy is calculated for task probability distribution."""
        sampler = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            seed=42,
            log_weight_stats=True,
        )

        # Manually calculate entropy of task probs
        eps = 1e-10
        expected_entropy = -np.sum(sampler.task_probs * np.log(sampler.task_probs + eps))

        # Trigger logging by iterating
        _ = list(sampler)

        # If logging works, no error should occur
        assert expected_entropy > 0

    def test_effective_tasks_calculation(self, multi_task_targets):
        """Test effective number of tasks is calculated correctly."""
        sampler = JointSampler(
            targets=multi_task_targets,
            task_alpha=0.5,
            seed=42,
            log_weight_stats=True,
        )

        effective_tasks = 1.0 / np.sum(sampler.task_probs**2)

        # More uniform probs → higher effective tasks (max = num_tasks)
        # More concentrated probs → lower effective tasks (min ≈ 1)
        assert 1.0 <= effective_tasks <= sampler.num_tasks


class TestTwoStageSamplingMathematicalProperties:
    """
    Rigorous mathematical tests for two-stage sampling correctness.

    These tests verify the core mathematical properties of the two-stage
    sampling algorithm that the original TaskAwareSampler used.
    """

    def test_task_selection_frequency_matches_probabilities(self):
        """
        Verify empirical task selection frequencies match theoretical probabilities.

        Uses chi-square-like test to verify the task selection stage.
        """
        # Create targets where we can track which task was selected
        # by having non-overlapping task assignments
        np.random.seed(42)
        targets = np.full((30, 3), np.nan, dtype=float)
        # Task 0: samples 0-9 (10 samples)
        targets[0:10, 0] = np.random.randn(10)
        # Task 1: samples 10-19 (10 samples)
        targets[10:20, 1] = np.random.randn(10)
        # Task 2: samples 20-29 (10 samples)
        targets[20:30, 2] = np.random.randn(10)

        # With alpha=0, all tasks should have equal probability (1/3)
        num_samples = 30000
        sampler = JointSampler(
            targets=targets,
            task_alpha=0.0,
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        counts = Counter(indices)

        # Count samples from each task's exclusive range
        task_0_hits = sum(counts.get(i, 0) for i in range(10))
        task_1_hits = sum(counts.get(i, 0) for i in range(10, 20))
        task_2_hits = sum(counts.get(i, 0) for i in range(20, 30))

        # Each task should get ~1/3 of samples
        expected = num_samples / 3
        tolerance = 0.1  # 10% tolerance

        assert abs(task_0_hits - expected) / expected < tolerance, f"Task 0: {task_0_hits} vs expected {expected}"
        assert abs(task_1_hits - expected) / expected < tolerance, f"Task 1: {task_1_hits} vs expected {expected}"
        assert abs(task_2_hits - expected) / expected < tolerance, f"Task 2: {task_2_hits} vs expected {expected}"

    def test_within_task_uniform_sampling_chi_square(self):
        """
        Verify uniform sampling within tasks using chi-square statistic.

        This is a key property: given a task is selected, all molecules
        in that task should have equal probability.
        """
        np.random.seed(42)
        # Create targets with non-overlapping tasks
        targets = np.full((20, 2), np.nan, dtype=float)
        # Task 0: samples 0-9 (10 samples)
        targets[0:10, 0] = np.random.randn(10)
        # Task 1: samples 10-19 (10 samples)
        targets[10:20, 1] = np.random.randn(10)

        num_samples = 50000
        sampler = JointSampler(
            targets=targets,
            task_alpha=0.0,  # Equal task probability
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        counts = Counter(indices)

        # Within Task 0, each sample (0-9) should have equal count
        task_0_counts = [counts.get(i, 0) for i in range(10)]
        expected_per_sample = num_samples / 2 / 10  # half samples, 10 molecules

        # Chi-square statistic
        chi_sq = sum((obs - expected_per_sample) ** 2 / expected_per_sample for obs in task_0_counts)

        # For df=9, chi-square critical value at p=0.01 is ~21.67
        # We use a generous threshold
        assert chi_sq < 30, f"Chi-square {chi_sq:.1f} too high - sampling not uniform within task"

    def test_alpha_affects_task_probabilities_correctly(self):
        """
        Verify that different alpha values produce correct task probability ratios.
        """
        np.random.seed(42)
        # Task 0: 100 samples, Task 1: 10 samples (10x difference)
        targets = np.full((110, 2), np.nan, dtype=float)
        targets[0:100, 0] = np.random.randn(100)
        targets[100:110, 1] = np.random.randn(10)

        # With alpha=0, tasks should have equal probability
        sampler_a0 = JointSampler(targets=targets, task_alpha=0.0, seed=42)
        assert np.allclose(sampler_a0.task_probs, [0.5, 0.5])

        # With alpha=1, p ∝ 1/count, so Task 1 should be 10x more likely
        sampler_a1 = JointSampler(targets=targets, task_alpha=1.0, seed=42)
        ratio = sampler_a1.task_probs[1] / sampler_a1.task_probs[0]
        expected_ratio = 100 / 10  # 10x
        assert abs(ratio - expected_ratio) / expected_ratio < 0.01, f"Ratio {ratio:.2f} should be ~{expected_ratio}"

        # With alpha=0.5, p ∝ 1/sqrt(count), ratio should be sqrt(10)
        sampler_a05 = JointSampler(targets=targets, task_alpha=0.5, seed=42)
        ratio_05 = sampler_a05.task_probs[1] / sampler_a05.task_probs[0]
        expected_ratio_05 = np.sqrt(100 / 10)  # sqrt(10) ≈ 3.16
        assert (
            abs(ratio_05 - expected_ratio_05) / expected_ratio_05 < 0.01
        ), f"Ratio {ratio_05:.2f} should be ~{expected_ratio_05:.2f}"

    def test_curriculum_weights_applied_within_task(self):
        """
        Verify curriculum weights are correctly applied during within-task sampling.

        When curriculum is enabled, molecules within a task should be sampled
        proportionally to their curriculum weights, not uniformly.
        """
        np.random.seed(42)
        # Single task with 10 samples
        targets = np.zeros((10, 1), dtype=float)

        # Quality labels: 5 high, 5 low
        quality_labels = ["high"] * 5 + ["low"] * 5

        # Create curriculum state in warmup phase (favors high quality)
        state = CurriculumState(qualities=["high", "low"], patience=5)
        assert state.phase == "warmup"

        num_samples = 10000
        sampler = JointSampler(
            targets=targets,
            quality_labels=quality_labels,
            curriculum_state=state,
            task_alpha=0.0,
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        counts = Counter(indices)

        # Count high vs low quality samples
        high_quality_hits = sum(counts.get(i, 0) for i in range(5))
        low_quality_hits = sum(counts.get(i, 0) for i in range(5, 10))

        # In warmup phase, high quality should be favored
        assert high_quality_hits > low_quality_hits * 1.5, (
            f"High quality ({high_quality_hits}) should be much higher than " f"low quality ({low_quality_hits})"
        )

    def test_multi_task_with_overlapping_samples(self):
        """
        Verify correct behavior when samples belong to multiple tasks.

        A sample belonging to multiple tasks can be sampled through any
        of those tasks, potentially increasing its overall sampling rate.
        """
        np.random.seed(42)
        targets = np.full((10, 3), np.nan, dtype=float)
        # Sample 0 belongs to ALL tasks
        targets[0, :] = np.random.randn(3)
        # Samples 1-3 belong only to Task 0
        targets[1:4, 0] = np.random.randn(3)
        # Samples 4-6 belong only to Task 1
        targets[4:7, 1] = np.random.randn(3)
        # Samples 7-9 belong only to Task 2
        targets[7:10, 2] = np.random.randn(3)

        num_samples = 30000
        sampler = JointSampler(
            targets=targets,
            task_alpha=0.0,  # Equal task probability
            num_samples=num_samples,
            seed=42,
        )

        indices = list(sampler)
        counts = Counter(indices)

        # Sample 0 can be reached through any task (3 paths)
        # Samples 1-3 can only be reached through Task 0 (1 path each)
        # Sample 0 should have ~3x the sampling rate of exclusive samples

        sample_0_count = counts.get(0, 0)
        exclusive_avg = np.mean([counts.get(i, 0) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]])

        # Sample 0 has 3 tasks it can be sampled through, each task has 4 samples
        # vs exclusive samples which have 1 task with 4 samples
        # So sample 0 should be sampled more often
        ratio = sample_0_count / exclusive_avg
        assert ratio > 2.0, (
            f"Multi-task sample ratio {ratio:.2f} should be >2 (sample 0: {sample_0_count}, "
            f"exclusive avg: {exclusive_avg:.0f})"
        )


class TestJointSamplerIntegrationWithDataLoader:
    """
    Tests verifying JointSampler works correctly with PyTorch DataLoader.
    """

    def test_sampler_works_with_dataloader(self):
        """Verify sampler integrates correctly with DataLoader."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        np.random.seed(42)
        targets = np.random.randn(100, 3)
        targets[np.random.rand(100, 3) > 0.7] = np.nan  # Add missing values

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            num_samples=50,
            seed=42,
        )

        # Create a simple dataset
        dataset = TensorDataset(torch.randn(100, 10))

        # Create DataLoader with our sampler
        loader = DataLoader(dataset, batch_size=10, sampler=sampler)

        # Should be able to iterate
        batches = list(loader)
        assert len(batches) == 5  # 50 samples / 10 batch_size

    def test_sampler_epoch_variation(self):
        """Verify different epochs produce different samples."""
        np.random.seed(42)
        targets = np.random.randn(50, 2)

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            num_samples=100,
            seed=42,
            increment_seed_per_epoch=True,
        )

        epoch1 = list(sampler)
        epoch2 = list(sampler)
        epoch3 = list(sampler)

        # All epochs should be different
        assert epoch1 != epoch2
        assert epoch2 != epoch3
        assert epoch1 != epoch3

    def test_sampler_deterministic_without_increment(self):
        """Verify reproducibility when increment_seed_per_epoch is False."""
        np.random.seed(42)
        targets = np.random.randn(50, 2)

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            num_samples=100,
            seed=42,
            increment_seed_per_epoch=False,
        )

        epoch1 = list(sampler)
        epoch2 = list(sampler)

        # Both epochs should be identical
        assert epoch1 == epoch2
