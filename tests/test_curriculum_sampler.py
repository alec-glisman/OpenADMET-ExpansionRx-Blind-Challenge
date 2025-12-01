"""
Unit tests for admet.model.chemprop.curriculum_sampler module.

Tests the weighted sampling functionality for curriculum learning.
"""

import warnings

import numpy as np
import pytest

from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.curriculum_sampler import (
    build_curriculum_sampler,
    compute_per_quality_weights,
    get_quality_indices,
)


class TestBuildCurriculumSampler:
    """Tests for build_curriculum_sampler function."""

    def test_basic_sampler_creation(self) -> None:
        """Test creating a sampler with basic inputs."""
        quality_labels = ["high", "high", "medium", "low", "high"]
        state = CurriculumState(qualities=["high", "medium", "low"])

        sampler = build_curriculum_sampler(quality_labels, state, seed=42)

        assert sampler is not None
        assert len(sampler) == len(quality_labels)

    def test_warmup_phase_favors_high_quality(self) -> None:
        """Test that warmup phase samples more high-quality data."""
        # Create dataset with equal distribution
        quality_labels = ["high"] * 100 + ["medium"] * 100 + ["low"] * 100
        state = CurriculumState(qualities=["high", "medium", "low"])
        state.phase = "warmup"
        state.weights = state._weights_for_phase("warmup")

        sampler = build_curriculum_sampler(quality_labels, state, seed=42)

        # Sample many indices and count quality distribution
        sample_count = {"high": 0, "medium": 0, "low": 0}
        for idx in sampler:
            sample_count[quality_labels[idx]] += 1

        # In warmup phase, high should dominate (weight=0.9 vs 0.1 for medium)
        assert sample_count["high"] > sample_count["medium"]
        assert sample_count["high"] > sample_count["low"]

    def test_robust_phase_includes_low_quality(self) -> None:
        """Test that robust phase includes low-quality samples."""
        quality_labels = ["high"] * 100 + ["medium"] * 100 + ["low"] * 100
        state = CurriculumState(qualities=["high", "medium", "low"])
        state.phase = "robust"
        state.weights = state._weights_for_phase("robust")

        sampler = build_curriculum_sampler(quality_labels, state, seed=42)

        sample_count = {"high": 0, "medium": 0, "low": 0}
        for idx in sampler:
            sample_count[quality_labels[idx]] += 1

        # In robust phase, low quality should have non-trivial representation
        assert sample_count["low"] > 0
        # All qualities should be sampled
        assert sample_count["high"] > 0
        assert sample_count["medium"] > 0

    def test_empty_quality_labels_raises(self) -> None:
        """Test that empty quality labels raises ValueError."""
        state = CurriculumState(qualities=["high", "medium", "low"])

        with pytest.raises(ValueError, match="cannot be empty"):
            build_curriculum_sampler([], state)

    def test_unknown_quality_warning(self) -> None:
        """Test that unknown quality labels trigger warning."""
        quality_labels = ["high", "unknown_quality", "medium"]
        state = CurriculumState(qualities=["high", "medium", "low"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sampler = build_curriculum_sampler(quality_labels, state, seed=42)
            # Should warn about unknown quality
            assert len(w) >= 1
            assert "unknown_quality" in str(w[0].message)

    def test_all_zero_weights_fallback(self) -> None:
        """Test fallback to uniform sampling when all weights are zero."""
        # All labels are unknown, so all weights would be zero
        quality_labels = ["unknown1", "unknown2", "unknown3"]
        state = CurriculumState(qualities=["high", "medium", "low"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sampler = build_curriculum_sampler(quality_labels, state, seed=42)
            # Should warn about zero weights AND unknown qualities
            assert len(w) >= 1

        # Sampler should still work (uniform fallback)
        assert sampler is not None
        assert len(sampler) == 3

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same sampling order."""
        quality_labels = ["high", "medium", "low", "high", "medium"]
        state = CurriculumState(qualities=["high", "medium", "low"])

        sampler1 = build_curriculum_sampler(quality_labels, state, seed=42)
        sampler2 = build_curriculum_sampler(quality_labels, state, seed=42)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_different_seeds_produce_different_samples(self) -> None:
        """Test that different seeds produce different sampling orders."""
        quality_labels = ["high"] * 50 + ["medium"] * 50
        state = CurriculumState(qualities=["high", "medium", "low"])

        sampler1 = build_curriculum_sampler(quality_labels, state, seed=42)
        sampler2 = build_curriculum_sampler(quality_labels, state, seed=123)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        # Very unlikely to be identical with different seeds
        assert indices1 != indices2

    def test_custom_num_samples(self) -> None:
        """Test specifying custom number of samples."""
        quality_labels = ["high", "medium", "low"]
        state = CurriculumState(qualities=["high", "medium", "low"])

        sampler = build_curriculum_sampler(quality_labels, state, num_samples=100, seed=42)

        assert len(sampler) == 100


class TestGetQualityIndices:
    """Tests for get_quality_indices function."""

    def test_basic_indices(self) -> None:
        """Test getting indices for each quality level."""
        quality_labels = ["high", "medium", "high", "low", "medium"]
        qualities = ["high", "medium", "low"]

        indices = get_quality_indices(quality_labels, qualities)

        assert indices["high"] == [0, 2]
        assert indices["medium"] == [1, 4]
        assert indices["low"] == [3]

    def test_missing_quality_level(self) -> None:
        """Test when a quality level has no samples."""
        quality_labels = ["high", "high", "medium"]
        qualities = ["high", "medium", "low"]

        indices = get_quality_indices(quality_labels, qualities)

        assert indices["high"] == [0, 1]
        assert indices["medium"] == [2]
        assert indices["low"] == []  # No low quality samples

    def test_empty_labels(self) -> None:
        """Test with empty quality labels."""
        indices = get_quality_indices([], ["high", "medium", "low"])

        assert indices["high"] == []
        assert indices["medium"] == []
        assert indices["low"] == []

    def test_single_quality(self) -> None:
        """Test with only one quality level."""
        quality_labels = ["high", "high", "high"]
        qualities = ["high"]

        indices = get_quality_indices(quality_labels, qualities)

        assert indices["high"] == [0, 1, 2]


class TestComputePerQualityWeights:
    """Tests for compute_per_quality_weights function."""

    def test_warmup_weights(self) -> None:
        """Test effective weights in warmup phase."""
        quality_labels = ["high"] * 100 + ["medium"] * 100 + ["low"] * 100
        state = CurriculumState(qualities=["high", "medium", "low"])
        state.phase = "warmup"
        state.weights = state._weights_for_phase("warmup")

        weights = compute_per_quality_weights(quality_labels, state)

        # Weights should be normalized and reflect phase + count
        assert sum(weights.values()) == pytest.approx(1.0)
        # High quality should have highest effective weight in warmup
        assert weights["high"] > weights["medium"]

    def test_weights_with_unequal_counts(self) -> None:
        """Test weights when quality levels have different sample counts."""
        # Many high, few low
        quality_labels = ["high"] * 500 + ["medium"] * 100 + ["low"] * 10
        state = CurriculumState(qualities=["high", "medium", "low"])

        weights = compute_per_quality_weights(quality_labels, state)

        assert sum(weights.values()) == pytest.approx(1.0)

    def test_weights_with_missing_quality(self) -> None:
        """Test weights when a quality level is absent."""
        quality_labels = ["high"] * 50 + ["medium"] * 50  # No low
        state = CurriculumState(qualities=["high", "medium", "low"])

        weights = compute_per_quality_weights(quality_labels, state)

        assert weights["low"] == 0.0
        assert weights["high"] > 0
        assert weights["medium"] > 0
