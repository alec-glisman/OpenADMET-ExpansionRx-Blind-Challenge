"""
Unit tests for admet.model.chemprop.curriculum_sampler module.

Tests the weighted sampling functionality for curriculum learning.
"""

import warnings

import numpy as np
import pytest

from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.curriculum_sampler import (
    DynamicCurriculumSampler,
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
            build_curriculum_sampler(quality_labels, state, seed=42)
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
            _sampler2 = build_curriculum_sampler(quality_labels, state, seed=42)
            # Should warn about zero weights AND unknown qualities
            assert len(w) >= 1

        # Sampler should still work (uniform fallback)
        assert _sampler2 is not None
        assert len(_sampler2) == 3

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


class ReferenceDynamicCurriculumSamplerNonVectorized:
    """
    Non-vectorized reference implementation for regression testing.

    This implements the original per-sample loop algorithm with dict lookups
    that the optimized DynamicCurriculumSampler must match. Used to verify
    that vectorization optimizations don't change the sampling behavior.
    """

    def __init__(
        self,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
        num_samples: int | None = None,
        seed: int | None = None,
    ):
        self.quality_labels = list(quality_labels)
        self.curriculum_state = curriculum_state
        self._num_samples = num_samples or len(quality_labels)
        self.seed = seed

        # Compute quality counts (original way)
        self._quality_counts: dict[str, int] = {}
        for label in self.quality_labels:
            self._quality_counts[label] = self._quality_counts.get(label, 0) + 1

    def _compute_weights(self) -> np.ndarray:
        """Original non-vectorized weight computation with per-sample dict lookups."""
        target_probs = self.curriculum_state.sampling_probs()
        config = self.curriculum_state.config
        count_normalize = getattr(config, "count_normalize", True)

        weights = np.zeros(len(self.quality_labels), dtype=np.float64)

        if count_normalize:
            # Original: per-sample loop with dict lookups
            for i, label in enumerate(self.quality_labels):
                target_prop = target_probs.get(label, 0.0)
                count = self._quality_counts.get(label, 1)
                weights[i] = target_prop / count if count > 0 else 0.0
        else:
            # Legacy behavior
            for i, label in enumerate(self.quality_labels):
                weights[i] = target_probs.get(label, 0.0)

        if weights.sum() == 0:
            weights = np.ones(len(self.quality_labels), dtype=np.float64)

        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        weights = self._compute_weights()

        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng()

        indices = rng.choice(
            len(self.quality_labels),
            size=self._num_samples,
            replace=True,
            p=weights,
        )

        return iter(indices.tolist())

    def __len__(self):
        return self._num_samples


class TestDynamicCurriculumSamplerVectorizationRegression:
    """
    Regression tests verifying vectorized DynamicCurriculumSampler matches original.

    These tests ensure that the performance optimizations (vectorized weight
    computation using precomputed label indices) produce statistically equivalent
    results to the original per-sample dict lookup implementation.
    """

    @pytest.fixture
    def regression_test_data(self) -> tuple[list[str], CurriculumState]:
        """Create test data for regression testing."""
        np.random.seed(42)
        num_samples = 10000

        # Quality labels: 5% high, 75% medium, 20% low (imbalanced)
        quality_labels = (
            ["high"] * int(num_samples * 0.05)
            + ["medium"] * int(num_samples * 0.75)
            + ["low"] * int(num_samples * 0.20)
        )
        np.random.shuffle(quality_labels)

        curriculum_state = CurriculumState(
            qualities=["high", "medium", "low"],
            patience=5,
        )

        return quality_labels, curriculum_state

    def test_weight_computation_matches_reference(self, regression_test_data):
        """Verify vectorized weight computation matches original dict lookups."""
        quality_labels, curriculum_state = regression_test_data

        # Reference implementation
        ref_sampler = ReferenceDynamicCurriculumSamplerNonVectorized(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=1000,
            seed=42,
        )

        # Optimized implementation
        opt_sampler = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=1000,
            seed=42,
        )

        # Compare weights directly
        ref_weights = ref_sampler._compute_weights()
        opt_weights = opt_sampler._compute_weights()

        np.testing.assert_array_almost_equal(
            ref_weights,
            opt_weights,
            decimal=10,
            err_msg="Vectorized weight computation differs from reference",
        )

    def test_sampling_distribution_matches_reference(self, regression_test_data):
        """Verify sampling distribution matches reference implementation."""
        quality_labels, curriculum_state = regression_test_data
        num_samples = 100_000
        seed = 123

        # Reference (non-vectorized)
        ref_sampler = ReferenceDynamicCurriculumSamplerNonVectorized(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=num_samples,
            seed=seed,
        )

        # Optimized (vectorized)
        opt_sampler = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=num_samples,
            seed=seed,
            increment_seed_per_epoch=False,
        )

        def get_quality_distribution(indices):
            counts = {"high": 0, "medium": 0, "low": 0}
            for idx in indices:
                counts[quality_labels[idx]] += 1
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}

        ref_indices = list(ref_sampler)
        opt_indices = list(opt_sampler)

        ref_dist = get_quality_distribution(ref_indices)
        opt_dist = get_quality_distribution(opt_indices)

        # Distributions should be very close (within 1% for 100k samples)
        for quality in ["high", "medium", "low"]:
            diff = abs(ref_dist[quality] - opt_dist[quality])
            assert diff < 0.01, (
                f"Quality '{quality}' distribution mismatch: ref={ref_dist[quality]:.4f}, "
                f"opt={opt_dist[quality]:.4f}, diff={diff:.4f}"
            )

    def test_determinism_with_same_seed(self, regression_test_data):
        """Verify same seed produces identical results."""
        quality_labels, curriculum_state = regression_test_data

        sampler1 = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=5000,
            seed=42,
            increment_seed_per_epoch=False,
        )

        sampler2 = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=5000,
            seed=42,
            increment_seed_per_epoch=False,
        )

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2, "Same seed should produce identical samples"

    def test_phase_transitions_affect_distribution(self, regression_test_data):
        """Verify curriculum phase transitions affect sampling distribution."""
        quality_labels, curriculum_state = regression_test_data

        sampler = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=50_000,
            seed=456,
            increment_seed_per_epoch=False,
        )

        # Sample in warmup phase
        curriculum_state.phase = "warmup"
        curriculum_state.weights = curriculum_state._weights_for_phase("warmup")
        warmup_indices = list(sampler)
        warmup_high = sum(1 for idx in warmup_indices if quality_labels[idx] == "high")

        # Reset epoch counter
        sampler._current_epoch = 0

        # Sample in robust phase
        curriculum_state.phase = "robust"
        curriculum_state.weights = curriculum_state._weights_for_phase("robust")
        robust_indices = list(sampler)
        robust_high = sum(1 for idx in robust_indices if quality_labels[idx] == "high")

        # Warmup should have higher proportion of high-quality samples
        warmup_prop = warmup_high / len(warmup_indices)
        robust_prop = robust_high / len(robust_indices)

        assert warmup_prop > robust_prop, (
            f"Warmup high-quality proportion ({warmup_prop:.3f}) should be > " f"robust proportion ({robust_prop:.3f})"
        )

    def test_count_normalization_effect(self, regression_test_data):
        """Verify count normalization achieves target proportions."""
        quality_labels, curriculum_state = regression_test_data

        # With count_normalize=True (default), target proportions should be achieved
        sampler = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=100_000,
            seed=789,
            increment_seed_per_epoch=False,
        )

        indices = list(sampler)

        # Calculate actual sampled proportions
        counts = {"high": 0, "medium": 0, "low": 0}
        for idx in indices:
            counts[quality_labels[idx]] += 1
        total = sum(counts.values())
        actual_props = {k: v / total for k, v in counts.items()}

        # Target proportions from curriculum state
        target_props = curriculum_state.sampling_probs()

        # Actual proportions should be close to target (within 5%)
        # Note: Due to two-stage sampling in JointSampler, exact match isn't expected
        # but DynamicCurriculumSampler should achieve closer to target
        for quality in ["high", "medium", "low"]:
            diff = abs(actual_props[quality] - target_props[quality])
            assert diff < 0.05, (
                f"Quality '{quality}' count normalization not achieving target: "
                f"actual={actual_props[quality]:.3f}, target={target_props[quality]:.3f}"
            )

    def test_all_indices_in_valid_range(self, regression_test_data):
        """Verify all sampled indices are within valid range."""
        quality_labels, curriculum_state = regression_test_data

        sampler = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=10_000,
            seed=111,
        )

        indices = list(sampler)

        # All indices should be valid
        assert all(0 <= idx < len(quality_labels) for idx in indices)

        # Should return exactly num_samples
        assert len(indices) == 10_000
