"""
Extended tests for curriculum sampler: reproducibility, unknown labels, fallback to uniform.
"""

import warnings

import pytest

from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.curriculum_sampler import (
    build_curriculum_sampler,
    compute_per_quality_weights,
    get_quality_indices,
)


def test_get_quality_indices_basic():
    labels = ["high", "high", "medium", "low", "high"]
    indices = get_quality_indices(labels, ["high", "medium", "low"])
    assert indices["high"] == [0, 1, 4]
    assert indices["medium"] == [2]
    assert indices["low"] == [3]


def test_compute_per_quality_weights_counts():
    labels = ["high"] * 10 + ["medium"] * 20 + ["low"] * 5
    state = CurriculumState(qualities=["high", "medium", "low"])
    state.phase = "expand"
    eff = compute_per_quality_weights(labels, state)
    # More medium samples but expand should still assign a higher share of effective weight to high
    assert eff["high"] > 0
    assert sum(eff.values()) == pytest.approx(1.0, rel=1e-6)


def test_build_sampler_reproducibility(sample_quality_labels):
    state = CurriculumState(qualities=["high", "medium", "low"])
    state.phase = "warmup"

    # Build two samplers with same seed and verify reproducible sampling indices
    sampler_a = build_curriculum_sampler(sample_quality_labels, state, num_samples=8, seed=123)
    sampler_b = build_curriculum_sampler(sample_quality_labels, state, num_samples=8, seed=123)

    # Draw a small sample list of indices using PyTorch sampler logic
    # Convert generator-based sampling to indices: WeightedRandomSampler yields indices per sample
    indices_a = list(iter(sampler_a))
    indices_b = list(iter(sampler_b))

    assert indices_a == indices_b


def test_build_sampler_unknown_labels_warns():
    labels = ["high", "unknown", "medium"]
    state = CurriculumState(qualities=["high", "medium"])
    state.phase = "warmup"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = build_curriculum_sampler(labels, state, num_samples=3, seed=42)
        # Unknown label warning expected
        assert any("not in curriculum qualities" in str(x.message) for x in w)


def test_build_sampler_all_zero_weights():
    # Introduce labels that are not part of known qualities so per-sample weights are all zeros
    labels = ["x", "x", "x"]
    state = CurriculumState(qualities=["high", "medium", "low"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = build_curriculum_sampler(labels, state, num_samples=3, seed=42)
        assert any("All sample weights are zero" in str(x.message) for x in w)


def test_compute_per_quality_weights_empty_counts():
    labels = ["a", "b", "c"]
    state = CurriculumState(qualities=["x", "y"])  # qualities not present in labels
    eff = compute_per_quality_weights(labels, state)
    # All effective weights should be zero (no matching qualities)
    assert eff["x"] == 0.0 and eff["y"] == 0.0
