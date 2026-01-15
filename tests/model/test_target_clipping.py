"""Tests for target clipping configuration and functionality.

Tests cover:
- TargetClippingConfig dataclass validation
- apply_target_clipping utility function
- Config validation (missing/extra targets, invalid bounds)
- Integration with UnifiedModelConfig
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from admet.model.config import (
    ConfigValidationError,
    TargetClippingConfig,
    UnifiedModelConfig,
    apply_target_clipping,
    validate_model_config,
    validate_target_clipping,
)


class TestTargetClippingConfig:
    """Tests for TargetClippingConfig dataclass."""

    def test_default_values(self):
        """Default config should have clipping disabled."""
        config = TargetClippingConfig()
        assert config.enabled is False
        assert config.clip_ranges == {}
        assert config.apply_to_individual_models is False
        assert config.apply_after_ensemble is True

    def test_enabled_config(self):
        """Enabled config with clip ranges should work."""
        config = TargetClippingConfig(
            enabled=True,
            clip_ranges={"LogD": [-3.0, 5.0], "Log KSOL": [-7.0, 0.5]},
            apply_to_individual_models=False,
            apply_after_ensemble=True,
        )
        assert config.enabled is True
        assert len(config.clip_ranges) == 2
        assert config.clip_ranges["LogD"] == [-3.0, 5.0]

    def test_omegaconf_structured(self):
        """Config should work with OmegaConf structured configs."""
        config = OmegaConf.structured(TargetClippingConfig)
        assert config.enabled is False
        assert config.clip_ranges == {}


class TestApplyTargetClipping:
    """Tests for apply_target_clipping utility function."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions DataFrame for testing."""
        return pd.DataFrame(
            {
                "LogD": [2.0, 6.0, -5.0, 0.0],  # Some values out of range
                "Log KSOL": [-8.0, 0.0, 1.0, -3.0],  # Some values out of range
                "SMILES": ["C", "CC", "CCC", "CCCC"],
            }
        )

    @pytest.fixture
    def clip_ranges(self):
        """Standard clip ranges for testing."""
        return {
            "LogD": [-3.0, 5.0],
            "Log KSOL": [-7.0, 0.5],
        }

    def test_clipping_applied_correctly(self, sample_predictions, clip_ranges):
        """Values should be clipped to specified bounds."""
        target_cols = ["LogD", "Log KSOL"]
        result = apply_target_clipping(sample_predictions, target_cols, clip_ranges)

        # Check LogD clipping: [-3.0, 5.0]
        # Original: [2.0, 6.0, -5.0, 0.0] -> [2.0, 5.0, -3.0, 0.0]
        assert result["LogD"].tolist() == [2.0, 5.0, -3.0, 0.0]

        # Check Log KSOL clipping: [-7.0, 0.5]
        # Original: [-8.0, 0.0, 1.0, -3.0] -> [-7.0, 0.0, 0.5, -3.0]
        assert result["Log KSOL"].tolist() == [-7.0, 0.0, 0.5, -3.0]

    def test_clipping_with_suffix(self, sample_predictions, clip_ranges):
        """Clipping should work with column suffix (for ensemble _mean columns)."""
        # Rename columns to have _mean suffix
        df = sample_predictions.rename(
            columns={
                "LogD": "LogD_mean",
                "Log KSOL": "Log KSOL_mean",
            }
        )
        target_cols = ["LogD", "Log KSOL"]
        result = apply_target_clipping(df, target_cols, clip_ranges, column_suffix="_mean")

        assert result["LogD_mean"].tolist() == [2.0, 5.0, -3.0, 0.0]
        assert result["Log KSOL_mean"].tolist() == [-7.0, 0.0, 0.5, -3.0]

    def test_clipping_modifies_in_place(self, sample_predictions, clip_ranges):
        """Function should modify DataFrame in place and return it."""
        target_cols = ["LogD", "Log KSOL"]
        original_id = id(sample_predictions)
        result = apply_target_clipping(sample_predictions, target_cols, clip_ranges)
        assert id(result) == original_id

    def test_clipping_ignores_missing_columns(self, sample_predictions, clip_ranges):
        """Missing columns in DataFrame should be silently ignored."""
        target_cols = ["LogD", "Log KSOL", "NonExistent"]
        # Should not raise - just ignores NonExistent
        result = apply_target_clipping(sample_predictions, target_cols, clip_ranges)
        assert "LogD" in result.columns
        assert "NonExistent" not in result.columns

    def test_clipping_ignores_missing_ranges(self, sample_predictions):
        """Targets without clip_ranges should be unchanged."""
        target_cols = ["LogD", "Log KSOL"]
        # Only clip LogD
        clip_ranges = {"LogD": [-3.0, 5.0]}
        original_ksol = sample_predictions["Log KSOL"].copy()
        result = apply_target_clipping(sample_predictions, target_cols, clip_ranges)

        # LogD should be clipped
        assert result["LogD"].tolist() == [2.0, 5.0, -3.0, 0.0]
        # Log KSOL should be unchanged
        assert result["Log KSOL"].equals(original_ksol)

    def test_clipping_with_default_bounds(self):
        """Default bounds (-1e9, 1e9) should effectively not clip."""
        df = pd.DataFrame({"LogD": [-1e8, 0, 1e8]})
        clip_ranges = {"LogD": [-1e9, 1e9]}
        result = apply_target_clipping(df, ["LogD"], clip_ranges)
        np.testing.assert_array_equal(result["LogD"].values, [-1e8, 0, 1e8])


class TestValidateTargetClipping:
    """Tests for validate_target_clipping function."""

    def test_valid_config(self):
        """Valid config should not raise."""
        clip_ranges = {"LogD": [-3.0, 5.0], "Log KSOL": [-7.0, 0.5]}
        target_cols = ["LogD", "Log KSOL"]
        # Should not raise
        validate_target_clipping(clip_ranges, target_cols)

    def test_missing_targets(self):
        """Missing targets in clip_ranges should raise."""
        clip_ranges = {"LogD": [-3.0, 5.0]}  # Missing Log KSOL
        target_cols = ["LogD", "Log KSOL"]
        with pytest.raises(ConfigValidationError, match="missing targets"):
            validate_target_clipping(clip_ranges, target_cols)

    def test_extra_targets(self):
        """Extra targets in clip_ranges should raise."""
        clip_ranges = {"LogD": [-3.0, 5.0], "Log KSOL": [-7.0, 0.5], "Extra": [0, 1]}
        target_cols = ["LogD", "Log KSOL"]
        with pytest.raises(ConfigValidationError, match="unknown targets"):
            validate_target_clipping(clip_ranges, target_cols)

    def test_empty_clip_ranges(self):
        """Empty clip_ranges should raise."""
        with pytest.raises(ConfigValidationError, match="clip_ranges is empty"):
            validate_target_clipping({}, ["LogD"])

    def test_empty_target_cols(self):
        """Empty target_cols should raise."""
        with pytest.raises(ConfigValidationError, match="target_cols is empty"):
            validate_target_clipping({"LogD": [0, 1]}, [])

    def test_invalid_bounds_not_list(self):
        """Bounds not as list/tuple should raise."""
        clip_ranges = {"LogD": 5.0}  # Should be [min, max]
        with pytest.raises(ConfigValidationError, match="must be \\[min, max\\]"):
            validate_target_clipping(clip_ranges, ["LogD"])

    def test_invalid_bounds_wrong_length(self):
        """Bounds with wrong length should raise."""
        clip_ranges = {"LogD": [-3.0, 5.0, 0.0]}  # Three values
        with pytest.raises(ConfigValidationError, match="must be \\[min, max\\]"):
            validate_target_clipping(clip_ranges, ["LogD"])

    def test_invalid_bounds_min_greater_than_max(self):
        """Bounds with min > max should raise."""
        clip_ranges = {"LogD": [5.0, -3.0]}  # min > max
        with pytest.raises(ConfigValidationError, match="min > max"):
            validate_target_clipping(clip_ranges, ["LogD"])


class TestValidateModelConfigWithTargetClipping:
    """Tests for validate_model_config with target_clipping section."""

    def test_disabled_clipping_not_validated(self):
        """Disabled target clipping should not trigger validation."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.target_clipping.enabled = False
        config.target_clipping.clip_ranges = {}  # Empty - would fail if validated
        config.data.target_cols = ["LogD"]
        # Should not raise
        validate_model_config(config)

    def test_enabled_clipping_validated(self):
        """Enabled target clipping should be validated."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.target_clipping.enabled = True
        config.target_clipping.clip_ranges = {"LogD": [-3.0, 5.0]}
        config.data.target_cols = ["LogD", "Log KSOL"]  # Missing Log KSOL in clip_ranges
        with pytest.raises(ConfigValidationError, match="missing ranges"):
            validate_model_config(config)

    def test_valid_enabled_clipping(self):
        """Valid enabled target clipping should pass validation."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.target_clipping.enabled = True
        config.target_clipping.clip_ranges = {
            "LogD": [-3.0, 5.0],
            "Log KSOL": [-7.0, 0.5],
        }
        config.data.target_cols = ["LogD", "Log KSOL"]
        # Should not raise
        validate_model_config(config)


class TestUnifiedModelConfigIntegration:
    """Integration tests for target_clipping in UnifiedModelConfig."""

    def test_target_clipping_in_unified_config(self):
        """TargetClippingConfig should be accessible from UnifiedModelConfig."""
        config = UnifiedModelConfig()
        assert hasattr(config, "target_clipping")
        assert isinstance(config.target_clipping, TargetClippingConfig)

    def test_yaml_roundtrip(self, tmp_path):
        """Config should survive YAML save/load roundtrip."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.target_clipping.enabled = True
        config.target_clipping.clip_ranges = {
            "LogD": [-3.0, 5.0],
            "Log KSOL": [-7.0, 0.5],
        }
        config.target_clipping.apply_to_individual_models = False
        config.target_clipping.apply_after_ensemble = True

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        OmegaConf.save(config, yaml_path)

        # Load and verify
        loaded = OmegaConf.load(yaml_path)
        assert loaded.target_clipping.enabled is True
        assert loaded.target_clipping.clip_ranges["LogD"] == [-3.0, 5.0]
        assert loaded.target_clipping.clip_ranges["Log KSOL"] == [-7.0, 0.5]
        assert loaded.target_clipping.apply_after_ensemble is True
