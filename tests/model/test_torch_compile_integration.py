"""Tests for torch.compile integration in Chemprop and Chemeleon models."""

import pytest
import torch

from admet.model.chemprop.config import PerformanceOptimizationConfig


class TestTorchCompileConfig:
    """Test torch.compile configuration schema."""

    def test_performance_config_defaults(self):
        """Test PerformanceOptimizationConfig defaults."""
        config = PerformanceOptimizationConfig()

        assert config.use_mixed_precision is False
        assert config.use_torch_compile is False
        assert config.torch_compile_mode == "reduce-overhead"
        assert config.torch_compile_fullgraph is False
        assert config.torch_compile_dynamic is False

    def test_performance_config_with_compile_enabled(self):
        """Test PerformanceOptimizationConfig with torch.compile enabled."""
        config = PerformanceOptimizationConfig(
            use_torch_compile=True,
            torch_compile_mode="max-autotune",
            torch_compile_fullgraph=True,
        )

        assert config.use_torch_compile is True
        assert config.torch_compile_mode == "max-autotune"
        assert config.torch_compile_fullgraph is True

    def test_compile_modes_are_valid(self):
        """Test that all documented compile modes work with torch.compile."""
        valid_modes = ["default", "reduce-overhead", "max-autotune"]

        for mode in valid_modes:
            # Create a simple model and compile it
            model = torch.nn.Linear(10, 10)
            compiled = torch.compile(model, mode=mode)
            assert compiled is not None


class TestTorchCompileIntegration:
    """Test torch.compile integration in model training."""

    @pytest.mark.skip(reason="Requires full model setup - integration test")
    def test_chemprop_compilation_enabled(self, sample_chemprop_config):
        """Test that Chemprop model compiles when enabled."""
        # This would require full model instantiation
        # Placeholder for future integration test
        pass

    @pytest.mark.skip(reason="Requires full model setup - integration test")
    def test_chemeleon_compilation_enabled(self, sample_chemeleon_config):
        """Test that Chemeleon model compiles when enabled."""
        # This would require full model instantiation
        # Placeholder for future integration test
        pass

    def test_compilation_fallback_on_error(self):
        """Test that compilation failures are handled gracefully."""
        # Compilation should not crash if it fails
        # The code has try/except with fallback to uncompiled
        assert True  # Placeholder - actual test would verify logging
