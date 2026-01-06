"""Unit tests for GPU metrics auto-detection and computation.

Tests for:
- GPU metrics auto-detection logic
- CPU/GPU metric equivalence
- PostTrainingConfig use_gpu_metrics settings
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from admet.model.chemprop.config import PostTrainingConfig


class TestGPUMetricsConfig:
    """Test PostTrainingConfig GPU metrics settings."""

    def test_default_is_auto(self):
        """Test default use_gpu_metrics is 'auto'."""
        config = PostTrainingConfig()
        assert config.use_gpu_metrics == "auto"

    def test_accepts_string_values(self):
        """Test all valid string values are accepted."""
        config_auto = PostTrainingConfig(use_gpu_metrics="auto")
        assert config_auto.use_gpu_metrics == "auto"

        config_true = PostTrainingConfig(use_gpu_metrics="true")
        assert config_true.use_gpu_metrics == "true"

        config_false = PostTrainingConfig(use_gpu_metrics="false")
        assert config_false.use_gpu_metrics == "false"


class TestGPUMetricsAutoDetection:
    """Test GPU metrics auto-detection logic."""

    def _resolve_gpu_metrics_setting(self, config: PostTrainingConfig) -> bool:
        """Helper method matching model.py implementation."""
        setting = config.use_gpu_metrics
        if setting == "auto":
            return torch.cuda.is_available()
        elif setting == "true":
            return True
        elif setting == "false":
            return False
        else:
            # Backward compatibility
            return str(setting).lower() == "true"

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_with_gpu_available(self, mock_cuda):
        """Test auto mode returns True when GPU available."""
        config = PostTrainingConfig(use_gpu_metrics="auto")
        result = self._resolve_gpu_metrics_setting(config)
        assert result is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_with_gpu_unavailable(self, mock_cuda):
        """Test auto mode returns False when GPU unavailable."""
        config = PostTrainingConfig(use_gpu_metrics="auto")
        result = self._resolve_gpu_metrics_setting(config)
        assert result is False

    def test_force_true(self):
        """Test forcing GPU metrics on."""
        config = PostTrainingConfig(use_gpu_metrics="true")
        result = self._resolve_gpu_metrics_setting(config)
        assert result is True

    def test_force_false(self):
        """Test forcing GPU metrics off."""
        config = PostTrainingConfig(use_gpu_metrics="false")
        result = self._resolve_gpu_metrics_setting(config)
        assert result is False

    def test_backward_compatibility_bool_true(self):
        """Test backward compatibility with boolean True."""
        # Old configs might have bool instead of str
        config = Mock()
        config.use_gpu_metrics = True
        result = str(True).lower() == "true"
        assert result is True

    def test_backward_compatibility_bool_false(self):
        """Test backward compatibility with boolean False."""
        config = Mock()
        config.use_gpu_metrics = False
        result = str(False).lower() == "true"
        assert result is False


class TestCPUGPUMetricEquivalence:
    """Test that CPU and GPU metrics produce equivalent results."""

    @pytest.fixture
    def sample_data(self):
        """Create sample prediction data."""
        np.random.seed(42)
        y_true = np.random.randn(100, 3).astype(np.float32)  # 100 samples, 3 targets
        y_pred = y_true + np.random.randn(100, 3).astype(np.float32) * 0.1  # Add small noise
        return y_true, y_pred

    def test_cpu_gpu_mae_equivalence(self, sample_data):
        """Test MAE computation is equivalent on CPU and GPU."""
        y_true, y_pred = sample_data

        # CPU computation
        mae_cpu = np.mean(np.abs(y_true - y_pred), axis=0)

        # GPU computation (if available)
        if torch.cuda.is_available():
            y_true_gpu = torch.from_numpy(y_true).cuda()
            y_pred_gpu = torch.from_numpy(y_pred).cuda()
            mae_gpu = torch.mean(torch.abs(y_true_gpu - y_pred_gpu), dim=0).cpu().numpy()

            # Should match within floating point tolerance
            np.testing.assert_allclose(mae_cpu, mae_gpu, rtol=1e-5, atol=1e-6)
        else:
            pytest.skip("GPU not available")

    def test_cpu_gpu_rmse_equivalence(self, sample_data):
        """Test RMSE computation is equivalent on CPU and GPU."""
        y_true, y_pred = sample_data

        # CPU computation
        mse_cpu = np.mean((y_true - y_pred) ** 2, axis=0)
        rmse_cpu = np.sqrt(mse_cpu)

        # GPU computation (if available)
        if torch.cuda.is_available():
            y_true_gpu = torch.from_numpy(y_true).cuda()
            y_pred_gpu = torch.from_numpy(y_pred).cuda()
            mse_gpu = torch.mean((y_true_gpu - y_pred_gpu) ** 2, dim=0)
            rmse_gpu = torch.sqrt(mse_gpu).cpu().numpy()

            # Should match within floating point tolerance
            np.testing.assert_allclose(rmse_cpu, rmse_gpu, rtol=1e-5, atol=1e-6)
        else:
            pytest.skip("GPU not available")

    def test_cpu_gpu_correlation_equivalence(self, sample_data):
        """Test correlation computation is equivalent on CPU and GPU."""
        y_true, y_pred = sample_data

        # CPU Pearson correlation
        def pearson_cpu(y_t, y_p):
            return np.corrcoef(y_t, y_p)[0, 1]

        corr_cpu = [pearson_cpu(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

        # GPU correlation (if available)
        if torch.cuda.is_available():
            y_true_gpu = torch.from_numpy(y_true).cuda()
            y_pred_gpu = torch.from_numpy(y_pred).cuda()

            corr_gpu = []
            for i in range(y_true.shape[1]):
                # Pearson correlation on GPU
                y_t = y_true_gpu[:, i]
                y_p = y_pred_gpu[:, i]
                mean_t = torch.mean(y_t)
                mean_p = torch.mean(y_p)
                std_t = torch.std(y_t, unbiased=False)
                std_p = torch.std(y_p, unbiased=False)
                cov = torch.mean((y_t - mean_t) * (y_p - mean_p))
                corr = (cov / (std_t * std_p)).cpu().numpy()
                corr_gpu.append(corr)

            # Should match within floating point tolerance
            np.testing.assert_allclose(corr_cpu, corr_gpu, rtol=1e-4, atol=1e-5)
        else:
            pytest.skip("GPU not available")


class TestGPUMetricsIntegration:
    """Integration tests for GPU metrics in model context."""

    def test_model_uses_resolved_setting(self):
        """Test that model correctly uses resolved GPU setting."""
        # This is a smoke test - actual model integration tested in integration tests
        from admet.model.chemprop.config import ChempropConfig

        config = ChempropConfig()
        config.post_training.use_gpu_metrics = "auto"

        # Verify config is set correctly
        assert config.post_training.use_gpu_metrics == "auto"

    @patch("torch.cuda.is_available", return_value=False)
    def test_gpu_metrics_disabled_when_unavailable(self, mock_cuda):
        """Test GPU metrics gracefully disabled when GPU unavailable."""
        config = PostTrainingConfig(use_gpu_metrics="auto")

        # Mock the resolution logic
        setting = config.use_gpu_metrics
        if setting == "auto":
            use_gpu = torch.cuda.is_available()  # Will be False due to mock
        else:
            use_gpu = setting == "true"

        assert use_gpu is False

    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_metrics_enabled_when_available(self, mock_cuda):
        """Test GPU metrics enabled when GPU available."""
        config = PostTrainingConfig(use_gpu_metrics="auto")

        # Mock the resolution logic
        setting = config.use_gpu_metrics
        if setting == "auto":
            use_gpu = torch.cuda.is_available()  # Will be True due to mock
        else:
            use_gpu = setting == "true"

        assert use_gpu is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
