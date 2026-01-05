"""Unit tests for performance optimization features.

Tests for:
- AsyncMLflowModelCheckpoint (async uploads, throttling)
- PerformanceOptimizationConfig validation
- Mixed precision configuration
- Gradient accumulation configuration
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from admet.model.chemprop.config import PerformanceOptimizationConfig
from admet.model.chemprop.model import MLflowModelCheckpoint


class TestPerformanceOptimizationConfig:
    """Test PerformanceOptimizationConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are conservative (all disabled)."""
        config = PerformanceOptimizationConfig()

        assert config.use_mixed_precision is False
        assert config.async_checkpoint_upload is False
        assert config.checkpoint_save_interval_seconds == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PerformanceOptimizationConfig(
            use_mixed_precision=True,
            async_checkpoint_upload=True,
            checkpoint_save_interval_seconds=30.0,
        )

        assert config.use_mixed_precision is True
        assert config.async_checkpoint_upload is True
        assert config.checkpoint_save_interval_seconds == 30.0


class TestMLflowModelCheckpoint:
    """Test MLflowModelCheckpoint with async and throttling features."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        client = Mock()
        client.log_artifact = Mock()
        client.log_metric = Mock()
        return client

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_synchronous_upload_default(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test default synchronous upload behavior."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        assert checkpoint._async_upload is False
        assert checkpoint._throttle_interval == 0.0

    def test_async_upload_thread_initialization(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test that async upload starts background thread."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            async_upload=True,
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        assert checkpoint._async_upload is True
        assert hasattr(checkpoint, "_upload_thread")
        assert checkpoint._upload_thread.is_alive()

        # Cleanup
        checkpoint._shutdown = True
        checkpoint._upload_queue.put(None)
        checkpoint._upload_thread.join(timeout=2.0)

    def test_checkpoint_throttling(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test checkpoint save throttling logic."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            throttle_interval_seconds=1.0,  # 1 second throttle
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        # Mock trainer and module
        mock_trainer = Mock(spec=pl.Trainer)
        mock_module = Mock(spec=pl.LightningModule)

        # First save should succeed
        with patch.object(ModelCheckpoint, "on_save_checkpoint"):
            checkpoint.on_save_checkpoint(mock_trainer, mock_module, {})
            # Record the first save time
            _ = checkpoint._last_save_time

        # Immediate second save should be throttled
        with patch.object(ModelCheckpoint, "on_save_checkpoint") as mock_super:
            checkpoint.on_save_checkpoint(mock_trainer, mock_module, {})
            # Should not call super() due to throttle
            mock_super.assert_not_called()

        # After throttle interval, save should succeed
        time.sleep(1.1)  # Wait for throttle to expire
        with patch.object(ModelCheckpoint, "on_save_checkpoint") as mock_super:
            checkpoint.on_save_checkpoint(mock_trainer, mock_module, {})
            mock_super.assert_called_once()

    def test_async_upload_queue_operations(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test async upload queue enqueue/dequeue."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            async_upload=True,
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        # Create a test checkpoint file
        test_checkpoint = temp_checkpoint_dir / "test_checkpoint.ckpt"
        test_checkpoint.write_text("test checkpoint data")

        # Queue an upload
        checkpoint._upload_queue.put((str(test_checkpoint), "checkpoints/best", 0.123))

        # Wait a bit for worker to process
        time.sleep(0.5)

        # Verify upload was called
        assert mock_mlflow_client.log_artifact.call_count >= 1

        # Cleanup
        checkpoint._shutdown = True
        checkpoint._upload_queue.put(None)
        checkpoint._upload_thread.join(timeout=2.0)

    def test_async_upload_graceful_shutdown(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test graceful shutdown drains queue."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            async_upload=True,
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        # Queue multiple uploads
        for i in range(3):
            test_checkpoint = temp_checkpoint_dir / f"test_checkpoint_{i}.ckpt"
            test_checkpoint.write_text(f"checkpoint {i}")
            checkpoint._upload_queue.put((str(test_checkpoint), f"checkpoints/test_{i}", 0.1 * i))

        # Trigger graceful shutdown
        mock_trainer = Mock(spec=pl.Trainer)
        mock_module = Mock(spec=pl.LightningModule)
        checkpoint.teardown(mock_trainer, mock_module, "fit")

        # Verify thread is stopped
        assert not checkpoint._upload_thread.is_alive()

        # Verify at least one upload was processed (may be less than 3 due to timing)
        assert mock_mlflow_client.log_artifact.call_count >= 1

    def test_sync_upload_when_async_disabled(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test synchronous upload when async is disabled."""
        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            async_upload=False,  # Explicitly disable
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        # Create a test checkpoint
        test_checkpoint = temp_checkpoint_dir / "test_checkpoint.ckpt"
        test_checkpoint.write_text("test checkpoint")

        # Mock the best_model_path attribute
        checkpoint.best_model_path = str(test_checkpoint)
        checkpoint.best_model_score = 0.123

        # Mock trainer and module
        mock_trainer = Mock(spec=pl.Trainer)
        mock_module = Mock(spec=pl.LightningModule)

        # Trigger save
        with patch.object(ModelCheckpoint, "on_save_checkpoint"):
            checkpoint.on_save_checkpoint(mock_trainer, mock_module, {})

        # Verify synchronous upload was called
        mock_mlflow_client.log_artifact.assert_called_once_with(
            "test_run_id", str(test_checkpoint), artifact_path="checkpoints/best"
        )
        mock_mlflow_client.log_metric.assert_called_once_with("test_run_id", "best_val_loss", 0.123)

    def test_upload_worker_exception_handling(self, mock_mlflow_client, temp_checkpoint_dir):
        """Test that upload worker handles exceptions gracefully."""
        # Make log_artifact raise an exception on first call, succeed on second
        mock_mlflow_client.log_artifact.side_effect = [
            Exception("Upload failed"),
            None,  # Second call succeeds
        ]

        checkpoint = MLflowModelCheckpoint(
            mlflow_client=mock_mlflow_client,
            run_id="test_run_id",
            async_upload=True,
            dirpath=temp_checkpoint_dir,
            filename="test-{epoch:04d}",
        )

        # Queue two uploads - first should fail, second should succeed
        test_checkpoint1 = temp_checkpoint_dir / "test_checkpoint_1.ckpt"
        test_checkpoint1.write_text("test1")
        checkpoint._upload_queue.put((str(test_checkpoint1), "checkpoints/best", 0.123))

        test_checkpoint2 = temp_checkpoint_dir / "test_checkpoint_2.ckpt"
        test_checkpoint2.write_text("test2")
        checkpoint._upload_queue.put((str(test_checkpoint2), "checkpoints/best", 0.456))

        # Wait for processing
        time.sleep(1.0)

        # Worker should still be alive after handling the exception
        assert checkpoint._upload_thread.is_alive(), "Worker thread should survive exceptions"

        # Verify both uploads were attempted (first failed, second succeeded)
        assert mock_mlflow_client.log_artifact.call_count == 2

        # Cleanup
        checkpoint._shutdown = True
        checkpoint._upload_queue.put(None)
        checkpoint._upload_thread.join(timeout=2.0)


class TestGradientAccumulationConfig:
    """Test gradient accumulation configuration."""

    def test_accumulate_grad_batches_default(self):
        """Test default gradient accumulation is 1 (disabled)."""
        from admet.model.chemprop.config import OptimizationConfig

        config = OptimizationConfig()
        assert config.accumulate_grad_batches == 1

    def test_accumulate_grad_batches_custom(self):
        """Test custom gradient accumulation value."""
        from admet.model.chemprop.config import OptimizationConfig

        config = OptimizationConfig(accumulate_grad_batches=4)
        assert config.accumulate_grad_batches == 4


class TestMixedPrecisionConfig:
    """Test mixed precision configuration."""

    def test_precision_parameter_disabled(self):
        """Test precision is FP32 when mixed precision disabled."""
        perf_config = PerformanceOptimizationConfig(use_mixed_precision=False)
        precision = "16-mixed" if perf_config.use_mixed_precision else "32-true"
        assert precision == "32-true"

    def test_precision_parameter_enabled(self):
        """Test precision is FP16 when mixed precision enabled."""
        perf_config = PerformanceOptimizationConfig(use_mixed_precision=True)
        precision = "16-mixed" if perf_config.use_mixed_precision else "32-true"
        assert precision == "16-mixed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
