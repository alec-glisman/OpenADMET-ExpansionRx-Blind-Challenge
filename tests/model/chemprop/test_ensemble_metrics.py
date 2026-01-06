"""
Unit tests for ensemble metric aggregation from child MLflow runs.

Tests verify that metrics from child runs are correctly fetched,
filtered, aggregated, and logged to the parent MLflow run.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from admet.model.chemprop.ensemble import ModelEnsemble


class TestEnsembleMetricAggregation:
    """Test suite for ensemble metric aggregation functionality."""

    def test_should_aggregate_metric_excludes_profiling(self):
        """Test that profiling metrics are excluded from aggregation."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)

        assert not ensemble._should_aggregate_metric("profiling/training_time")
        assert not ensemble._should_aggregate_metric("profiling.ensemble.mean_time")

    def test_should_aggregate_metric_excludes_system_metrics(self):
        """Test that system metrics are excluded from aggregation."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)

        assert not ensemble._should_aggregate_metric("system/cpu_utilization_percentage")
        assert not ensemble._should_aggregate_metric("system/disk_usage")

    def test_should_aggregate_metric_excludes_step_counters(self):
        """Test that step/epoch counters are excluded from aggregation."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)

        assert not ensemble._should_aggregate_metric("epoch")
        assert not ensemble._should_aggregate_metric("step")
        assert not ensemble._should_aggregate_metric("global_step")

    def test_should_aggregate_metric_includes_validation(self):
        """Test that validation metrics are included in aggregation."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)

        assert ensemble._should_aggregate_metric("validation/mean/mae")
        assert ensemble._should_aggregate_metric("validation/mean/rmse")
        assert ensemble._should_aggregate_metric("best_val_loss")

    def test_should_aggregate_metric_includes_test_and_train(self):
        """Test that test and training metrics are included in aggregation."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)

        assert ensemble._should_aggregate_metric("train_loss")
        assert ensemble._should_aggregate_metric("test/LogD/r2")
        assert ensemble._should_aggregate_metric("test/mean/mae")

    @patch("mlflow.log_metrics")
    def test_log_aggregated_metrics_computes_correct_stats(self, mock_log_metrics):
        """Test that mean and stddev are computed correctly."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._log_aggregated_metrics = ModelEnsemble._log_aggregated_metrics.__get__(ensemble)

        # Mock data: 3 models with validation/mean/mae values
        all_metrics = {"validation/mean/mae": [0.5, 0.6, 0.7]}

        ensemble._log_aggregated_metrics(all_metrics)

        # Verify log_metrics was called with correct aggregates
        assert mock_log_metrics.called
        logged_metrics = mock_log_metrics.call_args[0][0]

        expected_mean = np.mean([0.5, 0.6, 0.7])
        expected_stddev = np.std([0.5, 0.6, 0.7], ddof=1)

        assert logged_metrics["ensemble/validation/mean/mae_mean"] == pytest.approx(expected_mean)
        assert logged_metrics["ensemble/validation/mean/mae_stddev"] == pytest.approx(expected_stddev)

    @patch("mlflow.log_metrics")
    def test_log_aggregated_metrics_handles_single_value(self, mock_log_metrics):
        """Test that single-model ensemble sets stddev to 0."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._log_aggregated_metrics = ModelEnsemble._log_aggregated_metrics.__get__(ensemble)

        # Single model
        all_metrics = {"validation/mean/mae": [0.5]}

        ensemble._log_aggregated_metrics(all_metrics)

        # Verify stddev is 0 for single value
        assert mock_log_metrics.called
        logged_metrics = mock_log_metrics.call_args[0][0]

        assert logged_metrics["ensemble/validation/mean/mae_mean"] == pytest.approx(0.5)
        assert logged_metrics["ensemble/validation/mean/mae_stddev"] == 0.0

    @patch("mlflow.log_metrics")
    def test_log_aggregated_metrics_handles_empty_dict(self, mock_log_metrics):
        """Test that empty metrics dict is handled gracefully."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._log_aggregated_metrics = ModelEnsemble._log_aggregated_metrics.__get__(ensemble)

        all_metrics = {}

        ensemble._log_aggregated_metrics(all_metrics)

        # Should not call log_metrics for empty dict
        assert not mock_log_metrics.called

    def test_aggregate_child_run_metrics_handles_missing_client(self):
        """Test that missing MLflow client is handled gracefully."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._mlflow_client = None
        ensemble.parent_run_id = "parent_run_123"
        ensemble._child_run_ids = ["child_1", "child_2"]
        ensemble._aggregate_child_run_metrics = ModelEnsemble._aggregate_child_run_metrics.__get__(ensemble)

        # Should log warning and return early
        with patch("admet.model.chemprop.ensemble.logger") as mock_logger:
            ensemble._aggregate_child_run_metrics()
            assert mock_logger.warning.called

    def test_aggregate_child_run_metrics_handles_missing_child_ids(self):
        """Test that missing child run IDs is handled gracefully."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._mlflow_client = MagicMock()
        ensemble.parent_run_id = "parent_run_123"
        ensemble._child_run_ids = []
        ensemble._aggregate_child_run_metrics = ModelEnsemble._aggregate_child_run_metrics.__get__(ensemble)

        # Should log warning and return early
        with patch("admet.model.chemprop.ensemble.logger") as mock_logger:
            ensemble._aggregate_child_run_metrics()
            assert mock_logger.warning.called

    def test_aggregate_child_run_metrics_handles_failed_runs(self):
        """Test that failed child run fetches are handled gracefully."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._mlflow_client = MagicMock()
        ensemble.parent_run_id = "parent_run_123"
        ensemble._child_run_ids = ["child_1", "child_2", "child_3"]
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)
        ensemble._log_aggregated_metrics = MagicMock()
        ensemble._aggregate_child_run_metrics = ModelEnsemble._aggregate_child_run_metrics.__get__(ensemble)

        # Mock get_run to fail for child_2
        def mock_get_run(run_id):
            if run_id == "child_2":
                raise Exception("Network error")
            # Mock run data
            mock_run = MagicMock()
            mock_run.data.metrics = {"validation/mean/mae": 0.5 if run_id == "child_1" else 0.7}
            return mock_run

        ensemble._mlflow_client.get_run.side_effect = mock_get_run

        # Should continue processing other runs despite failure
        with patch("admet.model.chemprop.ensemble.logger") as mock_logger:
            ensemble._aggregate_child_run_metrics()
            # Should have logged warning for failed run
            assert mock_logger.warning.called
            # Should have called _log_aggregated_metrics with data from successful runs
            assert ensemble._log_aggregated_metrics.called

    @patch("mlflow.log_metrics")
    def test_aggregate_child_run_metrics_filters_and_aggregates(self, mock_log_metrics):
        """Test end-to-end aggregation with filtering."""
        ensemble = MagicMock(spec=ModelEnsemble)
        ensemble._mlflow_client = MagicMock()
        ensemble.parent_run_id = "parent_run_123"
        ensemble._child_run_ids = ["child_1", "child_2"]
        ensemble._should_aggregate_metric = ModelEnsemble._should_aggregate_metric.__get__(ensemble)
        ensemble._log_aggregated_metrics = ModelEnsemble._log_aggregated_metrics.__get__(ensemble)
        ensemble._aggregate_child_run_metrics = ModelEnsemble._aggregate_child_run_metrics.__get__(ensemble)

        # Mock run data with mixed metrics
        def mock_get_run(run_id):
            mock_run = MagicMock()
            if run_id == "child_1":
                mock_run.data.metrics = {
                    "validation/mean/mae": 0.5,
                    "train_loss": 0.3,
                    "profiling/training_time": 100.0,  # Should be filtered
                    "system/cpu": 50.0,  # Should be filtered
                    "epoch": 10,  # Should be filtered
                }
            else:
                mock_run.data.metrics = {
                    "validation/mean/mae": 0.7,
                    "train_loss": 0.4,
                    "profiling/training_time": 110.0,  # Should be filtered
                }
            return mock_run

        ensemble._mlflow_client.get_run.side_effect = mock_get_run

        ensemble._aggregate_child_run_metrics()

        # Verify only non-profiling metrics were aggregated
        assert mock_log_metrics.called
        logged_metrics = mock_log_metrics.call_args[0][0]

        # Should have validation/mean/mae and train_loss, but not profiling or system metrics
        assert "ensemble/validation/mean/mae_mean" in logged_metrics
        assert "ensemble/train_loss_mean" in logged_metrics
        assert "ensemble/profiling/training_time_mean" not in logged_metrics
        assert "ensemble/system/cpu_mean" not in logged_metrics
        assert "ensemble/epoch_mean" not in logged_metrics

        # Verify aggregated values
        assert logged_metrics["ensemble/validation/mean/mae_mean"] == pytest.approx(0.6)  # mean of 0.5, 0.7
        assert logged_metrics["ensemble/train_loss_mean"] == pytest.approx(0.35)  # mean of 0.3, 0.4
