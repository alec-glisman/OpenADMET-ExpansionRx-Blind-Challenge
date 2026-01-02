"""Tests for MLflow backfilling from Ray Tune results."""

from unittest.mock import MagicMock, patch

import pytest
from ray.tune.result_grid import ResultGrid

from admet.model.hpo_mlflow_callback import RobustMLflowLoggerCallback, backfill_mlflow_from_ray_results


class TestRobustMLflowLoggerCallback:
    """Test the robust MLflow logger callback."""

    def test_callback_initialization(self):
        """Test callback can be initialized."""
        callback = RobustMLflowLoggerCallback(
            tracking_uri="http://localhost:5000",
            experiment_name="test_exp",
            save_artifact=False,
            tags={"key": "value"},
        )
        assert callback._inner is not None
        assert callback._failed_trials == set()

    def test_callback_handles_failed_trials(self):
        """Test callback tracks failed trials."""
        callback = RobustMLflowLoggerCallback()

        # Mock a trial
        trial = MagicMock()
        trial.trial_id = "test_trial_123"

        # Simulate an error with inactive run
        callback._inner = MagicMock()
        callback._inner.on_trial_result.side_effect = Exception("must be in the 'active' state")

        # Should not raise, but should add to failed trials
        callback.on_trial_result(
            iteration=1,
            trials=[trial],
            trial=trial,
            result={"val_loss": 0.5},
        )

        assert "test_trial_123" in callback._failed_trials

        # Second call should be skipped
        callback._inner.on_trial_result.side_effect = None
        callback._inner.on_trial_result.reset_mock()

        callback.on_trial_result(
            iteration=2,
            trials=[trial],
            trial=trial,
            result={"val_loss": 0.4},
        )

        # Inner callback should not have been called
        callback._inner.on_trial_result.assert_not_called()


class TestBackfillMLflowFromRayResults:
    """Test the MLflow backfilling function."""

    @patch("admet.model.hpo_mlflow_callback.mlflow")
    def test_backfill_with_mock_results(self, mock_mlflow):
        """Test backfilling with mocked Ray Tune results."""
        # Create mock results
        mock_result1 = MagicMock()
        mock_result1.config = {"learning_rate": 0.001, "batch_size": 32}
        mock_result1.metrics = {
            "trial_id": "trial_1",
            "val_loss": 0.5,
            "val_mae": 0.3,
            "_internal": "skip_this",
        }
        mock_result1.checkpoint = None

        mock_result2 = MagicMock()
        mock_result2.config = {"learning_rate": 0.01, "batch_size": 64}
        mock_result2.metrics = {
            "trial_id": "trial_2",
            "val_loss": 0.4,
            "val_mae": 0.25,
        }
        mock_result2.checkpoint = None

        mock_results = [mock_result1, mock_result2]

        # Configure mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        # Call backfill
        backfill_mlflow_from_ray_results(
            results=mock_results,
            experiment_name="test_exp",
            parent_run_id="parent_123",
            tracking_uri="http://localhost:5000",
        )

        # Verify MLflow was called correctly
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.start_run.call_count == 2
        assert mock_mlflow.log_params.call_count == 2
        assert mock_mlflow.log_metrics.call_count == 2

    @patch("admet.model.hpo_mlflow_callback.mlflow")
    def test_backfill_handles_errors_gracefully(self, mock_mlflow):
        """Test backfilling continues even if some trials fail."""
        # Create mock results with one that will fail
        mock_result1 = MagicMock()
        mock_result1.config = {"learning_rate": 0.001}
        mock_result1.metrics = {"val_loss": 0.5}
        mock_result1.checkpoint = None

        mock_result2 = MagicMock()
        mock_result2.config = None  # This will cause an error
        mock_result2.metrics = None
        mock_result2.checkpoint = None

        mock_results = [mock_result1, mock_result2]

        # Configure mock MLflow to succeed for first, fail for second
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        # Should not raise exception
        backfill_mlflow_from_ray_results(
            results=mock_results,
            experiment_name="test_exp",
        )

        # Should have attempted to process both trials
        assert mock_mlflow.set_experiment.called


class TestFlattenDict:
    """Test the dictionary flattening utility."""

    def test_flatten_nested_dict(self):
        """Test flattening nested dictionaries."""
        from admet.model.hpo_mlflow_callback import _flatten_dict

        nested = {
            "a": 1,
            "b": {"c": 2, "d": {"e": 3}},
            "f": "text",
        }

        flat = _flatten_dict(nested)

        assert flat == {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3,
            "f": "text",
        }

    def test_flatten_converts_complex_types(self):
        """Test flattening converts complex types to strings."""
        from admet.model.hpo_mlflow_callback import _flatten_dict

        data = {
            "list": [1, 2, 3],
            "tuple": (4, 5),
            "dict": {"nested": "value"},
        }

        flat = _flatten_dict(data)

        assert flat["list"] == "[1, 2, 3]"
        assert flat["tuple"] == "(4, 5)"
        assert flat["dict.nested"] == "value"
