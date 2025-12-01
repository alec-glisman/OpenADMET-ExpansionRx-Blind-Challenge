"""
Tests for model utility functions in ChempropModel: sanitizer, Criterion resolver, and MLflowModelCheckpoint.
"""

import pytest

from admet.model.chemprop.model import (
    CriterionName,
    MLflowModelCheckpoint,
    _sanitize_mlflow_metric_name,
)


def test_sanitize_mlflow_metric_name():
    """Test that special characters are sanitized in metric names."""
    name = "test:metric/with#chars%"
    sanitized = _sanitize_mlflow_metric_name(name)
    assert ":" not in sanitized
    assert "#" not in sanitized
    assert "%" not in sanitized


def test_criterion_resolve_known():
    """Test that known criterion names resolve to callable metrics."""
    crit = CriterionName.resolve("MSE")
    assert crit is not None


def test_criterion_resolve_unknown():
    """Test that unknown criterion names raise ValueError."""
    with pytest.raises(ValueError):
        CriterionName.resolve("NOT_A_CRIT")


def test_mlflow_model_checkpoint_logs(tmp_path, mocker):
    """Test that MLflowModelCheckpoint logs artifacts and metrics."""
    # Create a dummy checkpoint file so that best_model_path exists
    tmp_file = tmp_path / "best.ckpt"
    tmp_file.write_text("dummy")

    mlflow_client = mocker.MagicMock()
    run_id = "run-123"
    cb = MLflowModelCheckpoint(mlflow_client, run_id, dirpath=str(tmp_path), filename="best.ckpt")

    # Simulate best model path and score
    cb.best_model_path = str(tmp_file)
    cb.best_model_score = 0.123

    # Create dummy trainer/module and checkpoint
    trainer = mocker.MagicMock()
    pl_module = mocker.MagicMock()
    checkpoint = {}

    cb.on_save_checkpoint(trainer, pl_module, checkpoint)

    # Ensure log_artifact called
    assert mlflow_client.log_artifact.called
    assert mlflow_client.log_metric.called
