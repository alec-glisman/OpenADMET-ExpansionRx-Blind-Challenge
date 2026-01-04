"""
Tests for model utility functions in ChempropModel: sanitizer, Criterion resolver, and MLflowModelCheckpoint.
"""

import pytest

from admet.model.chemprop.model import CriterionName, MLflowModelCheckpoint, _sanitize_mlflow_metric_name


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


def test_get_best_checkpoint_path_from_callback(tmp_path, mocker):
    """Test that _get_best_checkpoint_path finds checkpoint from callback."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    from admet.model.chemprop.model import ChempropModel

    # Create a dummy best checkpoint file
    best_ckpt = tmp_path / "best-0001-val_loss=0.10.ckpt"
    best_ckpt.write_text("dummy checkpoint content")

    # Create mock trainer with ModelCheckpoint callback
    mock_checkpoint_callback = mocker.MagicMock(spec=ModelCheckpoint)
    mock_checkpoint_callback.best_model_path = str(best_ckpt)

    mock_trainer = mocker.MagicMock()
    mock_trainer.callbacks = [mock_checkpoint_callback]

    # Create a minimal ChempropModel mock to test the method
    model = mocker.MagicMock(spec=ChempropModel)
    model.trainer = mock_trainer
    model.output_dir = None
    model._checkpoint_temp_dir = None

    # Call the actual method
    result = ChempropModel._get_best_checkpoint_path(model)

    assert result is not None
    assert result == best_ckpt


def test_get_best_checkpoint_path_from_directory(tmp_path, mocker):
    """Test that _get_best_checkpoint_path falls back to directory search."""
    from admet.model.chemprop.model import ChempropModel

    # Create best checkpoint files in directory
    best_ckpt1 = tmp_path / "best-0001-val_loss=0.15.ckpt"
    best_ckpt2 = tmp_path / "best-0002-val_loss=0.10.ckpt"
    best_ckpt1.write_text("older checkpoint")
    best_ckpt2.write_text("newer checkpoint")

    # Make ckpt2 appear newer (modification time)
    import time

    time.sleep(0.01)  # Small delay to ensure different mtime
    best_ckpt2.write_text("newer checkpoint updated")

    # Create model mock with no trainer callback but with checkpoint directory
    model = mocker.MagicMock(spec=ChempropModel)
    model.trainer = None
    model.output_dir = tmp_path
    model._checkpoint_temp_dir = None

    # Call the actual method
    result = ChempropModel._get_best_checkpoint_path(model)

    assert result is not None
    # Should return the most recently modified checkpoint
    assert result.name.startswith("best-")


def test_get_best_checkpoint_path_returns_none_when_not_found(mocker):
    """Test that _get_best_checkpoint_path returns None when no checkpoint exists."""
    from admet.model.chemprop.model import ChempropModel

    model = mocker.MagicMock(spec=ChempropModel)
    model.trainer = None
    model.output_dir = None
    model._checkpoint_temp_dir = None

    result = ChempropModel._get_best_checkpoint_path(model)

    assert result is None
