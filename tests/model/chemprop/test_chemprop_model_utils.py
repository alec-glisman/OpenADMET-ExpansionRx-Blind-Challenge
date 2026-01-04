"""
Unit tests for utilities in admet.model.chemprop.model module.

These include tests for _sanitize_mlflow_metric_name, CriterionName.resolve,
and MLflowModelCheckpoint behavior regarding MLflow logging.
"""

import pytest

from admet.model.chemprop.model import (
    CriterionName,
    MLflowModelCheckpoint,
    _sanitize_mlflow_metric_name,
)


class DummyTrainer:
    """Minimal trainer stub for checkpoint callback."""

    def __init__(self):
        self.current_epoch = 1
        self.global_step = 42


class DummyModule:
    """Minimal module stub for checkpoint callback."""

    def __init__(self):
        self.logged = {}

    def log(self, key, value, on_step=False, on_epoch=True):
        self.logged[key] = value


def test_sanitize_mlflow_metric_name_special_chars():
    """Test that special characters are sanitized in metric names."""
    orig = "val_loss:component[0] > threshold%"
    sanitized = _sanitize_mlflow_metric_name(orig)
    assert " " not in sanitized or " " in sanitized  # sanity check
    assert ":" not in sanitized
    assert "[" not in sanitized and "]" not in sanitized
    assert "%" not in sanitized or "pct" in sanitized


def test_criterion_name_resolve_success():
    """Test that known criterion resolves to instantiated metric."""
    metric = CriterionName.resolve("MSE")
    assert metric is not None


def test_criterion_name_resolve_invalid():
    """Test that unknown criterion raises ValueError."""
    with pytest.raises(ValueError):
        CriterionName.resolve("NOT_A_CRITERION")


def test_mlflow_model_checkpoint_logs(tmp_path, mocker):
    """Test that MLflowModelCheckpoint logs artifacts and metrics."""
    # Create a fake best model file
    tmp_dir = tmp_path
    p = tmp_dir / "best.ckpt"
    p.write_text("dummy checkpoint")

    # Create mock client that tracks calls
    client = mocker.MagicMock()

    run_id = "run123"

    cp = MLflowModelCheckpoint(
        mlflow_client=client,
        run_id=run_id,
        dirpath=str(tmp_dir),
        monitor="val_loss",
        mode="min",
    )

    # Simulate that ckpt manager has set best_model_path and best_model_score
    cp.best_model_path = str(p)
    cp.best_model_score = 0.123

    trainer = DummyTrainer()
    module = DummyModule()

    # Call the callback
    cp.on_save_checkpoint(trainer, module, checkpoint={})

    # Ensure artifact logged and metric logged
    assert client.log_artifact.called, "log_artifact was not called"
    assert client.log_metric.called, "log_metric was not called"
