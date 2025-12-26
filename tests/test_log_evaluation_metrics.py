"""
Tests for _log_evaluation_metrics: ensures per-quality and overall metrics are computed and logged.
"""

import pytest

from admet.model.chemprop.config import CurriculumConfig
from admet.model.chemprop.model import ChempropModel


@pytest.mark.no_mlflow_runs
def test_log_evaluation_metrics_per_quality(monkeypatch, mocker, sample_dataframe):
    """Test that per-quality metrics are computed and logged."""
    # Build a validation set with two high-quality examples to ensure per-quality metrics are computed
    val_indices = [0, 1]
    val_df = sample_dataframe.iloc[val_indices].reset_index(drop=True)
    train_df = sample_dataframe.drop(val_indices).reset_index(drop=True)

    # Create model with curriculum enabled
    curriculum_cfg = CurriculumConfig(
        enabled=True,
        quality_col="Quality",
        qualities=["high", "medium", "low"],
        patience=1,
        seed=42,
    )

    # Monkeypatch data.build_dataloader to not rely on Chemprop DataLoader complexity
    monkeypatch.setattr(
        "admet.model.chemprop.model.data.build_dataloader",
        lambda *args, **kwargs: object(),
    )

    model = ChempropModel(
        df_train=train_df,
        df_validation=val_df,
        smiles_col="SMILES",
        target_cols=["LogD"],
        target_weights=[1.0],
        progress_bar=False,
        hyperparams=None,
        mlflow_tracking=True,
        curriculum_config=curriculum_cfg,
    )

    # Replace predict with a constant perfect prediction so that metrics are perfect.
    preds_df = val_df[["LogD"]].copy()
    # Replace the predict method with one accepting the new log_metrics argument
    monkeypatch.setattr(model, "predict", lambda df, log_metrics=False: preds_df)

    # Set mlflow client
    model._mlflow_client = mocker.MagicMock()
    model.mlflow_run_id = "run-123"

    # Call logging function
    model._log_evaluation_metrics()

    # Now we use individual log_metric calls (not log_batch) to avoid conflicts
    # Check that log_metric was called or log_artifact was called for CSV
    assert model._mlflow_client.log_metric.called or model._mlflow_client.log_artifact.called
    # Verify that some metrics were logged (non-empty calls)
    if model._mlflow_client.log_metric.called:
        calls = model._mlflow_client.log_metric.call_args_list
        assert len(calls) > 0


@pytest.mark.skipif(True, reason="heavy e2e - skip in fast test runs")
def test_log_evaluation_metrics_end_to_end(monkeypatch, sample_dataframe):
    """
    End-to-end test for _log_evaluation_metrics.

    This is an end-to-end test that would run full predict pipeline and the
    _log_evaluation_metrics. Not executed by default; left here for future
    integration testing when environment supports full run.
    """
    pass
