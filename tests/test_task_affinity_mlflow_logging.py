"""
Tests for MLflow logging from task affinity computation.
"""

import numpy as np
import pytest

from admet.model.chemprop.task_affinity import TaskAffinityConfig
from admet.model.chemprop.model import ChempropModel


@pytest.mark.no_mlflow_runs
def test_compute_task_affinity_logs_to_mlflow(monkeypatch, mocker, sample_dataframe):
    """Ensure that task affinity computation logs metrics and params to MLflow client."""

    train_df = sample_dataframe
    val_df = sample_dataframe

    # Create model with mlflow tracking enabled
    model = ChempropModel(
        df_train=train_df,
        df_validation=val_df,
        smiles_col="SMILES",
        target_cols=["LogD", "KSOL"],
        target_weights=[1.0, 1.0],
        progress_bar=False,
        hyperparams=None,
        mlflow_tracking=True,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="chemprop",
        mlflow_run_name=None,
        mlflow_run_id=None,
        mlflow_parent_run_id=None,
        mlflow_nested=False,
        curriculum_config=None,
        curriculum_state=None,
        task_affinity_config=TaskAffinityConfig(enabled=True, n_groups=2),
    )

    # Monkeypatch heavy computation to be fast/deterministic
    fake_affinity = np.array([[1.0, 0.5], [0.5, 1.0]])
    monkeypatch.setattr(
        "admet.model.chemprop.task_affinity.TaskAffinityComputer.compute_from_dataframe",
        lambda self, df, smiles_col, target_cols: (fake_affinity, ["LogD", "KSOL"]),
    )

    # Monkeypatch TaskGrouper to return stable grouping
    monkeypatch.setattr(
        "admet.model.chemprop.task_affinity.TaskGrouper.cluster",
        lambda self, affinity_matrix, task_names: [["LogD"], ["KSOL"]],
    )

    # Attach a mocked mlflow client and run id
    model._mlflow_client = mocker.MagicMock()
    model.mlflow_run_id = "run-42"

    # Execute computation
    model._compute_task_affinity()

    # Expect some metrics logged for affinity matrix and mean
    assert model._mlflow_client.log_metric.called

    # Verify that a specific metric name was logged
    called_metric_names = [c.args[1] for c in model._mlflow_client.log_metric.call_args_list]
    assert "task_affinity/matrix/LogD/KSOL" in called_metric_names
    assert "task_affinity/mean" in called_metric_names

    # Verify group params were logged
    assert model._mlflow_client.log_param.called
    param_calls = [c.args[1] for c in model._mlflow_client.log_param.call_args_list]
    assert "task_affinity/group/LogD" in param_calls
    assert "task_affinity/group/KSOL" in param_calls
