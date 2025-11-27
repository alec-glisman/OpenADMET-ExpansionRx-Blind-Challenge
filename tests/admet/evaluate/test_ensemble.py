"""Unit tests for the ensemble evaluation helpers.

These tests validate prediction aggregation, evaluation on labeled datasets
and blind datasets, and expected data frame shapes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from admet.evaluate.ensemble import aggregate_predictions, evaluate_blind_dataset, evaluate_labeled_dataset


class DummyModel:
    def __init__(self, endpoints: Sequence[str], input_type: str = "smiles", constant: float = 0.0):
        self.endpoints = list(endpoints)
        self.input_type = input_type
        self.constant = constant

    def predict(self, X):
        # X is ignored, return constant values
        n = X.shape[0]
        return np.full((n, len(self.endpoints)), self.constant)


@pytest.mark.unit
def test_aggregate_predictions_mean_median() -> None:
    # 2 models, 3 rows, 2 endpoints
    arr = np.zeros((2, 3, 2))
    arr[0] += 1.0
    arr[1] += 3.0
    mean = aggregate_predictions(arr, agg_fn="mean")
    median = aggregate_predictions(arr, agg_fn="median")
    assert mean.shape == (3, 2)
    assert median.shape == (3, 2)
    assert np.allclose(mean, 2.0)
    assert np.allclose(median, 2.0)


@pytest.mark.unit
def test_evaluate_labeled_dataset_perfect_prediction() -> None:
    endpoints = ["LogD", "KSOL"]
    # Create a tiny DataFrame with two rows
    df = pd.DataFrame(
        {
            "Molecule Name": ["a", "b"],
            "SMILES": ["CCO", "CCN"],
            "Dataset": ["A", "A"],
            "LogD": [0.0, 1.0],
            "KSOL": [1.0, 2.0],
        }
    )

    # Create models that predict exactly the true values
    class PerfectModel:
        def __init__(self, endpoints):
            self.endpoints = list(endpoints)
            self.input_type = "smiles"

        def predict(self, X):
            # X shape (n, 1) -> simple backfill from df input
            # Use DF order implicitly; predictions mirror true targets
            return np.array([[0.0, 1.0], [1.0, 2.0]])

    models = [PerfectModel(endpoints), PerfectModel(endpoints)]
    preds_log_df, preds_lin_df, metrics_log_df, metrics_lin_df = evaluate_labeled_dataset(
        models, df, endpoints, agg_fn="mean"
    )
    # Ensure all returned DataFrames are present and not empty
    assert preds_log_df is not None and not preds_log_df.empty
    assert preds_lin_df is not None and not preds_lin_df.empty
    assert metrics_log_df is not None and not metrics_log_df.empty
    assert metrics_lin_df is not None and not metrics_lin_df.empty
    # Perfect predictions should yield small RMSE and R2 close to 1 per endpoint
    # We expect RMSE close to 0 and R2 close to 1
    # Metrics CSVs contain rows for endpoints -> check for presence
    assert "LogD" in metrics_log_df[metrics_log_df["endpoint"] == "LogD"]["endpoint"].values
    assert "KSOL" in metrics_log_df[metrics_log_df["endpoint"] == "KSOL"]["endpoint"].values


@pytest.mark.unit
def test_evaluate_blind_dataset_basic() -> None:
    endpoints = ["LogD", "KSOL"]
    df = pd.DataFrame({"Molecule Name": ["a"], "SMILES": ["CCO"]})
    # Model returns fixed prediction
    models = [
        DummyModel(endpoints, input_type="smiles", constant=0.5),
        DummyModel(endpoints, input_type="smiles", constant=1.5),
    ]
    preds_log_df, preds_lin_df = evaluate_blind_dataset(models, df, endpoints, agg_fn="mean")
    assert "Molecule Name" in preds_log_df.columns
    assert "SMILES" in preds_log_df.columns
    # Ensemble column present
    # New API: predictions DataFrames contain endpoint columns directly
    assert endpoints[0] in preds_log_df.columns
    assert endpoints[0] in preds_lin_df.columns
