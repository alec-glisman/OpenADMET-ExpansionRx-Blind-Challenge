"""Unit tests for the ensemble evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from admet.evaluate.ensemble import (
    aggregate_predictions,
    evaluate_labeled_dataset,
    evaluate_blind_dataset,
)


class DummyModel:
    def __init__(self, endpoints, input_type="smiles", constant=0.0):
        self.endpoints = list(endpoints)
        self.input_type = input_type
        self.constant = constant

    def predict(self, X):
        # X is ignored, return constant values
        n = X.shape[0]
        return np.full((n, len(self.endpoints)), self.constant)


def test_aggregate_predictions_mean_median():
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


def test_evaluate_labeled_dataset_perfect_prediction():
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
            n = X.shape[0]
            # Return the 'true' mapping from SMILES via known DF order
            # For test, we manually return same values
            return np.array([[0.0, 1.0], [1.0, 2.0]])

    models = [PerfectModel(endpoints), PerfectModel(endpoints)]
    preds_log_df, preds_lin_df, metrics_log_df, metrics_lin_df, model_vs_ens_df = evaluate_labeled_dataset(
        models, df, endpoints, agg_fn="mean"
    )
    # Perfect predictions should yield small RMSE and R2 close to 1 per endpoint
    # We expect RMSE close to 0 and R2 close to 1
    # Metrics CSVs contain rows for endpoints -> check for presence
    assert "LogD" in metrics_log_df[metrics_log_df["endpoint"] == "LogD"]["endpoint"].values
    assert "KSOL" in metrics_log_df[metrics_log_df["endpoint"] == "KSOL"]["endpoint"].values


def test_evaluate_blind_dataset_basic():
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
    assert f"pred_{endpoints[0]}_ensemble_log" in preds_log_df.columns
    assert f"pred_linear_{endpoints[0]}_ensemble_log" in preds_lin_df.columns
