import numpy as np
from admet.evaluate.metrics import compute_metrics_log_and_linear
import pytest


def test_metrics_log_and_linear_masking():
    endpoints = ["LogD", "KSOL"]
    y_true = np.array([[1.0, 0.0], [2.0, np.nan], [3.0, 1.0]])
    y_pred = np.array([[1.1, 0.2], [1.9, 5.0], [2.9, 0.9]])
    mask = (~np.isnan(y_true)).astype(int)
    metrics = compute_metrics_log_and_linear(y_true, y_pred, mask, endpoints)
    # KSOL second row ignored due to NaN
    if "LogD" not in metrics:
        pytest.fail("LogD not present in metrics")
    if "KSOL" not in metrics:
        pytest.fail("KSOL not present in metrics")
    if np.isnan(metrics["LogD"]["log"]["mae"]):
        pytest.fail("Expected numeric MAE for LogD metric, found NaN")
    # linear transform for KSOL uses 10^x
    if metrics["KSOL"]["linear"]["mae"] < 0.0:
        pytest.fail("Expected non-negative MAE for KSOL linear metric")
