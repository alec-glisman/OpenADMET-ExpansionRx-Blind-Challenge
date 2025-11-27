"""Unit tests for metric computation routines.

This module checks the log/linear metrics computation and masking behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from admet.evaluate.metrics import compute_metrics_log_and_linear


@pytest.mark.unit
def test_metrics_log_and_linear_masking() -> None:
    endpoints = ["LogD", "KSOL"]
    y_true = np.array([[1.0, 0.0], [2.0, np.nan], [3.0, 1.0]])
    y_pred = np.array([[1.1, 0.2], [1.9, 5.0], [2.9, 0.9]])
    mask = (~np.isnan(y_true)).astype(int)
    metrics = compute_metrics_log_and_linear(y_true, y_pred, mask, endpoints)
    assert "LogD" in metrics
    assert "KSOL" in metrics
    assert not np.isnan(metrics["LogD"]["log"]["mae"])
    assert metrics["KSOL"]["linear"]["mae"] >= 0.0
