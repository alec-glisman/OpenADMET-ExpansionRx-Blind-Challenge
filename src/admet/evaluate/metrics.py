"""Metric computations for multi-output regression with masking.

Supports log-space and linear-space evaluation (10^x for non-LogD).
"""

from __future__ import annotations

from typing import Dict, Sequence
import numpy as np


def _apply_linear_transform(y: np.ndarray, endpoints: Sequence[str]) -> np.ndarray:
    out = y.copy()
    for j, ep in enumerate(endpoints):
        if ep != "LogD":  # transform log10 -> linear
            out[:, j] = np.power(10.0, out[:, j])
    return out


def _masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert y_true.shape == y_pred.shape == mask.shape
    # Flatten only valid entries
    valid = mask == 1
    return y_true[valid], y_pred[valid]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float("nan")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    endpoints: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-endpoint metrics (log space) and macro averages.

    Returns dict: { endpoint: { mae, rmse, r2 }, "macro": {...} }
    """
    results: Dict[str, Dict[str, float]] = {}
    per_mae = []
    per_rmse = []
    per_r2 = []
    for j, ep in enumerate(endpoints):
        mt, mp = _masked(y_true[:, j : j + 1], y_pred[:, j : j + 1], mask[:, j : j + 1])
        m_mae = mae(mt, mp)
        m_rmse = rmse(mt, mp)
        m_r2 = r2(mt, mp)
        results[ep] = {"mae": m_mae, "rmse": m_rmse, "r2": m_r2}
        if not np.isnan(m_mae):
            per_mae.append(m_mae)
        if not np.isnan(m_rmse):
            per_rmse.append(m_rmse)
        if not np.isnan(m_r2):
            per_r2.append(m_r2)
    results["macro"] = {
        "mae": float(np.mean(per_mae)) if per_mae else float("nan"),
        "rmse": float(np.mean(per_rmse)) if per_rmse else float("nan"),
        "r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
    }
    return results


def compute_metrics_log_and_linear(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    endpoints: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute metrics in both log space and linear space.

    Output structure: { endpoint: { "log": {...}, "linear": {...} }, "macro": {...} }
    """
    log_metrics = compute_metrics(y_true, y_pred, mask, endpoints)
    y_true_lin = _apply_linear_transform(y_true, endpoints)
    y_pred_lin = _apply_linear_transform(y_pred, endpoints)
    linear_metrics = compute_metrics(y_true_lin, y_pred_lin, mask, endpoints)
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ep in endpoints + ["macro"]:
        combined[ep] = {"log": log_metrics[ep], "linear": linear_metrics[ep]}
    return combined


__all__ = [
    "compute_metrics",
    "compute_metrics_log_and_linear",
]
