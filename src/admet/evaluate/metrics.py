"""Metric computations for multi-output regression with masking.

Supports log-space and linear-space evaluation (10^x for non-LogD).
"""

from __future__ import annotations

from typing import Dict, Sequence, TypedDict
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr, kendalltau


def _apply_linear_transform(y: np.ndarray, endpoints: Sequence[str]) -> np.ndarray:
    out = y.copy()
    for j, ep in enumerate(endpoints):
        if ep != "LogD":  # transform log10 -> linear
            out[:, j] = np.power(10.0, out[:, j])
    return out


def _masked(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert y_true.shape == y_pred.shape == mask.shape
    # Flatten only valid entries
    # Accept both integer and boolean masks
    valid = mask.astype(bool)
    return y_true[valid], y_pred[valid]


Endpoints = Sequence[str]


class EndpointMetrics(TypedDict):
    mae: float
    rmse: float
    R2: float
    pearson_r2: float
    spearman_rho2: float
    kendall_tau: float


SplitMetrics = Dict[str, EndpointMetrics]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    endpoints: Endpoints,
) -> SplitMetrics:
    """Compute per-endpoint metrics (log space) and macro averages.

    Returns dict: { endpoint: { mae, rmse, r2 }, "macro": {...} }
    """
    results: Dict[str, Dict[str, float]] = {}

    per_mae = []
    per_rmse = []
    per_r2 = []
    per_pearson = []
    per_spearman = []
    per_tau = []
    for j, ep in enumerate(endpoints):
        mt, mp = _masked(y_true[:, j : j + 1], y_pred[:, j : j + 1], mask[:, j : j + 1])
        m_mae = mean_absolute_error(mt, mp)
        m_rmse = root_mean_squared_error(mt, mp)
        m_R2 = r2_score(mt, mp)
        m_r2 = pearsonr(mt, mp).statistic ** 2 if mt.size >= 2 else float("nan")
        m_rho2 = spearmanr(mt, mp).statistic ** 2 if mt.size >= 2 else float("nan")
        m_tau = kendalltau(mt, mp).statistic if mt.size >= 2 else float("nan")

        results[ep] = {
            "mae": float(m_mae),
            "rmse": float(m_rmse),
            "R2": float(m_R2),
            "pearson_r2": float(m_r2),
            "spearman_rho2": float(m_rho2),
            "kendall_tau": float(m_tau),
        }
        per_mae.append(m_mae)
        per_rmse.append(m_rmse)
        per_r2.append(m_R2)
        per_pearson.append(m_r2)
        per_spearman.append(m_rho2)
        per_tau.append(m_tau)

    results["macro"] = {
        "mae": float(np.nanmean(per_mae)) if per_mae else float("nan"),
        "rmse": float(np.nanmean(per_rmse)) if per_rmse else float("nan"),
        "R2": float(np.nanmean(per_r2)) if per_r2 else float("nan"),
        "pearson_r2": float(np.nanmean(per_pearson)) if per_pearson else float("nan"),
        "spearman_rho2": float(np.nanmean(per_spearman)) if per_spearman else float("nan"),
        "kendall_tau": float(np.nanmean(per_tau)) if per_tau else float("nan"),
    }
    return results


EndpointSpaceMetrics = Dict[str, Dict[str, float]]


def compute_metrics_log_and_linear(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    endpoints: Endpoints,
) -> Dict[str, EndpointSpaceMetrics]:
    """Compute metrics in both log space and linear space.

    Output structure: { endpoint: { "log": {...}, "linear": {...} }, "macro": {...} }
    """
    log_metrics = compute_metrics(y_true, y_pred, mask, endpoints)
    y_true_lin = _apply_linear_transform(y_true, endpoints)
    y_pred_lin = _apply_linear_transform(y_pred, endpoints)
    linear_metrics = compute_metrics(y_true_lin, y_pred_lin, mask, endpoints)
    combined: Dict[str, EndpointSpaceMetrics] = {}
    for ep in list(endpoints) + ["macro"]:
        combined[ep] = {"log": log_metrics[ep], "linear": linear_metrics[ep]}
    return combined


__all__ = [
    "compute_metrics",
    "compute_metrics_log_and_linear",
    "EndpointMetrics",
    "SplitMetrics",
    "EndpointSpaceMetrics",
    "AllMetrics",
]


# AllMetrics is the full run-level metrics structure: per-split mapping to endpoint-space metrics.
AllMetrics = Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
