from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


class DistributionStats(TypedDict):
    min: float
    max: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    count: int


class CorrelationMetrics(TypedDict):
    mae: float
    rae: float
    mape: float
    rmse: float
    R2: float
    pearson_r: float
    spearman_rho: float
    kendall_tau: float


def distribution(array: np.ndarray) -> DistributionStats:
    """Return formatted summary statistics for a numeric series.

    Parameters
    ----------
    series : pandas.Series
        Numeric series (non-numeric entries should be coerced prior to call).

    Returns
    -------
    DistributionStats
        Dictionary with keys: ``min``, ``max``, ``mean``, ``median``, ``std``,
        ``skew``, ``kurtosis``, ``count``.
    """
    array = np.asarray(array).ravel()
    array = array[~np.isnan(array)]
    count = array.shape[0]
    if count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
            "count": 0,
        }
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)),
        "skew": float(stats.skew(array)),
        "kurtosis": float(stats.kurtosis(array)),
        "count": int(count),
    }


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> CorrelationMetrics:
    """Compute metrics for a single endpoint column.

    Reuses :func:`admet.evaluate.metrics.compute_metrics` (called on a single
    pseudo-endpoint) to ensure numeric parity with training evaluation.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        1-D arrays of true and predicted values.

    Returns
    -------
    CorrelationMetrics
        Dictionary with keys: ``mae``, ``rae``, ``mape``, ``rmse``, ``R2``,
        ``pearson_r``, ``spearman_rho``, ``kendall_tau``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Remove NaNs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if y_true.shape[0] == 0:
        return {
            "mae": float("nan"),
            "rae": float("nan"),
            "mape": float("nan"),
            "rmse": float("nan"),
            "R2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
        }

    mae = mean_absolute_error(y_true, y_pred)
    mae_baseline = mean_absolute_error(y_true, np.full_like(y_true, np.mean(y_true)))
    rae = mae / mae_baseline if mae_baseline != 0 else float("nan")

    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    pearson_r = pearsonr(y_true, y_pred).statistic if y_true.size >= 2 else float("nan")
    spearman_rho = spearmanr(y_true, y_pred).statistic if y_true.size >= 2 else float("nan")
    kendall_tau = kendalltau(y_true, y_pred).statistic if y_true.size >= 2 else float("nan")

    return {
        "mae": float(mae),
        "rae": float(rae),
        "mape": float(mape),
        "rmse": float(rmse),
        "R2": float(r2),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "kendall_tau": float(kendall_tau),
    }
