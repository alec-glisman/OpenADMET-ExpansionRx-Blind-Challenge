from __future__ import annotations

import logging

from numpy import ndarray

from admet.data.stats import correlation, distribution

logger = logging.getLogger(__name__)


def latex_sanitize(text: str) -> str:
    """Sanitize text for LaTeX rendering in plots.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Sanitized text.
    """
    return (
        text.replace(">", r"$>$")
        .replace("<", r"$<$")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("10^-6", r"10$^{-6}$")
    )


def text_distribution(array: ndarray) -> str:
    """Generate a LaTeX-formatted string summarizing distribution statistics.

    Parameters
    ----------
    array : numpy.ndarray
        Numeric array.

    Returns
    -------
    str
        LaTeX-formatted summary string.
    """
    stats = distribution(array)
    summary = (
        f"Min: {stats['min']:.2f}\n"
        f"Max: {stats['max']:.2f}\n"
        f"Mean: {stats['mean']:.2f}\n"
        f"Median: {stats['median']:.2f}\n"
        f"Std: {stats['std']:.2f}\n"
        f"Skew: {stats['skew']:.2f}\n"
        f"Kurtosis: {stats['kurtosis']:.2f}\n"
        f"$N$: {stats['count']}"
    )
    return summary


def text_correlation(y_true: ndarray, y_pred: ndarray) -> str:
    """Generate a LaTeX-formatted string summarizing correlation metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        True values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    str
        LaTeX-formatted summary string.
    """
    metrics = correlation(y_true, y_pred)
    summary = (
        f"MAE: {metrics['mae']:.2f}\n"
        f"RAE: {metrics['rae']:.2f}\n"
        f"MAPE: {metrics['mape']:.2f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"$R^2$: {metrics['R2']:.2f}\n"
        f"Pearson $r$: {metrics['pearson_r']:.2f}\n"
        f"Spearman $\\rho$: {metrics['spearman_rho']:.2f}"
    )
    return summary
