"""
Metric Bar Chart Utilities
==========================

Generate bar charts comparing model performance metrics across endpoints
and dataset splits.

This module provides functions to create grouped bar charts for metrics
like MAE, RMSE, R², Pearson r, Spearman ρ, and Kendall τ.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from admet.data.stats import correlation
from admet.plot import GLASBEY_PALETTE
from admet.plot.latex import latex_sanitize

logger = logging.getLogger(__name__)

# Standard metrics to plot (tuple to avoid mutable default argument issues)
METRIC_NAMES: Tuple[str, ...] = ("mae", "rae", "mape", "rmse", "R2", "pearson_r", "spearman_rho", "kendall_tau")

# Display labels for metrics (LaTeX-formatted)
METRIC_LABELS = {
    "mae": "MAE",
    "rae": "RAE",
    "rmse": "RMSE",
    "R2": r"$R^2$",
    "pearson_r": r"Pearson $r$",
    "spearman_rho": r"Spearman $\rho$",
    "kendall_tau": r"Kendall $\tau$",
}

# Y-axis limits for bounded metrics
METRIC_YLIMS = {
    "rae": (0.0, 2.0),
    "R2": (-0.5, 1.0),
    "pearson_r": (-1.0, 1.0),
    "spearman_rho": (-1.0, 1.0),
    "kendall_tau": (-1.0, 1.0),
}


def _safe_slug(name: str) -> str:
    """Return a filesystem-safe slug for a name."""
    replacements = {" ": "_", "/": "_", "<": "lt", ">": "gt", ":": "_"}
    out = name
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


def compute_metrics_df(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    endpoints: Sequence[str],
) -> pd.DataFrame:
    """Compute correlation metrics for each endpoint.

    Parameters
    ----------
    df_true : pandas.DataFrame
        Ground truth values.
    df_pred : pandas.DataFrame
        Predicted values.
    endpoints : sequence of str
        Endpoint column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with endpoints as index and metrics as columns.
    """
    records = []
    for endpoint in endpoints:
        if endpoint not in df_true.columns or endpoint not in df_pred.columns:
            continue

        y_true = np.asarray(df_true[endpoint].values)
        y_pred = np.asarray(df_pred[endpoint].values)

        metrics = correlation(y_true, y_pred)
        record = {"endpoint": endpoint, **metrics}
        records.append(record)

    return pd.DataFrame(records).set_index("endpoint")


def compute_metrics_by_split(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    endpoints: Sequence[str],
    splits: Sequence[str] = ("train", "validation", "test"),
) -> Dict[str, pd.DataFrame]:
    """Compute metrics for each split.

    Parameters
    ----------
    y_true_dict : dict[str, numpy.ndarray]
        Mapping split -> array of shape (N, D).
    y_pred_dict : dict[str, numpy.ndarray]
        Mapping split -> array of shape (N, D).
    endpoints : sequence of str
        Endpoint names (length D).
    splits : sequence of str
        Split names to process.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping split -> metrics DataFrame.
    """
    results = {}
    for split in splits:
        y_true = y_true_dict.get(split)
        y_pred = y_pred_dict.get(split)
        if y_true is None or y_pred is None:
            continue

        records = []
        for j, endpoint in enumerate(endpoints):
            y_t = y_true[:, j]
            y_p = y_pred[:, j]
            metrics = correlation(y_t, y_p)
            record = {"endpoint": endpoint, **metrics}
            records.append(record)

        results[split] = pd.DataFrame(records).set_index("endpoint")

    return results


def plot_metric_bar(
    values: np.ndarray,
    labels: Sequence[str],
    metric_name: str,
    *,
    errors: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_values: bool = True,
    show_mean: bool = True,
    value_fontsize: int = 9,
) -> Tuple[Figure, Axes]:
    """Create a single metric bar chart.

    Parameters
    ----------
    values : numpy.ndarray
        Metric values for each bar.
    labels : sequence of str
        Bar labels (e.g., endpoint names).
    metric_name : str
        Metric name for axis label.
    errors : numpy.ndarray, optional
        Error bar values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    color : str, optional
        Bar color.
    ylim : tuple[float, float], optional
        Y-axis limits.
    show_values : bool, default=True
        Whether to show values above bars.
    show_mean : bool, default=True
        Whether to show a mean bar with standard error.
    value_fontsize : int, default=9
        Font size for value labels.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes objects.
    """
    # Filter out NaN values for mean calculation
    valid_values = values[~np.isnan(values)]

    # Extend labels and values to include mean if requested
    if show_mean and len(valid_values) > 0:
        mean_val = np.mean(valid_values)
        std_err = np.std(valid_values, ddof=1) / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0.0
        extended_labels = list(labels) + ["Mean"]
        extended_values = np.append(values, mean_val)
        mean_error = std_err
    else:
        extended_labels = list(labels)
        extended_values = values
        mean_error = None

    if ax is None:
        fig_w = max(6, len(extended_labels) * 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, 5))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    x = np.arange(len(extended_labels))
    bar_color = color or GLASBEY_PALETTE[0]

    # Set up colors - use different color for mean bar
    colors = [bar_color] * len(labels)
    if show_mean and len(valid_values) > 0:
        colors.append(GLASBEY_PALETTE[1])  # Different color for mean

    # Set up error bars
    if errors is not None:
        extended_errors = list(errors) + ([mean_error] if show_mean and len(valid_values) > 0 else [])
    elif show_mean and len(valid_values) > 0:
        extended_errors = [0] * len(labels) + [mean_error]
    else:
        extended_errors = None

    bars = ax.bar(
        x,
        extended_values,
        yerr=extended_errors,
        capsize=5 if extended_errors is not None else 0,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([latex_sanitize(lbl) for lbl in extended_labels], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name), fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if ylim:
        ax.set_ylim(ylim)
    elif metric_name in METRIC_YLIMS:
        ax.set_ylim(METRIC_YLIMS[metric_name])

    if title:
        ax.set_title(title, fontsize=14)

    # Add value labels
    if show_values:
        for i, (rect, val) in enumerate(zip(bars, extended_values)):
            if np.isnan(val):
                continue
            height = rect.get_height()
            y_text = height + (0.02 if height >= 0 else -0.02)
            # Adjust for error bar if present
            if extended_errors is not None and i < len(extended_errors) and extended_errors[i]:
                y_text = height + extended_errors[i] + 0.02
            va = "bottom" if height >= 0 else "top"
            # Show ± std err for mean bar
            if show_mean and i == len(extended_values) - 1 and mean_error is not None:
                label_text = f"{val:.2f}±{mean_error:.2f}"
            else:
                label_text = f"{val:.2f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_text,
                label_text,
                ha="center",
                va=va,
                fontsize=value_fontsize,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            )

    return fig, ax  # type: ignore[return-value]


def plot_metrics_grouped_by_split(
    metrics_by_split: Dict[str, pd.DataFrame],
    metric_name: str,
    *,
    splits: Sequence[str] = ("train", "validation", "test"),
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show_mean: bool = True,
) -> Tuple[Figure, Axes]:
    """Create a grouped bar chart comparing a metric across splits.

    Parameters
    ----------
    metrics_by_split : dict[str, pandas.DataFrame]
        Mapping split -> metrics DataFrame with endpoints as index.
    metric_name : str
        Which metric to plot.
    splits : sequence of str
        Split names to include.
    figsize : tuple[float, float], optional
        Figure size.
    dpi : int, default=150
        Figure DPI.
    title : str, optional
        Plot title.
    save_path : pathlib.Path, optional
        If provided, save the figure.
    show_mean : bool, default=True
        Whether to show a mean bar with standard error for each split.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    # Get endpoints from first available split
    endpoints = None
    for split in splits:
        if split in metrics_by_split:
            endpoints = list(metrics_by_split[split].index)
            break

    if endpoints is None:
        raise ValueError("No valid splits found in metrics_by_split")

    # Add "Mean" to endpoints if show_mean is True
    extended_endpoints = list(endpoints) + (["Mean"] if show_mean else [])

    n_endpoints = len(extended_endpoints)
    n_splits = len([s for s in splits if s in metrics_by_split])

    if figsize is None:
        fig_w = max(8, n_endpoints * 1.2)
        figsize = (fig_w, 6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(n_endpoints)
    width = 0.8 / n_splits
    offset = -width * (n_splits - 1) / 2

    for i, split in enumerate(splits):
        if split not in metrics_by_split:
            continue

        df = metrics_by_split[split]
        values = np.array([df.loc[ep, metric_name] if ep in df.index else np.nan for ep in endpoints])

        # Calculate mean and std error
        valid_values = values[~np.isnan(values)]
        std_err: Optional[float] = None
        if show_mean and len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_err = np.std(valid_values, ddof=1) / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0.0
            extended_values = np.append(values, mean_val)
            yerr = [0.0] * len(values) + [std_err]
        else:
            extended_values = values
            yerr = None

        rects = ax.bar(
            x + offset + i * width,
            extended_values,
            width,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            label=split.capitalize(),
            color=GLASBEY_PALETTE[i],
            edgecolor="black",
            alpha=0.8,
        )

        # Add value labels
        for j, rect in enumerate(rects):
            height = rect.get_height()
            if np.isnan(height):
                continue
            va = "bottom" if height >= 0 else "top"
            # Adjust y position for error bar on mean
            y_offset: float = 3.0
            if show_mean and j == len(extended_values) - 1 and yerr is not None and yerr[j] is not None:
                y_offset = float(yerr[j]) * 50 + 3.0  # type: ignore[arg-type]  # Scale for visibility
            # Show ± std err for mean bar
            if show_mean and j == len(extended_values) - 1 and len(valid_values) > 0 and std_err is not None:
                label_text = f"{height:.2f}±{std_err:.2f}"
            else:
                label_text = f"{height:.2f}"
            ax.annotate(
                label_text,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, y_offset if height >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([latex_sanitize(ep) for ep in extended_endpoints], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name), fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if metric_name in METRIC_YLIMS:
        ax.set_ylim(METRIC_YLIMS[metric_name])

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{METRIC_LABELS.get(metric_name, metric_name)} by Endpoint", fontsize=14)

    ax.legend(title="Split", frameon=True, edgecolor="black")

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax  # type: ignore[return-value]


def plot_all_metrics(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    endpoints: Sequence[str],
    output_dir: Path,
    *,
    metrics: Sequence[str] = METRIC_NAMES,
    dpi: int = 300,
    title_prefix: str = "",
) -> None:
    """Generate and save bar charts for all metrics.

    Parameters
    ----------
    df_true : pandas.DataFrame
        Ground truth values.
    df_pred : pandas.DataFrame
        Predicted values.
    endpoints : sequence of str
        Endpoint column names.
    output_dir : pathlib.Path
        Directory to save plots.
    metrics : sequence of str
        Which metrics to plot.
    dpi : int, default=150
        Figure DPI.
    title_prefix : str, default=''
        Prefix for plot titles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = compute_metrics_df(df_true, df_pred, endpoints)
    available_endpoints = list(metrics_df.index)

    for metric_name in metrics:
        if metric_name not in metrics_df.columns:
            logger.warning("Metric '%s' not found; skipping.", metric_name)
            continue

        values = np.asarray(metrics_df[metric_name].values)
        title = f"{title_prefix}{METRIC_LABELS.get(metric_name, metric_name)}" if title_prefix else None

        fig, _ = plot_metric_bar(
            values,
            available_endpoints,
            metric_name,
            title=title,
        )

        fname = f"metrics_{_safe_slug(metric_name)}.png"
        fig.savefig(output_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved %d metric plots to %s", len(metrics), output_dir)


def plot_all_metrics_by_split(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    endpoints: Sequence[str],
    output_dir: Path,
    *,
    splits: Sequence[str] = ("train", "validation", "test"),
    metrics: Sequence[str] = METRIC_NAMES,
    dpi: int = 300,
) -> None:
    """Generate and save grouped bar charts for all metrics by split.

    Parameters
    ----------
    y_true_dict : dict[str, numpy.ndarray]
        Mapping split -> ground truth array (N, D).
    y_pred_dict : dict[str, numpy.ndarray]
        Mapping split -> prediction array (N, D).
    endpoints : sequence of str
        Endpoint names.
    output_dir : pathlib.Path
        Directory to save plots.
    splits : sequence of str
        Which splits to include.
    metrics : sequence of str
        Which metrics to plot.
    dpi : int, default=150
        Figure DPI.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_split = compute_metrics_by_split(y_true_dict, y_pred_dict, endpoints, splits)

    for metric_name in metrics:
        try:
            fig, _ = plot_metrics_grouped_by_split(
                metrics_by_split,
                metric_name,
                splits=splits,
                dpi=dpi,
            )

            fname = f"metrics_{_safe_slug(metric_name)}_by_split.png"
            fig.savefig(output_dir / fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to plot metric '%s': %s", metric_name, e)

    logger.info("Saved %d grouped metric plots to %s", len(metrics), output_dir)


def metrics_to_latex_table(
    metrics_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = METRIC_NAMES,
    caption: str = "Model Performance Metrics",
    label: str = "tab:metrics",
) -> str:
    """Convert metrics DataFrame to a LaTeX table string.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics with endpoints as index.
    metrics : sequence of str
        Which metrics to include.
    caption : str
        Table caption.
    label : str
        Table label for references.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Filter to requested metrics
    cols = [m for m in metrics if m in metrics_df.columns]
    df = metrics_df[cols].copy()

    # Rename columns to pretty names
    df.columns = [METRIC_LABELS.get(c, c) for c in df.columns]

    latex = df.to_latex(
        float_format="%.3f",
        caption=caption,
        label=label,
        escape=False,
    )
    return latex
