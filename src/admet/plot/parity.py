"""
Parity Plot Utilities
=====================

Generate parity plots (true vs predicted) for model evaluation with
correlation metrics overlay.

This module provides functions to create publication-quality parity plots
for single endpoints or grids of multiple endpoints across dataset splits.
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

from admet.plot import GLASBEY_PALETTE
from admet.plot.latex import latex_sanitize, text_correlation

logger = logging.getLogger(__name__)


def _safe_slug(name: str) -> str:
    """Return a filesystem-safe slug for a name."""
    replacements = {" ": "_", "/": "_", "<": "lt", ">": "gt", ":": "_"}
    out = name
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    yerr: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "True",
    ylabel: str = "Predicted",
    show_stats: bool = True,
    show_identity: bool = True,
    alpha: float = 0.6,
    s: int = 20,
    color: Optional[str] = None,
    max_points: int = 10_000,
    elinewidth: float = 0.5,
    capsize: float = 2,
) -> Tuple[Figure, Axes]:
    """Create a single parity plot with optional statistics overlay.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values (1-D array).
    y_pred : numpy.ndarray
        Predicted values (1-D array).
    yerr : numpy.ndarray, optional
        Error values for y_pred (e.g., standard error from ensemble).
        If provided, error bars are drawn instead of scatter points.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; creates new figure if None.
    title : str, optional
        Plot title.
    xlabel : str, default='True'
        X-axis label.
    ylabel : str, default='Predicted'
        Y-axis label.
    show_stats : bool, default=True
        Whether to display correlation metrics in a text box.
    show_identity : bool, default=True
        Whether to draw the y=x identity line.
    alpha : float, default=0.6
        Point transparency.
    s : int, default=20
        Point size (used for scatter, or markersize=sqrt(s) for errorbar).
    color : str, optional
        Point color (defaults to first palette color).
    max_points : int, default=10_000
        Maximum points to plot (randomly sampled if exceeded).
    elinewidth : float, default=0.5
        Error bar line width (only used when yerr is provided).
    capsize : float, default=2
        Error bar cap size (only used when yerr is provided).

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes objects.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if yerr is not None:
        yerr = np.asarray(yerr).ravel()

    # Remove NaN values
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if yerr is not None:
        valid = valid & ~np.isnan(yerr)
        yerr = yerr[valid]
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    if y_true.size == 0:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
        )
        if title:
            ax.set_title(title)
        return fig, ax  # type: ignore[return-value]

    # Subsample if needed
    if y_true.size > max_points:
        idx = np.random.choice(y_true.size, size=max_points, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
        yerr_plot = yerr[idx] if yerr is not None else None
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        yerr_plot = yerr

    # Determine axis limits
    combined = np.concatenate([y_true, y_pred])
    lo = float(np.nanpercentile(combined, 1))
    hi = float(np.nanpercentile(combined, 99))
    pad = 0.1 * (hi - lo) if hi > lo else 0.1 * abs(lo) if lo != 0 else 1.0
    lim_min, lim_max = lo - pad, hi + pad

    # Plot
    point_color = color or GLASBEY_PALETTE[0]
    if yerr_plot is not None:
        # Use errorbar for ensemble predictions with uncertainty
        ax.errorbar(
            y_true_plot,
            y_pred_plot,
            yerr=yerr_plot,
            fmt="o",
            alpha=alpha,
            markersize=np.sqrt(s),
            color=point_color,
            elinewidth=elinewidth,
            capsize=capsize,
        )
    else:
        ax.scatter(y_true_plot, y_pred_plot, alpha=alpha, s=s, color=point_color)

    if show_identity:
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", lw=1.5)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    if title:
        ax.set_title(title, fontsize=14)

    if show_stats:
        stats_text = text_correlation(y_true, y_pred)
        stats_text = f"$N$: {y_true.size}\n" + stats_text
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
        )

    return fig, ax  # type: ignore[return-value]


def plot_parity_by_split(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    endpoint: str,
    *,
    splits: Sequence[str] = ("train", "validation", "test"),
    figsize: Tuple[float, float] = (15, 5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
    **kwargs,
) -> Tuple[Figure, np.ndarray]:
    """Create parity plots for an endpoint across multiple splits.

    Parameters
    ----------
    y_true_dict : dict[str, numpy.ndarray]
        Mapping from split name to ground truth arrays.
    y_pred_dict : dict[str, numpy.ndarray]
        Mapping from split name to prediction arrays.
    endpoint : str
        Endpoint name for the title.
    splits : sequence of str, default=('train', 'validation', 'test')
        Which splits to plot.
    figsize : tuple[float, float], default=(15, 5)
        Figure size.
    dpi : int, default=300
        Figure DPI.
    save_path : pathlib.Path, optional
        If provided, save the figure to this path.
    **kwargs
        Additional arguments passed to plot_parity.

    Returns
    -------
    tuple[Figure, numpy.ndarray]
        Figure and array of axes.
    """
    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes)

    endpoint_latex = latex_sanitize(endpoint)
    fig.suptitle(f"Parity: {endpoint_latex}", fontsize=16)

    for ax, split in zip(axes, splits):
        y_true = y_true_dict.get(split, np.array([]))
        y_pred = y_pred_dict.get(split, np.array([]))

        plot_parity(
            y_true,
            y_pred,
            ax=ax,
            title=split.capitalize(),
            color=GLASBEY_PALETTE[list(splits).index(split)],
            **kwargs,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


def plot_parity_grid(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    endpoints: Sequence[str],
    *,
    n_cols: int = 3,
    figsize_per_cell: Tuple[float, float] = (5, 5),
    dpi: int = 300,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    **kwargs,
) -> Tuple[Figure, np.ndarray]:
    """Create a grid of parity plots for multiple endpoints.

    Parameters
    ----------
    df_true : pandas.DataFrame
        DataFrame with ground truth values (columns are endpoints).
    df_pred : pandas.DataFrame
        DataFrame with predicted values (columns are endpoints).
    endpoints : sequence of str
        Endpoint column names to plot.
    n_cols : int, default=3
        Number of columns in the grid.
    figsize_per_cell : tuple[float, float], default=(5, 5)
        Size per subplot.
    dpi : int, default=300
        Figure DPI.
    title : str, optional
        Figure super-title.
    save_path : pathlib.Path, optional
        If provided, save the figure.
    **kwargs
        Additional arguments passed to plot_parity.

    Returns
    -------
    tuple[Figure, numpy.ndarray]
        Figure and array of axes.
    """
    n_endpoints = len(endpoints)
    n_rows = int(np.ceil(n_endpoints / n_cols))
    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.atleast_1d(np.array(axes).flatten())

    for i, endpoint in enumerate(endpoints):
        ax = axes[i]
        y_true = df_true[endpoint].values if endpoint in df_true.columns else np.array([])
        y_pred = df_pred[endpoint].values if endpoint in df_pred.columns else np.array([])

        endpoint_latex = latex_sanitize(endpoint)
        plot_parity(
            np.asarray(y_true),
            np.asarray(y_pred),
            ax=ax,
            title=endpoint_latex,
            color=GLASBEY_PALETTE[i % len(GLASBEY_PALETTE)],
            **kwargs,
        )

    # Hide unused axes
    n_used = len(endpoints)
    for j in range(n_used, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


def save_parity_plots(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    endpoints: Sequence[str],
    output_dir: Path,
    *,
    dpi: int = 300,
    **kwargs,
) -> None:
    """Save individual parity plots for each endpoint.

    Parameters
    ----------
    df_true : pandas.DataFrame
        Ground truth values.
    df_pred : pandas.DataFrame
        Predicted values.
    endpoints : sequence of str
        Endpoint names.
    output_dir : pathlib.Path
        Directory to save plots.
    dpi : int, default=300
        Figure DPI.
    **kwargs
        Additional arguments passed to plot_parity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, endpoint in enumerate(endpoints):
        if endpoint not in df_true.columns or endpoint not in df_pred.columns:
            logger.warning("Endpoint '%s' not found in dataframes; skipping.", endpoint)
            continue

        y_true = np.asarray(df_true[endpoint].values)
        y_pred = np.asarray(df_pred[endpoint].values)

        fig, _ = plot_parity(
            y_true,
            y_pred,
            title=latex_sanitize(endpoint),
            color=GLASBEY_PALETTE[i % len(GLASBEY_PALETTE)],
            **kwargs,
        )

        fname = f"parity_{_safe_slug(endpoint)}.png"
        fig.savefig(output_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved %d parity plots to %s", len(endpoints), output_dir)
