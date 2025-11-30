"""
Plotting utilities for ADMET cluster CV diagnostics
==================================================

This module provides helper functions for visualizing cluster-based
cross-validation diagnostics, including:

- Global cluster size distribution (histogram/bar plot)
- Training cluster size distributions across folds (boxplots)
- Training vs. validation dataset sizes per fold (bar plots)
- Finite value counts per endpoint for train/validation splits

Design notes
------------
- Functions return a tuple of the created matplotlib `Figure` and `Axes` to
    allow downstream saving or further customization.
- Matplotlib logger level is set to WARNING to avoid noisy INFO/DEBUG logs
    when the application uses more verbose logging.

"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pandas import DataFrame

from admet.plot.latex import latex_sanitize

LOGGER = logging.getLogger(__name__)
# Ensure matplotlib internal logging remains quiet even when the user sets
# a global logging level for the application. This keeps matplotlib's
# internal INFO/DEBUG logs from flooding the console during plotting.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
N_COLS: int = 3
FIGURE_SIZE_PER_CELL: Tuple[int, int] = (6, 4)
FIGURE_DPI: int = 300


def plot_cluster_size_histogram(
    cluster_sizes: np.ndarray,
    title: str = "Cluster size distribution",
) -> Tuple[Figure, Axes]:
    """
    Create a bar plot of the largest cluster sizes after sorting.

    Parameters
    ----------
    cluster_sizes : numpy.ndarray
        Array of cluster sizes (one value per cluster).
    title : str, optional
        Title for the figure.

    Returns
    -------
    Figure
        Matplotlib figure for the plot.
    Axes
        Matplotlib axes for the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_PER_CELL, dpi=FIGURE_DPI, constrained_layout=True)

    # sort cluster sizes descending
    sorted_indices = np.argsort(-cluster_sizes)
    cluster_sizes = cluster_sizes[sorted_indices]

    # add a bar chart for first 100 clusters
    sns.barplot(
        x=np.arange(min(100, len(cluster_sizes))),
        y=cluster_sizes[:100],
        ax=ax,
        color="skyblue",
    )
    # add a text label for the number of clusters with box for readability
    ax.text(
        0.7,
        0.9,
        f"Total clusters: {len(cluster_sizes)}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85),
    )

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    ax.set_xlabel("Cluster rank (sorted)")
    ax.set_ylabel("Cluster size (number of molecules)")
    ax.set_title(title)

    # make y-axis log scale
    ax.set_yscale("log")

    # constrained_layout=True will handle spacing and layout
    return fig, ax


def plot_train_cluster_size_boxplots(
    fold_train_cluster_sizes: Dict[int, np.ndarray],
    title: str = "Training cluster size distribution across folds",
) -> Tuple[Figure, Axes]:
    """
    Create boxplots of training cluster sizes across folds.

    Parameters
    ----------
    fold_train_cluster_sizes : Dict[int, numpy.ndarray]
        Mapping from fold id to array of cluster sizes for the training set.
    title : str, optional
        Title for the figure.

    Returns
    -------
    Figure
        Matplotlib figure for the plot.
    Axes
        Matplotlib axes for the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_PER_CELL, dpi=FIGURE_DPI, constrained_layout=True)

    # Sort by fold id for deterministic order
    fold_ids = sorted(fold_train_cluster_sizes.keys())
    data = [fold_train_cluster_sizes[fid] for fid in fold_ids]

    # Seaborn accepts a list-of-arrays; draw on provided axis
    sns.boxplot(data=data, ax=ax, showfliers=True)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Cluster size (number of molecules)")
    ax.set_title(title)

    # constrained_layout=True handles layout
    return fig, ax


def plot_train_val_dataset_sizes(
    fold_train_sizes: List[int],
    fold_val_sizes: List[int],
    title: str = "Training and Validation Set Sizes Across Folds",
) -> Tuple[Figure, Axes]:
    """
    Create bar plots of training and validation set sizes across folds.

    Parameters
    ----------
    fold_train_sizes : List[int]
        Number of molecules in the training set per fold.
    fold_val_sizes : List[int]
        Number of molecules in the validation set per fold.
    title : str, optional
        Title for the figure.

    Returns
    -------
    Figure
        Matplotlib figure for the plot.
    Axes
        Matplotlib axes for the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_PER_CELL, dpi=FIGURE_DPI)

    # Input is expected as a list of sizes where the index corresponds to the fold id
    if len(fold_train_sizes) != len(fold_val_sizes):
        raise ValueError("fold_train_sizes and fold_val_sizes must have equal length")
    n_folds = len(fold_train_sizes)
    fold_ids = list(range(n_folds))
    train_sizes = fold_train_sizes
    val_sizes = fold_val_sizes

    x = np.arange(len(fold_ids))  # the label locations
    width = 0.35  # the width of the bars

    ax.bar(x - width / 2, train_sizes, width, label="Training Set Size")
    ax.bar(x + width / 2, val_sizes, width, label="Validation Set Size")

    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_yscale("log")
    ax.set_ylabel("Number of Molecules")
    ax.set_title(title)
    legend = ax.legend()
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(0.85)

    fig.tight_layout()
    return fig, ax


def plot_endpoint_finite_value_counts(
    df_train: DataFrame,
    df_test: DataFrame,
    target_cols: List[str],
    title: str = "Finite Value Counts per Endpoint",
) -> Tuple[Figure, Axes]:
    """
    Create a bar plot showing counts of finite (non-NaN) values per endpoint.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training split dataframe.
    df_test : pandas.DataFrame
        Validation split dataframe.
    target_cols : List[str]
        Endpoint column names to evaluate.
    title : str, optional
        Title for the figure.

    Returns
    -------
    Figure
        Matplotlib figure for the plot.
    Axes
        Matplotlib axes for the plot.
    """
    fig, axes = plt.subplots(figsize=(9, 6), dpi=FIGURE_DPI)

    x = np.arange(len(target_cols))
    width = 0.35
    train_counts = [df_train[col].notna().sum() for col in target_cols]
    val_counts = [df_test[col].notna().sum() for col in target_cols]

    axes.bar(x - width / 2, train_counts, width, label="Train", color="skyblue")
    axes.bar(x + width / 2, val_counts, width, label="Validation", color="salmon")

    axes.set_ylabel("Number of Non-NaN Entries", fontsize=14)
    axes.set_title(title, fontsize=16)
    axes.set_xticks(x)
    axes.set_xticklabels([latex_sanitize(c) for c in target_cols], rotation=30, ha="right", fontsize=12)
    axes.set_yscale("log")
    legend = axes.legend()
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(0.85)

    # add number labels on top of bars
    for i, _ in enumerate(x):
        axes.text(
            x[i] - width / 2,
            train_counts[i] * 1.1,
            f"{train_counts[i]:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
        )
        axes.text(
            x[i] + width / 2,
            val_counts[i] * 1.1,
            f"{val_counts[i]:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
        )

    # add percentage of data in test set
    for i, _ in enumerate(x):
        if train_counts[i] + val_counts[i] > 0:
            percent_test = (val_counts[i] / (train_counts[i] + val_counts[i])) * 100
            axes.text(
                x[i] + width / 1.1,
                val_counts[i] * 3,
                f"{percent_test:.0f}" + r"\%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="green",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black", alpha=0.85),
            )

    # constrained_layout used at creation ensures correct spacing
    return fig, axes
