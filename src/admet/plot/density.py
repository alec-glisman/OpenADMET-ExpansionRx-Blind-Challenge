from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame

from admet.plot.latex import latex_sanitize, text_correlation, text_distribution

LOGGER = logging.getLogger(__name__)
N_COLS: int = 3
FIGURE_SIZE_PER_CELL: Tuple[int, int] = (6, 4)


def plot_endpoint_distributions(
    df: DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_cols: int = N_COLS,
    figsize_per_cell: Tuple[int, int] = FIGURE_SIZE_PER_CELL,
    kde: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Tuple[Figure, ndarray]:
    """Plot histograms (with optional KDE) for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    columns : sequence of str, optional
        Explicit column list; by default selects all numeric columns.
    n_cols : int, optional
        Number of columns in subplot grid (default 3).
    figsize_per_cell : tuple[int, int], optional
        Base size per subplot (default (6, 4)).
    kde : bool, optional
        Whether to overlay KDE curve (default True).
    title : str, optional
        Figure super-title.
    save_path : pathlib.Path, optional
        If provided, figure saved to this path.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        Figure and flattened axes array.
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    n_features = len(columns)
    if n_features == 0:
        raise ValueError("No numeric columns to plot.")

    n_rows = int(np.ceil(n_features / float(n_cols)))
    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(np.array(axes).flatten())

    for i, col in enumerate(columns):
        ax = axes[i]
        series = df[col].dropna()
        col_latex = latex_sanitize(col)
        if series.empty:
            LOGGER.warning(f"Column {col} is empty after dropping NaNs; skipping plot.")
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="center",
            )
            ax.set_xlabel(col_latex, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            continue

        sns.histplot(series, kde=kde, ax=ax)

        try:
            stat_text = text_distribution(series.values)
        except Exception:
            LOGGER.exception(f"Error computing stats for column {col}")
            stat_text = ""

        ax.set_xlabel(col_latex, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        if stat_text:
            ax.text(
                0.95,
                0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"),
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, y=1.01)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)

    return fig, axes


def plot_property_distributions(
    props_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 4,
    figsize_per_cell: tuple = (4, 3),
    bins: int = 30,
    kde: bool = True,
    title: str = "Molecular Property Distributions",
    save_path: str | None = None,
) -> Tuple[Figure, ndarray]:
    """Plot histograms for computed molecular property columns.

    Parameters
    ----------
    props_df : pandas.DataFrame
        DataFrame produced by :func:`admet.data.chem.compute_molecular_properties`.
    columns : list[str], optional
        Subset of property columns to plot; defaults to the common property list.
    n_cols : int, optional
        Number of columns in subplot grid (default 4).
    figsize_per_cell : tuple, optional
        Per‑subplot size (default (4, 3)).
    bins : int, optional
        Histogram bin count (default 30).
    kde : bool, optional
        Overlay KDE curve (default True).
    title : str, optional
        Figure super-title.
    save_path : str, optional
        If provided, output PNG file path.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        Figure and flattened axes array.
    """
    if columns is None:
        columns = [
            "MW",
            "TPSA",
            "HBA",
            "HBD",
            "RotBonds",
            "LogP",
            "NumHeavyAtoms",
            "NumRings",
        ]
    cols = [c for c in columns if c in props_df.columns]
    if not cols:
        raise ValueError("No valid property columns found in props_df.")

    n_features = len(cols)
    n_rows = int(np.ceil(n_features / float(n_cols)))
    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(np.array(axes).flatten())

    i = -1
    for i, col in enumerate(cols):
        ax = axes[i]
        series = props_df[col].dropna()
        sns.histplot(series, bins=bins, kde=kde, ax=ax, color="C0")
        stat_text = text_distribution(series.values)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="x", labelrotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        if stat_text:
            ax.text(
                0.95,
                0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"),
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, axes
