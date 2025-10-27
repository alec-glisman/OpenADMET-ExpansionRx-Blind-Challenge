"""Visualization helpers for exploratory data analysis (EDA).

Provides small plotting utilities and statistics calculators to keep notebooks
lightweight and focused on analysis rather than boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from scipy import stats


def calc_stats(series: Series) -> str:
    """Calculate summary statistics for a pandas Series and format as text.

    Parameters
    ----------
    series : pandas.Series
        Input numeric series.

    Returns
    -------
    str
        Multiline string containing common summary statistics.
    """
    array = series.dropna().to_numpy()

    out = {
        "min": np.min(array) if len(array) else np.nan,
        "max": np.max(array) if len(array) else np.nan,
        "mean": np.mean(array) if len(array) else np.nan,
        "median": np.median(array) if len(array) else np.nan,
        "std": np.std(array, ddof=1) if len(array) > 1 else np.nan,
        "skew": stats.skew(array) if len(array) > 2 else np.nan,
        "kurtosis": stats.kurtosis(array) if len(array) > 3 else np.nan,
        "count": int(len(array)),
    }
    stats_str = (
        f"min: {out['min']:.4g}\n"
        f"max: {out['max']:.4g}\n"
        f"mean: {out['mean']:.4g}\n"
        f"median: {out['median']:.4g}\n"
        f"std: {out['std']:.4g}\n"
        f"skew: {out['skew']:.4g}\n"
        f"kurtosis: {out['kurtosis']:.4g}\n"
        f"count: {out['count']}"
    )
    return stats_str


def plot_numeric_distributions(
    df: DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    figsize_per_cell: Tuple[int, int] = (6, 4),
    stat_func: Callable[[Series], str] = calc_stats,
    kde: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Plot histograms (with optional KDE) for numeric columns with stats text.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and array of axes.
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
        sns.histplot(series, kde=kde, ax=ax)
        try:
            stat_text = stat_func(series)
        except Exception:
            stat_text = ""
        ax.set_xlabel(col, fontsize=12)
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


def plot_correlation_matrix(
    df: DataFrame,
    columns: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "mako",
    annot: bool = True,
    fmt: str = ".2f",
    square: bool = True,
    cbar_label: str = "Correlation",
    linewidths: float = 0.5,
    linecolor: str = "white",
    annot_kws: Optional[dict] = None,
    xtick_rotation: int = 45,
    xtick_ha: str = "right",
    xtick_fontsize: int = 12,
    ytick_rotation: int = 0,
    ytick_fontsize: int = 12,
    title: Optional[str] = "Correlation Matrix of Numeric Columns",
    save_path: Optional[Path] = None,
):
    """Compute and plot a correlation matrix for numeric columns in `df`."""
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(columns) == 0:
        raise ValueError("No numeric columns to compute correlation for.")

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        ax=ax,
        square=square,
        cbar_kws={"label": cbar_label},
        linewidths=linewidths,
        linecolor=linecolor,
        annot_kws=(annot_kws or {"size": 12}),
    )

    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel("Features", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xtick_rotation,
        ha=xtick_ha,
        fontsize=xtick_fontsize,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, fontsize=ytick_fontsize)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)

    return fig, ax


def plot_property_distributions(
    props_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 4,
    figsize_per_cell: tuple = (4, 3),
    bins: int = 30,
    kde: bool = True,
    title: str = "Molecular Property Distributions",
    save_path: str | None = None,
):
    """Plot distributions of molecular properties in a grid.

    Expected columns include: MW, TPSA, HBA, HBD, RotBonds, LogP,
    NumHeavyAtoms, NumRings. Columns not present are skipped.
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

    for i, col in enumerate(cols):
        ax = axes[i]
        series = props_df[col].dropna()
        sns.histplot(series, bins=bins, kde=kde, ax=ax, color="C0")
        stat_text = calc_stats(series)
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


__all__ = [
    "calc_stats",
    "plot_numeric_distributions",
    "plot_correlation_matrix",
    "plot_property_distributions",
]
