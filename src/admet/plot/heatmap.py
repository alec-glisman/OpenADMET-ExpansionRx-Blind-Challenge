from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from admet.plot.latex import latex_sanitize

LOGGER = logging.getLogger(__name__)


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
) -> Tuple[Figure, Axes]:
    """Compute and plot a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    columns : sequence of str, optional
        Explicit numeric columns; auto-detected if ``None``.
    figsize : tuple[int, int], optional
        Figure size.
    cmap : str, optional
        Colormap (default "mako").
    annot : bool, optional
        Annotate cells with numeric values (default True).
    fmt : str, optional
        Format for annotations.
    square : bool, optional
        Whether to enforce square cells.
    cbar_label : str, optional
        Colorbar label text.
    linewidths : float, optional
        Grid line width.
    linecolor : str, optional
        Grid line color.
    annot_kws : dict, optional
        Additional keyword args for annotation styling.
    xtick_rotation, ytick_rotation : int, optional
        Tick label rotations.
    title : str, optional
        Super-title text.
    save_path : pathlib.Path, optional
        Path to save figure.

    Returns
    -------
    (Figure, Axes)
        Figure and axes objects.
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(columns) == 0:
        raise ValueError("No numeric columns to compute correlation for.")

    # sanitize column names for LaTeX rendering
    df = df.copy()
    df.columns = [latex_sanitize(col) for col in df.columns]
    columns = [latex_sanitize(col) for col in columns]

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
