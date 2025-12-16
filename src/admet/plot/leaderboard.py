"""Leaderboard visualization functions.

This module provides publication-quality plotting functions for leaderboard analysis,
including rankings, performance comparisons, and diagnostic visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from admet.leaderboard.parser import extract_value_uncertainty

logger = logging.getLogger(__name__)

# Performance zone thresholds and colors
ZONE_THRESHOLDS = [10, 20, 40, 60]
ZONE_COLORS = ["#06D6A0", "#118AB2", "#FFD166", "#EF476F", "#E63946"]
ZONE_LABELS = ["Excellent (1-10)", "Good (11-20)", "Okay (21-40)", "Poor (41-60)", "Needs Improvement (60+)"]


def _get_rank_color(rank: int) -> str:
    """Get color for a rank based on performance zones.

    Parameters
    ----------
    rank : int
        Rank position

    Returns
    -------
    str
        Hex color code
    """
    for threshold, color in zip(ZONE_THRESHOLDS, ZONE_COLORS):
        if rank <= threshold:
            return color
    return ZONE_COLORS[-1]


def save_figure_formats(fig: Figure, output_path: Path, formats: List[str] = ["png", "svg", "pdf"]) -> None:
    """Save figure in multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    output_path : Path
        Base output path (without extension)
    formats : List[str], default=["png", "svg", "pdf"]
        File formats to save
    """
    for fmt in formats:
        fmt_dir = output_path.parent / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)
        out_file = fmt_dir / f"{output_path.name}.{fmt}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.debug("Saved figure to: %s", out_file)
    plt.close(fig)


def plot_overall_rank_distribution(
    df_overall: pd.DataFrame,
    user_rank: Optional[int] = None,
    *,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[Figure, np.ndarray]:
    """Plot overall ranking distribution with histogram and ECDF.

    Parameters
    ----------
    df_overall : pd.DataFrame
        Overall leaderboard DataFrame with 'rank' column
    user_rank : Optional[int]
        User's rank to highlight
    figsize : Tuple[float, float], default=(14, 6)
        Figure size

    Returns
    -------
    Tuple[Figure, np.ndarray]
        Figure and axes array
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Extract ranks
    rank_col = next((c for c in df_overall.columns if str(c).strip().lower() == "rank"), None)
    if rank_col is None:
        logger.warning("No 'rank' column found in overall DataFrame")
        return fig, axs

    ranks_all = pd.to_numeric(df_overall[rank_col], errors="coerce").dropna().astype(int).values

    # Histogram
    sns.histplot(ranks_all, bins=30, kde=False, ax=axs[0], color="#118AB2")
    if user_rank is not None:
        axs[0].axvline(user_rank, color="#E63946", linewidth=2, linestyle="--", label=f"Your Rank: {user_rank}")
    axs[0].set_xlabel("Rank Position", fontsize=12)
    axs[0].set_ylabel("Count", fontsize=12)
    axs[0].set_title("Overall Rank Distribution", fontsize=14, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    if user_rank is not None:
        axs[0].legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    # ECDF
    x_sorted = np.sort(ranks_all)
    y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    axs[1].plot(x_sorted, y_ecdf, color="#06D6A0", linewidth=2)
    if user_rank is not None:
        axs[1].axvline(user_rank, color="#E63946", linewidth=2, linestyle=":", label=f"Your Rank: {user_rank}")
    axs[1].set_xlabel("Rank Position", fontsize=12)
    axs[1].set_ylabel("ECDF", fontsize=12)
    axs[1].set_title("Overall Rank ECDF", fontsize=14, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    if user_rank is not None:
        axs[1].legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig, axs


def plot_task_rankings_bar(
    task_data: pd.DataFrame,
    target_user: str,
    *,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[Figure, Axes]:
    """Plot task-specific rankings as horizontal bar chart with performance zones.

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data with 'task' and 'rank' columns
    target_user : str
        Username for title
    figsize : Tuple[float, float], default=(12, 8)
        Figure size

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes
    """
    task_data_sorted = task_data.sort_values("rank")
    fig, ax = plt.subplots(figsize=figsize)

    colors = [_get_rank_color(int(r)) for r in task_data_sorted["rank"]]
    ax.barh(
        task_data_sorted["task"],
        task_data_sorted["rank"],
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_xlabel("Rank Position (lower is better)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(f"Task-Specific Rankings for {target_user}", fontsize=14, fontweight="bold")
    ax.invert_xaxis()

    # Add performance zone shading
    for i, (low, high, color, label) in enumerate(
        zip([0] + ZONE_THRESHOLDS, ZONE_THRESHOLDS + [100], ZONE_COLORS, ZONE_LABELS)
    ):
        ax.axvspan(low, high, alpha=0.1, color=color, label=label, zorder=1)

    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower left", fontsize=9, frameon=True, fancybox=True, shadow=True)

    # Add rank labels
    for i, (task, rank) in enumerate(zip(task_data_sorted["task"], task_data_sorted["rank"])):
        ax.annotate(
            f"{int(rank)}",
            (rank, i),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=4,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )

    plt.tight_layout()
    return fig, ax


def plot_delta_mae_comparison(
    task_data: pd.DataFrame,
    task_mins: Dict[str, float],
    target_user: str,
    *,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[Figure, Axes]:
    """Plot performance gap analysis (delta MAE to minimum).

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data with 'task' and 'mae' columns
    task_mins : Dict[str, float]
        Minimum MAE values per task
    target_user : str
        Username for title
    figsize : Tuple[float, float], default=(12, 8)
        Figure size

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes
    """
    # Calculate delta percentages
    task_data_with_delta = task_data[task_data["task"].isin(task_mins.keys())].copy()
    deltas = []

    for _, row in task_data_with_delta.iterrows():
        task = row["task"]
        mae_str = str(row.get("mae", "N/A"))
        val, _ = extract_value_uncertainty(mae_str)

        if val is not None:
            min_mae = task_mins.get(task, val)
            delta_pct = ((val - min_mae) / val) * 100 if val > 0 else 0
            deltas.append(delta_pct)
        else:
            deltas.append(0.0)

    task_data_with_delta["delta_mae_pct"] = deltas
    task_data_with_delta = task_data_with_delta.sort_values("delta_mae_pct")

    fig, ax = plt.subplots(figsize=figsize)

    colors_delta = [
        "#06D6A0" if d < 10 else "#FFD166" if d < 20 else "#EF476F" for d in task_data_with_delta["delta_mae_pct"]
    ]

    ax.barh(
        task_data_with_delta["task"],
        task_data_with_delta["delta_mae_pct"],
        color=colors_delta,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    ax.set_xlabel(r"$\Delta$ MAE to Minimum (\%)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(f"Performance Gap Analysis: {target_user} vs. Top Performer", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add percentage labels
    for i, (task, delta) in enumerate(zip(task_data_with_delta["task"], task_data_with_delta["delta_mae_pct"])):
        ax.annotate(
            f"{delta:.1f}%",
            (delta, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
        )

    plt.tight_layout()
    return fig, ax


def plot_mae_comparison_bar(
    task_data: pd.DataFrame,
    task_mins: Dict[str, float],
    target_user: str,
    *,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[Figure, Axes]:
    """Plot MAE values comparison with minimum (top performer).

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data with 'task' and 'mae' columns
    task_mins : Dict[str, float]
        Minimum MAE values per task
    target_user : str
        Username for title
    figsize : Tuple[float, float], default=(12, 8)
        Figure size

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes
    """
    mae_values = []
    mae_errors = []
    min_values = []
    valid_tasks = []

    for _, row in task_data.iterrows():
        task = row["task"]
        mae_str = str(row.get("mae", "N/A"))

        if mae_str != "N/A" and task in task_mins:
            val, err = extract_value_uncertainty(mae_str)
            if val is not None:
                mae_values.append(val)
                mae_errors.append(err if err is not None else 0.0)
                min_values.append(task_mins[task])
                valid_tasks.append(task)

    if not valid_tasks:
        logger.warning("No valid tasks for MAE comparison")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    x = np.arange(len(valid_tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, mae_values, width, yerr=mae_errors, label=target_user, color="#118AB2", alpha=0.8, capsize=4)
    ax.bar(x + width / 2, min_values, width, label="Top Performer", color="#06D6A0", alpha=0.8)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title(f"MAE Comparison: {target_user} vs. Top Performer", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_tasks, rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, ax


def plot_performance_category_pie(
    task_data: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (10, 7),
) -> Tuple[Figure, Axes]:
    """Plot performance distribution pie chart.

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data with 'rank' column
    figsize : Tuple[float, float], default=(10, 7)
        Figure size

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes
    """
    ranks = task_data["rank"].values
    categories = []

    for rank in ranks:
        if rank <= 10:
            categories.append("Excellent")
        elif rank <= 20:
            categories.append("Good")
        elif rank <= 40:
            categories.append("Okay")
        elif rank <= 60:
            categories.append("Poor")
        else:
            categories.append("Needs Improvement")

    category_counts = pd.Series(categories).value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    colors_pie = []
    for cat in category_counts.index:
        if cat == "Excellent":
            colors_pie.append(ZONE_COLORS[0])
        elif cat == "Good":
            colors_pie.append(ZONE_COLORS[1])
        elif cat == "Okay":
            colors_pie.append(ZONE_COLORS[2])
        elif cat == "Poor":
            colors_pie.append(ZONE_COLORS[3])
        else:
            colors_pie.append(ZONE_COLORS[4])

    pie_res = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05] * len(category_counts),
    )

    # Support different matplotlib returns across versions
    if len(pie_res) == 3:
        wedges, texts, autotexts = pie_res
    else:
        wedges, texts = pie_res
        autotexts = []

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title("Performance Category Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, ax


def plot_metrics_heatmap(
    task_data: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[Figure, np.ndarray]:
    """Plot multi-metric heatmaps for each task.

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data with metric columns
    figsize : Tuple[float, float], default=(14, 10)
        Figure size

    Returns
    -------
    Tuple[Figure, np.ndarray]
        Figure and axes array
    """
    metrics = ["r2", "spearman r", "kendall's tau", "mae"]
    metric_labels = ["$R^2$", "Spearman $R$", "Kendall's $\\tau$", "MAE"]

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        tasks = []

        for _, row in task_data.iterrows():
            task = row["task"]
            val_str = str(row.get(metric, "N/A"))

            if val_str != "N/A":
                val, _ = extract_value_uncertainty(val_str)
                if val is not None:
                    values.append(val)
                    tasks.append(task)

        if values:
            data = pd.DataFrame({"Task": tasks, label: values})
            pivot = data.pivot_table(values=label, index="Task", aggfunc="first")

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn" if metric != "mae" else "RdYlGn_r",
                cbar=True,
                ax=axs[idx],
                linewidths=0.5,
            )
            axs[idx].set_title(f"{label} by Task", fontsize=12, fontweight="bold")
            axs[idx].set_xlabel("")
            axs[idx].set_ylabel("")

    plt.tight_layout()
    return fig, axs


def plot_rank_vs_metric_scatter(
    task_data: pd.DataFrame,
    metric: str,
    metric_label: str,
    *,
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[Figure, Axes]:
    """Plot rank vs metric scatter plot.

    Parameters
    ----------
    task_data : pd.DataFrame
        Task data
    metric : str
        Metric column name
    metric_label : str
        Metric label for axis
    figsize : Tuple[float, float], default=(10, 6)
        Figure size

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes
    """
    ranks = []
    values = []
    colors = []

    for _, row in task_data.iterrows():
        rank = row["rank"]
        val_str = str(row.get(metric, "N/A"))

        if val_str != "N/A" and pd.notna(rank):
            val, _ = extract_value_uncertainty(val_str)
            if val is not None:
                ranks.append(rank)
                values.append(val)
                colors.append(_get_rank_color(int(rank)))

    fig, ax = plt.subplots(figsize=figsize)

    if ranks:
        ax.scatter(ranks, values, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=1)
        ax.set_xlabel("Rank Position", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"Rank vs {metric_label}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    return fig, ax


def generate_all_plots(
    results_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    task_mins: Dict[str, float],
    overall_min: Optional[float],
    output_dir: Path,
    target_user: str,
) -> None:
    """Generate all leaderboard plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        Summary results DataFrame
    tables : Dict[str, pd.DataFrame]
        Raw leaderboard tables
    task_mins : Dict[str, float]
        Minimum MAE per task
    overall_min : Optional[float]
        Minimum overall MA-RAE
    output_dir : Path
        Output directory for plots
    target_user : str
        Target username
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_data = results_df[results_df["task"] != "OVERALL"].copy()

    # 1. Overall rank distribution
    if "Average" in tables:
        overall_row = results_df[results_df["task"] == "OVERALL"]
        user_rank = int(overall_row.iloc[0]["rank"]) if not overall_row.empty else None
        fig, _ = plot_overall_rank_distribution(tables["Average"], user_rank)
        save_figure_formats(fig, output_dir / "01_overall_rank_hist_ecdf")

    # 2. Task rankings bar
    fig, _ = plot_task_rankings_bar(task_data, target_user)
    save_figure_formats(fig, output_dir / "02_task_rankings_bar")

    # 3. Delta MAE comparison
    fig, _ = plot_delta_mae_comparison(task_data, task_mins, target_user)
    save_figure_formats(fig, output_dir / "03_delta_mae_comparison")

    # 4. MAE comparison bar
    fig, _ = plot_mae_comparison_bar(task_data, task_mins, target_user)
    save_figure_formats(fig, output_dir / "04_mae_comparison_bar")

    # 5. Performance category pie
    fig, _ = plot_performance_category_pie(task_data)
    save_figure_formats(fig, output_dir / "05_performance_category_pie")

    # 6. Metrics heatmap
    fig, _ = plot_metrics_heatmap(task_data)
    save_figure_formats(fig, output_dir / "06_metrics_heatmap_multi")

    # 7-10. Rank vs metrics scatter plots
    for idx, (metric, label) in enumerate(
        [("r2", "R²"), ("mae", "MAE"), ("spearman r", "Spearman R"), ("kendall's tau", "Kendall's τ")], start=7
    ):
        fig, _ = plot_rank_vs_metric_scatter(task_data, metric, label)
        metric_filename = metric.replace(" ", "_").replace("'", "")
        save_figure_formats(fig, output_dir / f"{idx:02d}_rank_vs_{metric_filename}")

    logger.info("Generated all plots in: %s", output_dir)
