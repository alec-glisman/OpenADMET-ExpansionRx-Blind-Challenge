"""
Visualization Subpackage
========================

Plotting and visualization utilities.

.. module:: admet.visualize

"""

import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

GLASBEY_PALETTE = list(cc.glasbey)

try:
    import scienceplots  # noqa: F401 - style registration
except Exception:  # noqa: BLE001 - style import fallback
    scienceplots = None  # no-op if style not installed

# Use Agg backend for non-interactive environments (CI, headless servers)
matplotlib.use("Agg")
try:
    plt.style.use("science")
except Exception:  # noqa: BLE001 - fallback to seaborn when style not installed
    # Fallback: use seaborn theme if scienceplots not available
    sns.set_theme(style="whitegrid", palette=GLASBEY_PALETTE)
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:  # noqa: BLE001 - seaborn style fallback
        plt.style.use("default")

plt.rcParams["axes.prop_cycle"] = cycler(color=GLASBEY_PALETTE)
sns.set_palette(GLASBEY_PALETTE)

# Public API exports
from admet.plot.density import plot_endpoint_distributions, plot_property_distributions  # noqa: E402
from admet.plot.latex import latex_sanitize, text_correlation, text_distribution  # noqa: E402
from admet.plot.leaderboard import (  # noqa: E402
    generate_all_plots,
    plot_delta_mae_comparison,
    plot_mae_comparison_bar,
    plot_metrics_heatmap,
    plot_overall_rank_distribution,
    plot_performance_category_pie,
    plot_rank_vs_metric_scatter,
    plot_task_rankings_bar,
    save_figure_formats,
)
from admet.plot.metrics import (  # noqa: E402
    compute_metrics_by_split,
    compute_metrics_df,
    metrics_to_latex_table,
    plot_all_metrics,
    plot_all_metrics_by_split,
    plot_metric_bar,
    plot_metrics_grouped_by_split,
)
from admet.plot.parity import plot_parity, plot_parity_by_split, plot_parity_grid, save_parity_plots  # noqa: E402

__all__ = [
    # Constants
    "GLASBEY_PALETTE",
    # Density plots
    "plot_endpoint_distributions",
    "plot_property_distributions",
    # LaTeX helpers
    "latex_sanitize",
    "text_correlation",
    "text_distribution",
    # Parity plots
    "plot_parity",
    "plot_parity_by_split",
    "plot_parity_grid",
    "save_parity_plots",
    # Metric bar charts
    "compute_metrics_df",
    "compute_metrics_by_split",
    "plot_metric_bar",
    "plot_metrics_grouped_by_split",
    "plot_all_metrics",
    "plot_all_metrics_by_split",
    "metrics_to_latex_table",
    # Leaderboard plots
    "generate_all_plots",
    "plot_delta_mae_comparison",
    "plot_mae_comparison_bar",
    "plot_metrics_heatmap",
    "plot_overall_rank_distribution",
    "plot_performance_category_pie",
    "plot_rank_vs_metric_scatter",
    "plot_task_rankings_bar",
    "save_figure_formats",
]
