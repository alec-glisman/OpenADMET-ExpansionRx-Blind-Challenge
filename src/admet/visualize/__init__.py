"""
Visualization Subpackage
========================

Plotting and visualization utilities.

.. module:: admet.visualize

"""

from .model_performance import plot_parity_grid, plot_metric_bars, to_linear

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import scienceplots  # noqa: F401 - style registration
except Exception:
    scienceplots = None  # no-op if style not installed

# Use Agg backend for non-interactive environments (CI, headless servers)
matplotlib.use("Agg")
try:
    plt.style.use("science")
except Exception:
    # Fallback: use seaborn theme if scienceplots not available
    sns.set_theme(style="whitegrid")
    plt.style.use("seaborn")


__all__ = ["plot_parity_grid", "plot_metric_bars", "to_linear"]
