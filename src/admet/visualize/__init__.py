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

from .model_performance import plot_metric_bars, plot_parity_grid, to_linear

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


__all__ = ["plot_parity_grid", "plot_metric_bars", "to_linear", "GLASBEY_PALETTE"]
