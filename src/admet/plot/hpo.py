"""HPO analysis plotting utilities.

Provides publication-quality visualizations for hyperparameter optimization:
- Parameter importance bar charts
- Learning curve convergence plots
- Pareto frontier analysis for multi-objective optimization
- Parameter correlation heatmaps

This module integrates with Optuna studies and Ray Tune results to provide
comprehensive post-HPO analysis visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Use consistent color palette from existing plotting modules
GLASBEY_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC78BC",
    "#CA9161",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def plot_param_importance(
    study: "optuna.Study",
    output_dir: Path,
    top_n: int = 15,
    format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, Axes] | tuple[None, None]:
    """Plot parameter importance from Optuna study.

    Uses Optuna's built-in importance evaluator to compute and visualize
    which hyperparameters had the most impact on the optimization objective.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study with trials
    output_dir : Path
        Directory to save plot
    top_n : int, default=15
        Number of top parameters to show
    format : str, default="png"
        Output format (png, pdf, svg)
    dpi : int, default=300
        Resolution for raster formats

    Returns
    -------
    tuple[Figure, Axes] | tuple[None, None]
        Matplotlib figure and axes objects, or (None, None) if plotting failed

    Examples
    --------
    >>> import optuna
    >>> study = optuna.load_study(study_name="my_study", storage="sqlite:///db.sqlite")
    >>> fig, ax = plot_param_importance(study, Path("output"))
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, skipping parameter importance plot")
        return None, None

    # Compute parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning(f"Failed to compute parameter importance: {e}")
        return None, None

    if not importance:
        logger.warning("No parameter importance data available")
        return None, None

    # Sort by importance and take top N
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    params, scores = zip(*sorted_importance)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)))

    y_pos = np.arange(len(params))
    ax.barh(y_pos, scores, color=GLASBEY_PALETTE[0], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Top {len(params)} Hyperparameter Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f"optuna_param_importance.{format}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved parameter importance plot: {output_path}")

    return fig, ax


def plot_learning_curves(
    study: "optuna.Study",
    output_dir: Path,
    metric_name: str = "value",
    format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, Axes] | tuple[None, None]:
    """Plot HPO learning curves (best-so-far convergence).

    Visualizes the optimization progress by plotting all trial values and
    the best value achieved so far at each trial number.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    output_dir : Path
        Directory to save plot
    metric_name : str, default="value"
        Name of metric to plot (displayed on y-axis label)
    format : str, default="png"
        Output format
    dpi : int, default=300
        Resolution for raster formats

    Returns
    -------
    tuple[Figure, Axes] | tuple[None, None]
        Matplotlib figure and axes, or (None, None) if plotting failed

    Examples
    --------
    >>> fig, ax = plot_learning_curves(study, Path("output"), metric_name="val_mae")
    """
    if not OPTUNA_AVAILABLE:
        return None, None

    trials_df = study.trials_dataframe()
    if trials_df.empty:
        logger.warning("No trials data available for learning curves")
        return None, None

    # Get trial values (filter out failed trials)
    valid_trials = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if valid_trials.empty:
        logger.warning("No completed trials for learning curves")
        return None, None

    trial_numbers = valid_trials["number"].values
    trial_values = valid_trials["value"].values

    # Compute best-so-far curve
    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        best_so_far = np.minimum.accumulate(trial_values)
    else:
        best_so_far = np.maximum.accumulate(trial_values)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual trial values (scatter)
    ax.scatter(
        trial_numbers,
        trial_values,
        alpha=0.4,
        s=30,
        color=GLASBEY_PALETTE[1],
        label="Trial value",
        edgecolors="white",
        linewidths=0.5,
    )

    # Plot best-so-far curve (line)
    ax.plot(
        trial_numbers,
        best_so_far,
        linewidth=2.5,
        color=GLASBEY_PALETTE[0],
        label="Best so far",
        marker="o",
        markersize=4,
        markevery=max(1, len(trial_numbers) // 20),
    )

    ax.set_xlabel("Trial Number", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric_name.upper()}", fontsize=12, fontweight="bold")
    ax.set_title("HPO Convergence: Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, shadow=True, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = output_dir / f"optuna_learning_curve.{format}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved learning curve plot: {output_path}")

    return fig, ax


def plot_pareto_frontier(
    study: "optuna.Study",
    output_dir: Path,
    objective1_name: str = "val_mae",
    objective2_name: str = "model_params_millions",
    format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, Axes] | tuple[None, None]:
    """Plot Pareto frontier for multi-objective optimization.

    Identifies and visualizes the Pareto-optimal trials (non-dominated solutions)
    in the objective space. Useful for understanding trade-offs between model
    performance and complexity.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study (can be single or multi-objective)
    output_dir : Path
        Directory to save plot
    objective1_name : str, default="val_mae"
        Name of first objective (primary metric)
    objective2_name : str, default="model_params_millions"
        Name of second objective (model complexity)
    format : str, default="png"
        Output format
    dpi : int, default=300
        Resolution for raster formats

    Returns
    -------
    tuple[Figure, Axes] | tuple[None, None]
        Matplotlib figure and axes, or (None, None) if plotting failed

    Notes
    -----
    This function assumes both objectives should be minimized. The Pareto frontier
    consists of trials where no other trial performs better on both objectives.

    Examples
    --------
    >>> fig, ax = plot_pareto_frontier(study, Path("output"))
    """
    if not OPTUNA_AVAILABLE:
        return None, None

    trials_df = study.trials_dataframe()
    if trials_df.empty:
        logger.warning("No trials data for Pareto frontier")
        return None, None

    # Extract objectives from trials
    obj1_values = []
    obj2_values = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Primary objective (from Optuna value)
        obj1 = trial.value
        if obj1 is None:
            continue

        # Secondary objective (from user attributes)
        # Try different possible attribute names
        obj2 = None
        for attr_name in [objective2_name, "model_params_millions", "params_millions"]:
            obj2 = trial.user_attrs.get(attr_name)
            if obj2 is not None:
                break

        # Also try to get from intermediate values (reported metrics)
        if obj2 is None and trial.intermediate_values:
            # Get the last reported value
            last_step = max(trial.intermediate_values.keys())
            # This won't work for model_params as it's not in intermediate_values
            # Need to extract from trial params or attributes
            pass

        if obj2 is None:
            continue

        obj1_values.append(obj1)
        obj2_values.append(obj2)

    if not obj1_values:
        logger.warning(f"No {objective2_name} data found in trials. Skipping Pareto plot.")
        logger.info("Hint: Ensure model_params_millions is logged as a metric in Ray Tune trials")
        return None, None

    obj1_values = np.array(obj1_values)
    obj2_values = np.array(obj2_values)

    # Identify Pareto frontier (minimize both objectives)
    pareto_mask = np.ones(len(obj1_values), dtype=bool)
    for i in range(len(obj1_values)):
        for j in range(len(obj1_values)):
            if i != j:
                # Check if j dominates i (better on both objectives)
                if (
                    obj1_values[j] <= obj1_values[i]
                    and obj2_values[j] <= obj2_values[i]
                    and (obj1_values[j] < obj1_values[i] or obj2_values[j] < obj2_values[i])
                ):
                    pareto_mask[i] = False
                    break

    pareto_obj1 = obj1_values[pareto_mask]
    pareto_obj2 = obj2_values[pareto_mask]

    # Sort Pareto points by second objective for line plot
    pareto_sort_idx = np.argsort(pareto_obj2)
    pareto_obj1 = pareto_obj1[pareto_sort_idx]
    pareto_obj2 = pareto_obj2[pareto_sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all trials
    ax.scatter(
        obj2_values,
        obj1_values,
        alpha=0.5,
        s=50,
        color=GLASBEY_PALETTE[2],
        label="All trials",
        edgecolors="white",
        linewidths=0.5,
    )

    # Highlight Pareto frontier
    ax.scatter(
        pareto_obj2,
        pareto_obj1,
        s=120,
        color=GLASBEY_PALETTE[0],
        edgecolors="black",
        linewidths=2,
        label="Pareto optimal",
        zorder=5,
        marker="D",
    )

    # Connect Pareto points
    if len(pareto_obj2) > 1:
        ax.plot(
            pareto_obj2,
            pareto_obj1,
            linestyle="--",
            linewidth=2,
            color=GLASBEY_PALETTE[0],
            alpha=0.6,
            zorder=4,
        )

    ax.set_xlabel(f"{objective2_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{objective1_name.upper()}", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Frontier: Model Performance vs Complexity", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, shadow=True, fontsize=11, loc="best")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = output_dir / f"optuna_pareto_frontier.{format}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved Pareto frontier plot: {output_path} ({len(pareto_obj1)} optimal points)")

    return fig, ax


def plot_optimization_history(
    study: "optuna.Study",
    output_dir: Path,
    format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, Axes] | tuple[None, None]:
    """Plot optimization history with all trials over time.

    Similar to learning curves but focuses on temporal progression and
    includes visual indicators for pruned/failed trials.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    output_dir : Path
        Directory to save plot
    format : str, default="png"
        Output format
    dpi : int, default=300
        Resolution for raster formats

    Returns
    -------
    tuple[Figure, Axes] | tuple[None, None]
        Matplotlib figure and axes, or (None, None) if plotting failed
    """
    if not OPTUNA_AVAILABLE:
        return None, None

    try:
        from optuna.visualization.matplotlib import plot_optimization_history as optuna_plot

        fig = optuna_plot(study)
        ax = fig.gca()

        # Customize plot
        ax.set_title("Optimization History", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        output_path = output_dir / f"optuna_optimization_history.{format}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved optimization history plot: {output_path}")

        return fig, ax
    except Exception as e:
        logger.warning(f"Failed to plot optimization history: {e}")
        return None, None
