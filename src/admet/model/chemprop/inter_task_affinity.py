"""
Inter-Task Affinity Computation for Multi-Task Learning
========================================================

This module implements the inter-task affinity computation from
"Efficiently Identifying Task Groupings for Multi-Task Learning"
(Fifty et al., NeurIPS 2021, https://arxiv.org/abs/2109.04617).

The key insight from the paper is to measure inter-task affinity by computing
how a gradient update from one task affects the loss of other tasks. This is
done via a "lookahead" approach:

    Z^t_{ij} = 1 - L_j(X^t, θ^{t+1}_{s|i}, θ^t_j) / L_j(X^t, θ^t_s, θ^t_j)

Where:
- θ^t_s: Shared parameters at time t
- θ^{t+1}_{s|i} = θ^t_s - η∇_{θ_s} L_i: Updated shared params after task i's gradient
- L_j: Loss for task j
- X^t: Input batch at time t

A positive Z^t_{ij} indicates task i's update helps task j (positive transfer).
A negative Z^t_{ij} indicates task i's update hurts task j (negative transfer).

The training-level affinity is computed as:
    Ẑ_{ij} = (1/T) Σ_t Z^t_{ij}

Key Components
--------------
- :class:`InterTaskAffinityConfig`: Configuration for affinity computation
- :class:`InterTaskAffinityCallback`: Lightning callback for per-step computation
- :class:`InterTaskAffinityComputer`: Core computation logic

References
----------
Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., & Finn, C. (2021).
Efficiently Identifying Task Groupings for Multi-Task Learning.
Advances in Neural Information Processing Systems, 34.
https://arxiv.org/abs/2109.04617
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

# Module logger
logger = logging.getLogger("admet.model.chemprop.inter_task_affinity")


@dataclass
class InterTaskAffinityConfig:
    """
    Configuration for inter-task affinity computation during training.

    This configuration controls the lookahead-based inter-task affinity
    computation as described in the TAG paper. The affinity matrix is
    computed during training (not as a separate pre-training phase) and
    logged to MLflow.

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable inter-task affinity computation during training.
    compute_every_n_steps : int, default=1
        Compute affinity every N training steps. Higher values reduce
        computational overhead but provide less granular measurements.
        Set to 1 for full per-step computation as in the paper.
    log_every_n_steps : int, default=100
        Log running average affinity to MLflow every N steps.
        Individual step affinities can be very noisy, so we typically
        log aggregated values.
    log_epoch_summary : bool, default=True
        Log epoch-level summary statistics (mean, std) of affinity matrix.
    log_step_matrices : bool, default=False
        Log individual step affinity matrices. WARNING: This can generate
        a very large number of metrics. Only enable for debugging.
    lookahead_lr : float, default=0.001
        Learning rate η for computing the lookahead parameter update.
        This should typically match or be close to the training learning rate.
        If None, uses the current optimizer learning rate.
    use_optimizer_lr : bool, default=True
        If True, uses the current optimizer learning rate for lookahead.
        This overrides lookahead_lr during training.
    shared_param_patterns : List[str], default=[]
        Patterns to identify shared encoder parameters. Parameters matching
        these patterns are considered "shared" for affinity computation.
        If empty, uses default exclusion patterns (predictor, ffn, head).
    exclude_param_patterns : List[str], default=["predictor", "ffn", "output", "head"]
        Patterns to exclude from shared parameters. These are task-specific
        parameters that should not be included in the affinity computation.
    n_groups : Optional[int], default=None
        If provided, cluster tasks into this many groups using the final
        affinity matrix (TAG grouping step).
    clustering_method : str, default="agglomerative"
        Clustering algorithm for grouping: "agglomerative" or "spectral".
    clustering_linkage : str, default="average"
        Linkage criterion for agglomerative clustering.
    device : str, default="auto"
        Device for computation: "auto", "cpu", or "cuda".
    log_to_mlflow : bool, default=True
        Whether to log affinity metrics to MLflow.
    """

    enabled: bool = False
    compute_every_n_steps: int = 1
    log_every_n_steps: int = 100
    log_epoch_summary: bool = True
    log_step_matrices: bool = False
    lookahead_lr: float = 0.001
    use_optimizer_lr: bool = True
    shared_param_patterns: List[str] = field(default_factory=list)
    exclude_param_patterns: List[str] = field(default_factory=lambda: ["predictor", "ffn", "output", "head", "readout"])
    n_groups: Optional[int] = None
    clustering_method: str = "agglomerative"
    clustering_linkage: str = "average"
    device: str = "auto"
    log_to_mlflow: bool = True
    save_plots: bool = False
    plot_formats: List[str] = field(default_factory=lambda: ["png"])
    plot_dpi: int = 150


def _get_device(device_str: str) -> torch.device:
    """
    Resolve device string to torch.device.

    Parameters
    ----------
    device_str : str
        Device string: "auto", "cpu", or "cuda".

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _is_shared_param(
    name: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> bool:
    """
    Determine if a parameter belongs to the shared encoder.

    Parameters
    ----------
    name : str
        Parameter name.
    include_patterns : List[str]
        Patterns that indicate a shared parameter. If provided and non-empty,
        parameter must match at least one pattern.
    exclude_patterns : List[str]
        Patterns to exclude (task-specific layers).

    Returns
    -------
    bool
        True if the parameter is a shared encoder parameter.
    """
    name_lower = name.lower()

    # If include patterns provided, parameter must match one
    if include_patterns:
        if not any(p.lower() in name_lower for p in include_patterns):
            return False

    # Exclude task-specific parameters
    if any(p.lower() in name_lower for p in exclude_patterns):
        return False

    return True


def _masked_task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_idx: int,
) -> Optional[torch.Tensor]:
    """
    Compute MSE loss for a single task over non-NaN entries.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions tensor of shape (batch, n_tasks).
    target : torch.Tensor
        Target tensor of shape (batch, n_tasks), may contain NaN.
    task_idx : int
        Index of the task to compute loss for.

    Returns
    -------
    Optional[torch.Tensor]
        MSE loss scalar, or None if no valid entries for this task.
    """
    task_pred = pred[:, task_idx]
    task_target = target[:, task_idx]
    mask = ~torch.isnan(task_target)

    if mask.sum() == 0:
        return None

    diff = task_pred[mask] - task_target[mask]
    return (diff**2).mean()


def _order_tasks_by_groups(task_names: List[str], groups: Optional[List[List[str]]]) -> List[int]:
    """Return task indices ordered by provided groups (or identity if None)."""
    if not groups:
        return list(range(len(task_names)))
    name_to_idx = {name: i for i, name in enumerate(task_names)}
    ordered: List[int] = []
    for group in groups:
        for name in group:
            if name in name_to_idx:
                ordered.append(name_to_idx[name])
    # Add any missing tasks in original order
    missing = [i for i in range(len(task_names)) if i not in ordered]
    ordered.extend(missing)
    return ordered


def _cluster_tasks_from_affinity(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    n_groups: Optional[int],
    method: str = "agglomerative",
    linkage: str = "average",
) -> Tuple[Optional[List[List[str]]], Optional[np.ndarray]]:
    """
    Cluster tasks into groups from an affinity matrix using TAG grouping logic.

    Returns a list of task-name groups and the numeric labels array.
    """
    if n_groups is None or n_groups <= 0:
        return None, None

    n_tasks = len(task_names)
    if n_tasks == 0:
        return None, None

    if n_tasks <= n_groups:
        labels = np.arange(n_tasks)
        groups = [[task] for task in task_names]
        return groups, labels

    max_abs = float(np.max(np.abs(affinity_matrix))) if affinity_matrix.size else 1.0
    norm_aff = affinity_matrix / (max_abs + 1e-12)
    distance_matrix = 1.0 - norm_aff
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    if method == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            metric="precomputed",
            linkage=linkage,
        )
        labels = clustering.fit_predict(distance_matrix)
    elif method == "spectral":
        affinity_shifted = norm_aff - norm_aff.min() + 1e-6
        clustering = SpectralClustering(
            n_clusters=n_groups,
            affinity="precomputed",
            random_state=42,
        )
        labels = clustering.fit_predict(affinity_shifted)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    groups_dict: Dict[int, List[str]] = {}
    for task_name, label in zip(task_names, labels):
        groups_dict.setdefault(int(label), []).append(task_name)

    groups = [groups_dict[k] for k in sorted(groups_dict.keys())]
    return groups, labels


def _plot_affinity_heatmap(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    groups: Optional[List[List[str]]],
    dpi: int,
    title: str,
):
    """Create a simple heatmap (ordered by groups if provided)."""
    import matplotlib.pyplot as plt

    order = _order_tasks_by_groups(task_names, groups)
    ordered_matrix = affinity_matrix[np.ix_(order, order)]
    ordered_names = [task_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    im = ax.imshow(ordered_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=45, ha="right")
    ax.set_yticklabels(ordered_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _plot_affinity_clustermap(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    groups: Optional[List[List[str]]],
    dpi: int,
    title: str,
):
    """Plot affinity heatmap with group separators for visual grouping."""
    import matplotlib.pyplot as plt

    order = _order_tasks_by_groups(task_names, groups)
    ordered_matrix = affinity_matrix[np.ix_(order, order)]
    ordered_names = [task_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    im = ax.imshow(ordered_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90)
    ax.set_yticklabels(ordered_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if groups:
        idx = 0
        for group in groups:
            idx += len(group)
            ax.axhline(idx - 0.5, color="gray", linestyle="--", linewidth=0.7)
            ax.axvline(idx - 0.5, color="gray", linestyle="--", linewidth=0.7)

    return fig


def _plot_affinity_asymmetry(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    dpi: int,
) -> plt.Figure:
    """
    Plot asymmetry heatmap showing |Z_ij - Z_ji| for all task pairs.

    This visualization helps identify which task pairs have asymmetric
    transfer relationships (one task helps another more than vice versa).

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Affinity matrix of shape (n_tasks, n_tasks).
    task_names : List[str]
        Task names.
    dpi : int
        Resolution for the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    n_tasks = len(task_names)
    asymmetry = np.zeros((n_tasks, n_tasks))

    for i in range(n_tasks):
        for j in range(n_tasks):
            asymmetry[i, j] = abs(affinity_matrix[i, j] - affinity_matrix[j, i])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    im = ax.imshow(asymmetry, cmap="YlOrRd", vmin=0, vmax=asymmetry.max())
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels(task_names)
    ax.set_title("Task Affinity Asymmetry |Z_ij - Z_ji|")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Asymmetry")

    # Add text annotations for high asymmetry pairs
    for i in range(n_tasks):
        for j in range(n_tasks):
            if asymmetry[i, j] > 0.3:  # Highlight significant asymmetries
                ax.text(j, i, f"{asymmetry[i, j]:.2f}",
                       ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    return fig


def _plot_affinity_network(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    groups: Optional[List[List[str]]],
    threshold: float = 0.2,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot network graph showing strong affinity relationships between tasks.

    Nodes represent tasks, edges represent strong affinities (> threshold).
    Edge color indicates affinity strength (green=positive, red=negative).
    Node color indicates group membership if groups are provided.

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Affinity matrix of shape (n_tasks, n_tasks).
    task_names : List[str]
        Task names.
    groups : Optional[List[List[str]]]
        Task groups for coloring nodes.
    threshold : float
        Minimum absolute affinity to draw an edge.
    dpi : int
        Resolution for the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_tasks = len(task_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # Layout tasks in a circle
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Determine node colors by group
    if groups:
        task_to_group = {}
        for g_idx, group in enumerate(groups):
            for task in group:
                task_to_group[task] = g_idx

        # Use a colormap with distinct colors
        cmap = plt.cm.get_cmap("tab10")
        node_colors = [cmap(task_to_group.get(task, 0) / max(1, len(groups))) for task in task_names]
    else:
        node_colors = ["skyblue"] * n_tasks

    # Draw edges for strong affinities
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            # Use symmetrized affinity for undirected edges
            avg_affinity = (affinity_matrix[i, j] + affinity_matrix[j, i]) / 2

            if abs(avg_affinity) > threshold:
                # Edge color and width based on affinity
                if avg_affinity > 0:
                    color = plt.cm.Greens(min(1.0, avg_affinity / 0.5))
                    linestyle = "-"
                else:
                    color = plt.cm.Reds(min(1.0, abs(avg_affinity) / 0.5))
                    linestyle = "--"

                width = 1 + 3 * abs(avg_affinity)

                ax.plot([x[i], x[j]], [y[i], y[j]],
                       color=color, linewidth=width, linestyle=linestyle,
                       alpha=0.6, zorder=1)

    # Draw nodes
    ax.scatter(x, y, c=node_colors, s=500, zorder=2, edgecolors="black", linewidths=1.5)

    # Add labels
    for i, task in enumerate(task_names):
        # Shorten long task names for readability
        label = task if len(task) <= 20 else task[:17] + "..."
        # Position labels slightly outside the circle
        label_x = x[i] * 1.15
        label_y = y[i] * 1.15
        ax.text(label_x, label_y, label, ha="center", va="center",
               fontsize=9, weight="bold", zorder=3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Task Affinity Network (|affinity| > {threshold})\n"
                f"Green=Positive, Red=Negative, Width=Strength",
                fontsize=12, weight="bold")

    # Add legend for groups if available
    if groups and len(groups) > 1:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap(g_idx / len(groups)), edgecolor="black",
                 label=f"Group {g_idx}: {', '.join(group[:2])}" + ("..." if len(group) > 2 else ""))
            for g_idx, group in enumerate(groups)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
                 bbox_to_anchor=(0, 1), framealpha=0.9)

    plt.tight_layout()
    return fig


class InterTaskAffinityComputer:
    """
    Core computation logic for inter-task affinity.

    This class implements the lookahead-based affinity computation from the
    TAG paper. It computes Z^t_{ij} for each pair of tasks at a given step.

    The computation follows these steps:
    1. For each task i, compute the gradient ∇_{θ_s} L_i
    2. Apply the gradient to get θ^{t+1}_{s|i} = θ^t_s - η * ∇_{θ_s} L_i
    3. For each task j, compute:
       - L_j_before = L_j(X^t, θ^t_s, θ^t_j)
       - L_j_after = L_j(X^t, θ^{t+1}_{s|i}, θ^t_j)
       - Z^t_{ij} = 1 - L_j_after / L_j_before

    Parameters
    ----------
    config : InterTaskAffinityConfig
        Configuration for affinity computation.
    target_cols : List[str]
        Names of target columns (tasks).

    Attributes
    ----------
    config : InterTaskAffinityConfig
        The configuration object.
    device : torch.device
        The computation device.
    target_cols : List[str]
        Task names.
    n_tasks : int
        Number of tasks.
    step_count : int
        Number of steps computed.
    affinity_sum : np.ndarray
        Running sum of affinity matrices.
    epoch_affinity_sum : np.ndarray
        Sum of affinity matrices for current epoch.
    epoch_step_count : int
        Steps computed in current epoch.
    """

    def __init__(
        self,
        config: InterTaskAffinityConfig,
        target_cols: List[str],
    ) -> None:
        """
        Initialize the InterTaskAffinityComputer.

        Parameters
        ----------
        config : InterTaskAffinityConfig
            Configuration for affinity computation.
        target_cols : List[str]
            Names of target columns (tasks).
        """
        self.config = config
        self.device = _get_device(config.device)
        self.target_cols = target_cols
        self.n_tasks = len(target_cols)

        # Running statistics
        self.step_count = 0
        self.affinity_sum = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float64)

        # Epoch statistics
        self.epoch_affinity_sum = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float64)
        self.epoch_step_count = 0
        self._shared_param_cache: Dict[int, Dict[str, nn.Parameter]] = {}
        self._param_audit_cache: Dict[int, Dict[str, List[str]]] = {}
        self._last_param_audit: Dict[str, List[str]] = {}

    def reset_epoch_stats(self) -> None:
        """Reset epoch-level statistics."""
        self.epoch_affinity_sum = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float64)
        self.epoch_step_count = 0

    def get_running_average(self) -> np.ndarray:
        """
        Get the running average affinity matrix over all steps.

        Returns
        -------
        np.ndarray
            Running average affinity matrix Ẑ_{ij}.
        """
        if self.step_count == 0:
            return np.zeros((self.n_tasks, self.n_tasks))
        return self.affinity_sum / self.step_count

    def get_epoch_average(self) -> np.ndarray:
        """
        Get the average affinity matrix for the current epoch.

        Returns
        -------
        np.ndarray
            Epoch average affinity matrix.
        """
        if self.epoch_step_count == 0:
            return np.zeros((self.n_tasks, self.n_tasks))
        return self.epoch_affinity_sum / self.epoch_step_count

    def compute_step_affinity(
        self,
        model: nn.Module,
        batch: Any,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Compute inter-task affinity matrix for a single training step.

        This implements the core lookahead computation from the paper:
        Z^t_{ij} = 1 - L_j(θ^{t+1}_{s|i}) / L_j(θ^t_s)

        Parameters
        ----------
        model : nn.Module
            The neural network model (must have forward method).
        batch : Tuple[Any, ...]
            Training batch (bmg, v_d, e_d, targets, ...).
        learning_rate : float
            Learning rate η for lookahead computation.

        Returns
        -------
        np.ndarray
            Step affinity matrix Z^t of shape (n_tasks, n_tasks).
        """
        # Unpack batch - use named attributes from TrainingBatch
        # BatchMolGraph.to mutates in-place and returns None, so don't reassign
        batch.bmg.to(self.device)
        bmg = batch.bmg
        targets = batch.Y.to(self.device).float()

        model = model.to(self.device)
        prev_training_mode = model.training
        model.eval()  # Use eval mode for consistent forward passes

        # Identify shared parameters (cached for runtime efficiency)
        shared_params, audit = self._get_shared_parameters(model)
        self._last_param_audit = audit

        if len(shared_params) == 0:
            logger.warning(
                "No shared parameters found for affinity computation. "
                "Check shared_param_patterns and exclude_param_patterns."
            )
            return np.zeros((self.n_tasks, self.n_tasks))

        # Step 1: Compute baseline losses L_j(θ^t_s, θ^t_j) for all tasks j
        with torch.no_grad():
            preds_baseline = model(bmg)
            baseline_losses: List[Optional[float]] = []
            for j in range(self.n_tasks):
                loss_j = _masked_task_loss(preds_baseline, targets, j)
                if loss_j is not None:
                    baseline_losses.append(float(loss_j.item()))
                else:
                    baseline_losses.append(None)

        # Initialize affinity matrix for this step
        Z_t = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float64)

        # Step 2: For each task i, compute lookahead and measure effect on all j
        shared_items = list(shared_params.items())
        for i in range(self.n_tasks):
            # Skip if task i has no valid samples in this batch
            task_i_mask = ~torch.isnan(targets[:, i])
            if task_i_mask.sum() == 0:
                continue

            # Compute gradient of L_i with respect to shared parameters
            model.zero_grad()
            model.train()  # Need gradients
            preds_for_grad = model(bmg)
            loss_i = _masked_task_loss(preds_for_grad, targets, i)

            if loss_i is None:
                continue

            # Compute gradients for shared parameters
            task_i_grads: Dict[str, torch.Tensor] = {}
            loss_i.backward(retain_graph=False)

            for name, param in shared_items:
                if param.grad is not None:
                    task_i_grads[name] = param.grad.clone()
                else:
                    task_i_grads[name] = torch.zeros_like(param)

            # Apply lookahead: θ^{t+1}_{s|i} = θ^t_s - η * ∇_{θ_s} L_i
            # We temporarily modify the parameters, compute losses, then restore
            original_params: Dict[str, torch.Tensor] = {}
            try:
                for name, param in shared_items:
                    original_params[name] = param.data.clone()
                    param.data = param.data - learning_rate * task_i_grads[name]

                # Compute L_j(θ^{t+1}_{s|i}) for all tasks j
                model.eval()
                with torch.no_grad():
                    preds_lookahead = model(bmg)
                    for j in range(self.n_tasks):
                        loss_j_before_val = baseline_losses[j]
                        if loss_j_before_val is None:
                            continue

                        loss_j_after = _masked_task_loss(preds_lookahead, targets, j)
                        if loss_j_after is None:
                            continue

                        loss_j_after_val = float(loss_j_after.item())

                        # Avoid division by zero
                        if abs(loss_j_before_val) < 1e-10:
                            Z_t[i, j] = 0.0
                        else:
                            # Z^t_{ij} = 1 - L_j_after / L_j_before
                            Z_t[i, j] = 1.0 - (loss_j_after_val / loss_j_before_val)
            finally:
                # Restore original parameters no matter what
                for name, param in shared_items:
                    if name in original_params:
                        param.data = original_params[name]

        # Update running statistics
        self.affinity_sum += Z_t
        self.step_count += 1
        self.epoch_affinity_sum += Z_t
        self.epoch_step_count += 1

        if prev_training_mode:
            model.train()
        else:
            model.eval()

        return Z_t

    def _get_shared_parameters(
        self,
        model: nn.Module,
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, List[str]]]:
        """Return shared parameter dict and audit info for a model (cached)."""
        model_id = id(model)
        if model_id in self._shared_param_cache:
            return self._shared_param_cache[model_id], self._param_audit_cache[model_id]

        shared: Dict[str, nn.Parameter] = {}
        shared_names: List[str] = []
        non_shared_names: List[str] = []

        for name, param in model.named_parameters():
            if _is_shared_param(name, self.config.shared_param_patterns, self.config.exclude_param_patterns):
                shared[name] = param
                shared_names.append(name)
            else:
                non_shared_names.append(name)

        audit = {"shared": shared_names, "non_shared": non_shared_names}
        self._shared_param_cache[model_id] = shared
        self._param_audit_cache[model_id] = audit
        return shared, audit

    def get_param_audit(self) -> Dict[str, List[str]]:
        """Return the latest classification of shared vs non-shared params."""
        return self._last_param_audit


class InterTaskAffinityCallback(Callback):
    """
    PyTorch Lightning callback for computing inter-task affinity during training.

    This callback integrates the lookahead-based inter-task affinity computation
    from the TAG paper into the training loop. It computes affinity at each
    training step (or every N steps) and logs the results to MLflow.

    The callback tracks:
    - Per-step affinity matrices Z^t_{ij} (optionally)
    - Running average affinity Ẑ_{ij} across all steps
    - Per-epoch summary statistics

    Parameters
    ----------
    config : InterTaskAffinityConfig
        Configuration for affinity computation.
    target_cols : List[str]
        Names of target columns (tasks).

    Attributes
    ----------
    config : InterTaskAffinityConfig
        The configuration object.
    computer : InterTaskAffinityComputer
        The affinity computation engine.
    global_step : int
        Current global training step.

    Examples
    --------
    >>> config = InterTaskAffinityConfig(enabled=True, log_every_n_steps=50)
    >>> callback = InterTaskAffinityCallback(config, target_cols=["LogD", "KSOL"])
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        config: Any,
        target_cols: List[str],
    ) -> None:
        """
        Initialize the InterTaskAffinityCallback.

        Parameters
        ----------
        config : InterTaskAffinityConfig
            Configuration for affinity computation.
        target_cols : List[str]
            Names of target columns (tasks).
        """
        super().__init__()
        self.config = config
        self.target_cols = target_cols
        self.computer = InterTaskAffinityComputer(config, target_cols)
        self.global_step = 0
        self._current_batch: Optional[Tuple[Any, ...]] = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Compute inter-task affinity after each training batch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        outputs : Any
            Outputs from the training step.
        batch : Any
            The current training batch.
        batch_idx : int
            Index of the current batch.
        """
        if not self.config.enabled:
            return

        self.global_step += 1

        # Skip if not computing this step
        if self.global_step % self.config.compute_every_n_steps != 0:
            return

        # Get learning rate for lookahead
        if self.config.use_optimizer_lr and trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0]["lr"]
            except (IndexError, KeyError):
                lr = self.config.lookahead_lr
        else:
            lr = self.config.lookahead_lr

        # Compute step affinity
        try:
            Z_t = self.computer.compute_step_affinity(
                model=pl_module,
                batch=batch,
                learning_rate=lr,
            )
        except Exception as e:
            logger.warning("Failed to compute step affinity: %s", e)
            return

        # Log step matrix if enabled
        if self.config.log_step_matrices and self.config.log_to_mlflow:
            self._log_step_matrix(Z_t)

        # Log running average periodically
        if self.global_step % self.config.log_every_n_steps == 0 and self.config.log_to_mlflow:
            self._log_running_average()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Log epoch summary and reset epoch statistics.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not self.config.enabled:
            return

        if self.config.log_epoch_summary and self.config.log_to_mlflow:
            self._log_epoch_summary(trainer.current_epoch)

        # Reset epoch statistics
        self.computer.reset_epoch_stats()

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Log final affinity matrix at end of training.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not self.config.enabled:
            return

        if self.config.log_to_mlflow:
            self._log_final_matrix()

    def _log_step_matrix(self, Z_t: np.ndarray) -> None:
        """Log individual step affinity matrix to MLflow."""
        try:
            for i, task_i in enumerate(self.target_cols):
                for j, task_j in enumerate(self.target_cols):
                    metric_name = f"affinity/step/Z_{_sanitize(task_i)}_{_sanitize(task_j)}"
                    mlflow.log_metric(metric_name, float(Z_t[i, j]), step=self.global_step)
        except Exception as e:
            logger.debug("Failed to log step matrix: %s", e)

    def _log_running_average(self) -> None:
        """Log running average affinity matrix to MLflow."""
        try:
            Z_avg = self.computer.get_running_average()
            for i, task_i in enumerate(self.target_cols):
                for j, task_j in enumerate(self.target_cols):
                    metric_name = f"affinity/running/Z_{_sanitize(task_i)}_{_sanitize(task_j)}"
                    mlflow.log_metric(metric_name, float(Z_avg[i, j]), step=self.global_step)

            # Also log summary statistics
            mlflow.log_metric("affinity/running/mean", float(np.mean(Z_avg)), step=self.global_step)
            mlflow.log_metric("affinity/running/std", float(np.std(Z_avg)), step=self.global_step)

            # Log off-diagonal mean (excludes self-affinity)
            mask = ~np.eye(Z_avg.shape[0], dtype=bool)
            if mask.sum() > 0:
                off_diag_mean = float(np.mean(Z_avg[mask]))
                mlflow.log_metric("affinity/running/off_diag_mean", off_diag_mean, step=self.global_step)

        except Exception as e:
            logger.debug("Failed to log running average: %s", e)

    def _log_epoch_summary(self, epoch: int) -> None:
        """Log epoch summary statistics to MLflow."""
        try:
            Z_epoch = self.computer.get_epoch_average()

            # Log epoch-level metrics
            mlflow.log_metric("affinity/epoch/mean", float(np.mean(Z_epoch)), step=epoch)
            mlflow.log_metric("affinity/epoch/std", float(np.std(Z_epoch)), step=epoch)
            mlflow.log_metric("affinity/epoch/steps", float(self.computer.epoch_step_count), step=epoch)

            # Log per-task-pair epoch averages
            for i, task_i in enumerate(self.target_cols):
                for j, task_j in enumerate(self.target_cols):
                    metric_name = f"affinity/epoch/Z_{_sanitize(task_i)}_{_sanitize(task_j)}"
                    mlflow.log_metric(metric_name, float(Z_epoch[i, j]), step=epoch)

        except Exception as e:
            logger.debug("Failed to log epoch summary: %s", e)

    def _log_final_matrix(self) -> None:
        """Log final affinity matrix as artifact."""
        try:
            Z_final = self.computer.get_running_average()
            df = pd.DataFrame(
                Z_final,
                index=self.target_cols,
                columns=self.target_cols,
            )

            # Log as CSV artifact
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix="_affinity_matrix.csv", delete=False) as f:
                df.to_csv(f.name)
                mlflow.log_artifact(f.name, "inter_task_affinity")

            # Compute and log summary statistics for inter-group affinity
            self._log_affinity_summary_statistics(Z_final)
            # Log parameter classification (shared vs non-shared)
            param_audit = self.computer.get_param_audit()
            if param_audit:
                try:
                    mlflow.log_param("affinity/shared_param_count", str(len(param_audit.get("shared", []))))
                    mlflow.log_param("affinity/non_shared_param_count", str(len(param_audit.get("non_shared", []))))
                except Exception:
                    logger.debug("Failed to log parameter counts")
                with tempfile.NamedTemporaryFile(mode="w", suffix="_param_audit.json", delete=False) as pf:
                    json.dump(param_audit, pf, indent=2)
                    mlflow.log_artifact(pf.name, "inter_task_affinity")

            # Cluster tasks using TAG affinity (if requested)
            groups: Optional[List[List[str]]] = None
            labels: Optional[np.ndarray] = None
            if self.config.n_groups is not None:
                try:
                    groups, labels = _cluster_tasks_from_affinity(
                        Z_final,
                        self.target_cols,
                        self.config.n_groups,
                        method=self.config.clustering_method,
                        linkage=self.config.clustering_linkage,
                    )
                    if groups:
                        # Log group assignments as parameters for quick visibility
                        for group_idx, group in enumerate(groups):
                            for task in group:
                                mlflow.log_param(f"affinity/group/{_sanitize(task)}", str(group_idx))

                        # Save grouping artifact
                        with tempfile.NamedTemporaryFile(mode="w", suffix="_task_groups.json", delete=False) as gf:
                            json.dump(
                                {
                                    "task_groups": groups,
                                    "labels": labels.tolist() if labels is not None else None,
                                    "n_groups": len(groups),
                                    "tasks": self.target_cols,
                                },
                                gf,
                                indent=2,
                            )
                            mlflow.log_artifact(gf.name, "inter_task_affinity")

                        # Log detailed group affinity analysis
                        self._log_group_affinity_analysis(Z_final, groups, labels)

                except Exception as e:
                    logger.warning("Failed to cluster tasks from affinity matrix: %s", e)

            # Optionally save heatmap and clustermap plots
            if self.config.save_plots:
                try:
                    for fmt in self.config.plot_formats:
                        fmt = fmt.lstrip(".")
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=f"_affinity_heatmap.{fmt}", delete=False
                        ) as hf:
                            fig_hm = _plot_affinity_heatmap(
                                Z_final,
                                self.target_cols,
                                groups,
                                dpi=self.config.plot_dpi,
                                title="Inter-Task Affinity (TAG)",
                            )
                            fig_hm.savefig(hf.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                            mlflow.log_artifact(hf.name, "inter_task_affinity")
                            plt.close(fig_hm)

                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=f"_affinity_clustermap.{fmt}", delete=False
                        ) as cf:
                            fig_cm = _plot_affinity_clustermap(
                                Z_final,
                                self.target_cols,
                                groups,
                                dpi=self.config.plot_dpi,
                                title="Inter-Task Affinity (TAG) - Grouped",
                            )
                            fig_cm.savefig(cf.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                            mlflow.log_artifact(cf.name, "inter_task_affinity")
                            plt.close(fig_cm)

                        # Additional visualization: asymmetry heatmap
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=f"_affinity_asymmetry.{fmt}", delete=False
                        ) as asym_f:
                            fig_asym = _plot_affinity_asymmetry(
                                Z_final,
                                self.target_cols,
                                dpi=self.config.plot_dpi,
                            )
                            fig_asym.savefig(asym_f.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                            mlflow.log_artifact(asym_f.name, "inter_task_affinity")
                            plt.close(fig_asym)

                        # Network graph visualization of strong affinities
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=f"_affinity_network.{fmt}", delete=False
                        ) as net_f:
                            fig_net = _plot_affinity_network(
                                Z_final,
                                self.target_cols,
                                groups,
                                threshold=0.2,
                                dpi=self.config.plot_dpi,
                            )
                            fig_net.savefig(net_f.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                            mlflow.log_artifact(net_f.name, "inter_task_affinity")
                            plt.close(fig_net)

                except Exception as e:
                    logger.debug("Failed to create/save plot artifacts: %s", e)

            # Log final metrics
            mlflow.log_metric("affinity/final/mean", float(np.mean(Z_final)))
            mlflow.log_metric("affinity/final/std", float(np.std(Z_final)))
            mlflow.log_metric("affinity/final/total_steps", float(self.computer.step_count))

            # Log final matrix values
            for i, task_i in enumerate(self.target_cols):
                for j, task_j in enumerate(self.target_cols):
                    metric_name = f"affinity/final/Z_{_sanitize(task_i)}_{_sanitize(task_j)}"
                    mlflow.log_metric(metric_name, float(Z_final[i, j]))

            logger.info(
                "Final inter-task affinity matrix logged to MLflow " "(mean=%.4f, std=%.4f, steps=%d)",
                np.mean(Z_final),
                np.std(Z_final),
                self.computer.step_count,
            )

        except Exception as e:
            logger.warning("Failed to log final affinity matrix: %s", e)

    def _log_affinity_summary_statistics(self, Z_final: np.ndarray) -> None:
        """
        Log comprehensive summary statistics for task affinity analysis.

        This method computes and logs key metrics to help understand task relationships:
        1. Positive/negative transfer percentages
        2. Strong affinity pairs (both positive and negative)
        3. Per-task incoming/outgoing affinity averages
        4. Symmetry analysis
        5. Group-level affinity summaries if clustering is enabled

        Parameters
        ----------
        Z_final : np.ndarray
            Final affinity matrix of shape (n_tasks, n_tasks).
        """
        try:
            n_tasks = len(self.target_cols)

            # 1. Positive vs Negative Transfer Statistics
            mask = ~np.eye(n_tasks, dtype=bool)  # Exclude diagonal
            off_diag_values = Z_final[mask]

            positive_count = np.sum(off_diag_values > 0)
            negative_count = np.sum(off_diag_values < 0)
            near_zero_count = np.sum(np.abs(off_diag_values) < 0.05)

            total_pairs = len(off_diag_values)
            pct_positive = 100.0 * positive_count / total_pairs
            pct_negative = 100.0 * negative_count / total_pairs
            pct_neutral = 100.0 * near_zero_count / total_pairs

            mlflow.log_metric("affinity/summary/pct_positive_transfer", pct_positive)
            mlflow.log_metric("affinity/summary/pct_negative_transfer", pct_negative)
            mlflow.log_metric("affinity/summary/pct_neutral_transfer", pct_neutral)
            mlflow.log_metric("affinity/summary/mean_off_diagonal", float(np.mean(off_diag_values)))
            mlflow.log_metric("affinity/summary/std_off_diagonal", float(np.std(off_diag_values)))
            mlflow.log_metric("affinity/summary/max_affinity", float(np.max(off_diag_values)))
            mlflow.log_metric("affinity/summary/min_affinity", float(np.min(off_diag_values)))

            logger.info(
                "Task Affinity Summary: %.1f%% positive, %.1f%% negative, %.1f%% neutral (|Z|<0.05)",
                pct_positive, pct_negative, pct_neutral
            )

            # 2. Identify Strong Affinity Pairs
            # Find top-5 positive and top-5 negative task pairs
            strong_pairs_dict = {}

            # Get upper triangle indices (avoid duplicates since matrix may be asymmetric)
            for threshold, pair_type in [(0.3, "strong_positive"), (-0.3, "strong_negative")]:
                pairs = []
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i != j:
                            val = Z_final[i, j]
                            if (pair_type == "strong_positive" and val > threshold) or \
                               (pair_type == "strong_negative" and val < threshold):
                                pairs.append((self.target_cols[i], self.target_cols[j], val))

                # Sort by absolute value
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                strong_pairs_dict[pair_type] = pairs[:5]

            # Log strong pairs as JSON artifact
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix="_strong_pairs.json", delete=False) as spf:
                json.dump(strong_pairs_dict, spf, indent=2)
                mlflow.log_artifact(spf.name, "inter_task_affinity")

            # 3. Per-Task Affinity Summaries
            # For each task, compute average incoming and outgoing affinity
            task_summaries = {}
            for i, task in enumerate(self.target_cols):
                # Incoming: how much other tasks help this task (column j, average over i)
                incoming_affinities = [Z_final[k, i] for k in range(n_tasks) if k != i]
                avg_incoming = float(np.mean(incoming_affinities))

                # Outgoing: how much this task helps others (row i, average over j)
                outgoing_affinities = [Z_final[i, k] for k in range(n_tasks) if k != i]
                avg_outgoing = float(np.mean(outgoing_affinities))

                task_summaries[task] = {
                    "avg_incoming_affinity": avg_incoming,
                    "avg_outgoing_affinity": avg_outgoing,
                    "net_contribution": avg_outgoing,  # Positive = helps others
                    "receives_benefit": avg_incoming,   # Positive = helped by others
                }

                # Log per-task metrics
                mlflow.log_metric(f"affinity/per_task/{_sanitize(task)}/avg_incoming", avg_incoming)
                mlflow.log_metric(f"affinity/per_task/{_sanitize(task)}/avg_outgoing", avg_outgoing)
                mlflow.log_metric(f"affinity/per_task/{_sanitize(task)}/net_contribution", avg_outgoing)

            # Save task summaries as JSON
            with tempfile.NamedTemporaryFile(mode="w", suffix="_task_summaries.json", delete=False) as tsf:
                json.dump(task_summaries, tsf, indent=2)
                mlflow.log_artifact(tsf.name, "inter_task_affinity")

            # 4. Symmetry Analysis
            # Check how symmetric the affinity matrix is (Z_ij vs Z_ji)
            symmetry_diffs = []
            for i in range(n_tasks):
                for j in range(i+1, n_tasks):
                    diff = abs(Z_final[i, j] - Z_final[j, i])
                    symmetry_diffs.append(diff)

            avg_asymmetry = float(np.mean(symmetry_diffs))
            max_asymmetry = float(np.max(symmetry_diffs))

            mlflow.log_metric("affinity/summary/avg_asymmetry", avg_asymmetry)
            mlflow.log_metric("affinity/summary/max_asymmetry", max_asymmetry)

            logger.info(
                "Affinity matrix asymmetry: mean=%.4f, max=%.4f (lower is more symmetric)",
                avg_asymmetry, max_asymmetry
            )

            # 5. Log readable summary report
            summary_report = self._generate_affinity_report(
                Z_final, task_summaries, strong_pairs_dict,
                pct_positive, pct_negative, avg_asymmetry
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix="_affinity_report.txt", delete=False) as rf:
                rf.write(summary_report)
                mlflow.log_artifact(rf.name, "inter_task_affinity")

            logger.info("Affinity summary statistics logged to MLflow")

        except Exception as e:
            logger.warning("Failed to log affinity summary statistics: %s", e)

    def _generate_affinity_report(
        self,
        Z_final: np.ndarray,
        task_summaries: Dict[str, Dict[str, float]],
        strong_pairs: Dict[str, List[Tuple[str, str, float]]],
        pct_positive: float,
        pct_negative: float,
        avg_asymmetry: float,
    ) -> str:
        """Generate a human-readable affinity analysis report."""
        lines = [
            "=" * 80,
            "TASK AFFINITY ANALYSIS REPORT",
            "=" * 80,
            "",
            "Overall Statistics:",
            f"  - Positive Transfer: {pct_positive:.1f}% of task pairs",
            f"  - Negative Transfer: {pct_negative:.1f}% of task pairs",
            f"  - Avg Asymmetry: {avg_asymmetry:.4f}",
            "",
            "=" * 80,
            "TOP POSITIVE AFFINITY PAIRS (Strong Synergies):",
            "=" * 80,
        ]

        for task_i, task_j, affinity in strong_pairs.get("strong_positive", []):
            lines.append(f"  {task_i:40s} → {task_j:40s} : {affinity:+.4f}")

        lines.extend([
            "",
            "=" * 80,
            "TOP NEGATIVE AFFINITY PAIRS (Potential Interference):",
            "=" * 80,
        ])

        for task_i, task_j, affinity in strong_pairs.get("strong_negative", []):
            lines.append(f"  {task_i:40s} → {task_j:40s} : {affinity:+.4f}")

        lines.extend([
            "",
            "=" * 80,
            "PER-TASK SUMMARY (Incoming/Outgoing Affinity):",
            "=" * 80,
            f"{'Task':<45s} {'Incoming':>12s} {'Outgoing':>12s} {'Interpretation':<30s}",
            "-" * 80,
        ])

        # Sort tasks by net contribution
        sorted_tasks = sorted(
            task_summaries.items(),
            key=lambda x: x[1]["net_contribution"],
            reverse=True
        )

        for task, stats in sorted_tasks:
            incoming = stats["avg_incoming_affinity"]
            outgoing = stats["avg_outgoing_affinity"]

            # Interpretation
            if outgoing > 0.2 and incoming > 0.2:
                interp = "Synergistic (helps & helped)"
            elif outgoing > 0.2:
                interp = "Contributor (helps others)"
            elif incoming > 0.2:
                interp = "Beneficiary (helped by others)"
            elif outgoing < -0.1:
                interp = "Interfering (hurts others)"
            else:
                interp = "Independent"

            lines.append(f"{task:<45s} {incoming:>12.4f} {outgoing:>12.4f} {interp:<30s}")

        lines.extend([
            "",
            "=" * 80,
            "RECOMMENDATIONS FOR TASK GROUPING:",
            "=" * 80,
        ])

        # Provide recommendations based on affinity patterns
        high_synergy_count = sum(1 for _, stats in task_summaries.items()
                                 if stats["avg_incoming_affinity"] > 0.2 and stats["avg_outgoing_affinity"] > 0.2)

        if pct_positive > 70:
            lines.append("  ✓ High overall positive transfer (>70%) suggests tasks benefit from joint training")
            lines.append("  ✓ Consider using 1-2 groups to maximize knowledge sharing")
        elif pct_negative > 30:
            lines.append("  ⚠ Significant negative transfer (>30%) detected")
            lines.append("  ⚠ Consider using 3+ groups to isolate conflicting tasks")
        else:
            lines.append("  • Mixed transfer patterns - use affinity matrix to guide grouping")
            lines.append(f"  • Detected {high_synergy_count}/{len(task_summaries)} synergistic tasks")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _log_group_affinity_analysis(
        self,
        Z_final: np.ndarray,
        groups: List[List[str]],
        labels: np.ndarray,
    ) -> None:
        """
        Compute and log inter-group and intra-group affinity statistics.

        This helps answer: "Should I split my 9 tasks into separate model subgroups?"

        Parameters
        ----------
        Z_final : np.ndarray
            Final affinity matrix.
        groups : List[List[str]]
            Task groups from clustering.
        labels : np.ndarray
            Cluster labels for each task.
        """
        try:
            n_groups = len(groups)
            task_to_idx = {task: i for i, task in enumerate(self.target_cols)}

            # Compute intra-group and inter-group affinities
            intra_group_affinities = []
            inter_group_affinities = []

            group_stats = {}

            for g_idx, group in enumerate(groups):
                # Intra-group affinity (within this group)
                group_indices = [task_to_idx[task] for task in group]
                intra_values = []

                for i in group_indices:
                    for j in group_indices:
                        if i != j:
                            intra_values.append(Z_final[i, j])

                if intra_values:
                    avg_intra = float(np.mean(intra_values))
                    intra_group_affinities.extend(intra_values)
                else:
                    avg_intra = 0.0

                group_stats[g_idx] = {
                    "group_id": g_idx,
                    "tasks": group,
                    "size": len(group),
                    "avg_intra_affinity": avg_intra,
                }

            # Inter-group affinity (between different groups)
            for g1_idx in range(n_groups):
                for g2_idx in range(g1_idx + 1, n_groups):
                    group1_indices = [task_to_idx[task] for task in groups[g1_idx]]
                    group2_indices = [task_to_idx[task] for task in groups[g2_idx]]

                    inter_values = []
                    for i in group1_indices:
                        for j in group2_indices:
                            inter_values.append(Z_final[i, j])
                            inter_values.append(Z_final[j, i])  # Both directions

                    if inter_values:
                        inter_group_affinities.extend(inter_values)

            # Compute summary statistics
            avg_intra = float(np.mean(intra_group_affinities)) if intra_group_affinities else 0.0
            avg_inter = float(np.mean(inter_group_affinities)) if inter_group_affinities else 0.0

            # Key metric: intra-group affinity should be higher than inter-group
            separation_quality = avg_intra - avg_inter

            mlflow.log_metric("affinity/groups/avg_intra_group_affinity", avg_intra)
            mlflow.log_metric("affinity/groups/avg_inter_group_affinity", avg_inter)
            mlflow.log_metric("affinity/groups/separation_quality", separation_quality)
            mlflow.log_metric("affinity/groups/num_groups", float(n_groups))

            # Log per-group statistics
            for g_idx, stats in group_stats.items():
                mlflow.log_metric(f"affinity/groups/group_{g_idx}/size", float(stats["size"]))
                mlflow.log_metric(f"affinity/groups/group_{g_idx}/avg_intra_affinity", stats["avg_intra_affinity"])

            # Save group analysis report
            import json
            import tempfile

            group_analysis = {
                "num_groups": n_groups,
                "avg_intra_group_affinity": avg_intra,
                "avg_inter_group_affinity": avg_inter,
                "separation_quality": separation_quality,
                "groups": group_stats,
                "recommendation": self._get_grouping_recommendation(avg_intra, avg_inter, separation_quality),
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix="_group_analysis.json", delete=False) as gaf:
                json.dump(group_analysis, gaf, indent=2)
                mlflow.log_artifact(gaf.name, "inter_task_affinity")

            logger.info(
                "Group Affinity: Intra=%.4f, Inter=%.4f, Separation=%.4f (higher is better)",
                avg_intra, avg_inter, separation_quality
            )

        except Exception as e:
            logger.warning("Failed to log group affinity analysis: %s", e)

    def _get_grouping_recommendation(
        self,
        avg_intra: float,
        avg_inter: float,
        separation: float,
    ) -> str:
        """Generate recommendation for task grouping based on affinity analysis."""
        if separation > 0.3:
            return (
                "STRONG SEPARATION: Groups are well-separated with high intra-group affinity. "
                "This grouping is RECOMMENDED - train separate models for each group."
            )
        elif separation > 0.1:
            return (
                "MODERATE SEPARATION: Groups show some separation. "
                "Consider training separate models, but also test joint training with task weighting."
            )
        elif separation > -0.1:
            return (
                "WEAK SEPARATION: Groups have similar intra/inter-group affinity. "
                "Grouping may not provide significant benefit - consider joint training with 1-2 groups."
            )
        else:
            return (
                "POOR SEPARATION: Inter-group affinity is higher than intra-group affinity. "
                "This suggests the clustering is not optimal - try different n_groups or joint training."
            )

    def get_affinity_matrix(self) -> np.ndarray:
        """
        Get the current running average affinity matrix.

        Returns
        -------
        np.ndarray
            Running average affinity matrix Ẑ_{ij}.
        """
        return self.computer.get_running_average()

    def get_affinity_dataframe(self) -> pd.DataFrame:
        """
        Get the affinity matrix as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Affinity matrix with task names as index and columns.
        """
        return pd.DataFrame(
            self.computer.get_running_average(),
            index=self.target_cols,
            columns=self.target_cols,
        )


def _sanitize(name: str) -> str:
    """
    Sanitize a name for use in MLflow metric names.

    MLflow metric names can only contain alphanumeric characters, underscores,
    dashes, periods, spaces, and slashes.

    Parameters
    ----------
    name : str
        The name to sanitize.

    Returns
    -------
    str
        The sanitized name.
    """
    # Replace problematic characters
    replacements = {
        ">": "",
        "<": "_",
        ":": "_",
        ";": "_",
        "|": "_",
        "\\": "_",
        "?": "",
        "*": "",
        '"': "",
        "'": "",
        "[": "",
        "]": "",
        "(": "",
        ")": "",
        ",": "_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result
