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

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback

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
        batch: Tuple[Any, ...],
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
        model.eval()  # Use eval mode for consistent forward passes

        # Identify shared parameters
        shared_params: Dict[str, nn.Parameter] = {}
        for name, param in model.named_parameters():
            if _is_shared_param(
                name,
                self.config.shared_param_patterns,
                self.config.exclude_param_patterns,
            ):
                shared_params[name] = param

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

            for name, param in shared_params.items():
                if param.grad is not None:
                    task_i_grads[name] = param.grad.clone()
                else:
                    task_i_grads[name] = torch.zeros_like(param)

            # Apply lookahead: θ^{t+1}_{s|i} = θ^t_s - η * ∇_{θ_s} L_i
            # We temporarily modify the parameters, compute losses, then restore
            original_params: Dict[str, torch.Tensor] = {}
            try:
                for name, param in shared_params.items():
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
                for name, param in shared_params.items():
                    param.data = original_params.get(name, param.data)

        # Update running statistics
        self.affinity_sum += Z_t
        self.step_count += 1
        self.epoch_affinity_sum += Z_t
        self.epoch_step_count += 1

        return Z_t


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
        config: InterTaskAffinityConfig,
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

            # Optionally save heatmap and clustermap plots
            if self.config.save_plots:
                try:
                    import tempfile

                    import matplotlib.pyplot as plt

                    from admet.model.chemprop.task_affinity import (
                        plot_task_affinity_clustermap,
                        plot_task_affinity_heatmap,
                    )

                    # Heatmap
                    with tempfile.NamedTemporaryFile(mode="wb", suffix="_affinity_heatmap.png", delete=False) as hf:
                        fig_hm = plot_task_affinity_heatmap(Z_final, self.target_cols, figsize=(8, 6))
                        fig_hm.savefig(hf.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                        mlflow.log_artifact(hf.name, "inter_task_affinity")
                        plt.close(fig_hm)

                    # Clustermap
                    with tempfile.NamedTemporaryFile(mode="wb", suffix="_affinity_clustermap.png", delete=False) as cf:
                        fig_cm = plot_task_affinity_clustermap(Z_final, self.target_cols, groups=None, figsize=(8, 8))
                        fig_cm.savefig(cf.name, dpi=self.config.plot_dpi, bbox_inches="tight")
                        mlflow.log_artifact(cf.name, "inter_task_affinity")
                        plt.close(fig_cm)
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
