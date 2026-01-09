"""
HPO Final Trial Metrics Computation
====================================

Utility functions for computing comprehensive metrics at HPO trial completion.
Follows the naming convention used in ensemble runs: {split_name}/{target}_{metric}

This module provides a shared implementation for both Chemprop and CheMeleon HPO
to compute all 8 metrics (mae, rae, mape, rmse, R2, pearson_r, spearman_rho, kendall_tau)
at trial completion for train, validation, and test sets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from admet.data.stats import correlation
from admet.plot.metrics import METRIC_NAMES

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def sanitize_metric_label(label: str) -> str:
    """
    Sanitize label for use in MLflow/Ray Tune metric names.

    Converts labels to lowercase and replaces special characters with underscores
    to ensure valid metric names compatible with MLflow and Ray Tune.

    Parameters
    ----------
    label : str
        Raw label (e.g., "Log KSOL", "Spearman rho", "Log Caco-2 Permeability Papp A>B")

    Returns
    -------
    str
        Sanitized label (e.g., "log_ksol", "spearman_rho", "log_caco_2_permeability_papp_agt_b")

    Examples
    --------
    >>> sanitize_metric_label("Log KSOL")
    'log_ksol'
    >>> sanitize_metric_label("Log Caco-2 Permeability Papp A>B")
    'log_caco_2_permeability_papp_agt_b'
    >>> sanitize_metric_label("R2")
    'r2'
    """
    return (
        label.lower()
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("-", "_")
        .replace("$", "")
        .replace("^", "")
        .replace("\\", "")
        .replace("ρ", "rho")
        .replace("τ", "tau")
        .replace("²", "2")
    )


def _compute_split_metrics(
    predictions: np.ndarray | "torch.Tensor",
    targets: np.ndarray | "torch.Tensor",
    target_columns: list[str],
    split_name: str,
) -> dict[str, float]:
    """
    Compute all 8 metrics for a single data split.

    Parameters
    ----------
    predictions : np.ndarray or torch.Tensor
        Model predictions, shape (n_samples, n_targets)
    targets : np.ndarray or torch.Tensor
        Ground truth values, shape (n_samples, n_targets)
    target_columns : list[str]
        Names of target columns (for metric keys)
    split_name : str
        Name of the split (e.g., "train", "val", "test")

    Returns
    -------
    dict[str, float]
        Dictionary mapping "{split_name}/{safe_target}_{metric}" to metric values,
        plus aggregate "{split_name}/mean_{metric}" for each metric type.
    """
    import torch

    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure 2D arrays
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    metrics_dict: dict[str, float] = {}
    metric_aggregates: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

    for i, target in enumerate(target_columns):
        if i >= predictions.shape[1] or i >= targets.shape[1]:
            continue

        y_pred = predictions[:, i]
        y_true = targets[:, i]

        # Compute all metrics using the shared correlation function
        target_metrics = correlation(y_true, y_pred, compute_rank_correlations=True, use_torch=False)

        # Sanitize target name for metric key
        safe_target = sanitize_metric_label(target)

        # Store per-target metrics
        for metric_name in METRIC_NAMES:
            metric_value_any: Any = target_metrics.get(metric_name, float("nan"))
            metric_value = float(metric_value_any)
            key = f"{split_name}/{safe_target}_{metric_name}"
            metrics_dict[key] = metric_value

            # Collect for aggregate
            if not np.isnan(metric_value):
                metric_aggregates[metric_name].append(metric_value)

    # Compute aggregate (mean) metrics across all targets
    for metric_name in METRIC_NAMES:
        values = metric_aggregates[metric_name]
        mean_value = float(np.mean(values)) if values else float("nan")
        metrics_dict[f"{split_name}/mean_{metric_name}"] = mean_value

    return metrics_dict


def compute_final_trial_metrics(
    model: Any,
    train_loader: Any,
    val_loader: Any | None,
    test_loader: Any | None,
    target_columns: list[str],
    device: str | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive metrics for an HPO trial at completion.

    This function evaluates the model on train, validation (if provided),
    and test (if provided) data loaders, computing all 8 standard metrics
    for each target column plus aggregate means.

    Parameters
    ----------
    model : Any
        Trained model with a predict method that accepts a data loader.
        Should return predictions as numpy array or torch Tensor.
    train_loader : Any
        Data loader for training set evaluation.
    val_loader : Any or None
        Data loader for validation set evaluation. If None, skipped.
    test_loader : Any or None
        Data loader for test set evaluation. If None, skipped.
    target_columns : list[str]
        Names of target columns.
    device : str or None
        Device to use for prediction (e.g., "cuda", "cpu"). If None, auto-detect.

    Returns
    -------
    dict[str, float]
        Dictionary with keys following the pattern:
        - "{split}/mean_{metric}" for aggregate metrics
        - "{split}/{safe_target}_{metric}" for per-target metrics

        Where split is one of "train", "val", "test" and metric is one of:
        mae, rae, mape, rmse, R2, pearson_r, spearman_rho, kendall_tau

    Notes
    -----
    This function is designed to be called at HPO trial completion (after training
    finishes or early stopping triggers). The resulting metrics are reported to
    Ray Tune via train.report() and will be logged to MLflow via the callback.

    The naming convention matches the ensemble runs for consistency, allowing
    comparison between HPO trial metrics and ensemble model metrics.

    Examples
    --------
    >>> final_metrics = compute_final_trial_metrics(
    ...     model=trained_model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     test_loader=test_loader,
    ...     target_columns=["LogD", "Log KSOL", "Log HLM CLint"],
    ... )
    >>> train.report(final_metrics)
    """
    all_metrics: dict[str, float] = {}

    # Helper to extract targets from loader
    def get_targets_from_loader(loader: Any) -> np.ndarray:
        """Extract target values from a data loader."""
        import torch

        targets_list = []
        for batch in loader:
            if hasattr(batch, "Y"):
                # Chemprop MoleculeDataLoader
                targets_list.append(batch.Y)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                # Standard (X, y) tuple
                targets_list.append(batch[1])
            else:
                raise ValueError(f"Cannot extract targets from batch type: {type(batch)}")

        # Concatenate all targets
        if targets_list:
            if isinstance(targets_list[0], torch.Tensor):
                return torch.cat(targets_list, dim=0).detach().cpu().numpy()
            return np.concatenate(targets_list, axis=0)
        return np.array([])

    # Evaluate on train set
    logger.debug("Computing final metrics on train set...")
    try:
        train_preds = model.predict(train_loader)
        train_targets = get_targets_from_loader(train_loader)
        train_metrics = _compute_split_metrics(train_preds, train_targets, target_columns, "train")
        all_metrics.update(train_metrics)
        logger.info("Train metrics computed: %d entries", len(train_metrics))
    except Exception as e:
        logger.warning("Failed to compute train metrics: %s", e)

    # Evaluate on validation set
    if val_loader is not None:
        logger.debug("Computing final metrics on val set...")
        try:
            val_preds = model.predict(val_loader)
            val_targets = get_targets_from_loader(val_loader)
            val_metrics = _compute_split_metrics(val_preds, val_targets, target_columns, "val")
            all_metrics.update(val_metrics)
            logger.info("Val metrics computed: %d entries", len(val_metrics))
        except Exception as e:
            logger.warning("Failed to compute val metrics: %s", e)

    # Evaluate on test set
    if test_loader is not None:
        logger.debug("Computing final metrics on test set...")
        try:
            test_preds = model.predict(test_loader)
            test_targets = get_targets_from_loader(test_loader)
            test_metrics = _compute_split_metrics(test_preds, test_targets, target_columns, "test")
            all_metrics.update(test_metrics)
            logger.info("Test metrics computed: %d entries", len(test_metrics))
        except Exception as e:
            logger.warning("Failed to compute test metrics: %s", e)

    return all_metrics


def compute_final_trial_metrics_from_dataframes(
    model: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    smiles_column: str,
    target_columns: list[str],
    batch_size: int = 64,
) -> dict[str, float]:
    """
    Compute comprehensive metrics from DataFrames (alternative API).

    This is a convenience function for models that work directly with DataFrames
    rather than data loaders. Internally creates data loaders and calls
    compute_final_trial_metrics.

    Parameters
    ----------
    model : Any
        Trained model with predict method.
    train_df : pd.DataFrame
        Training data with SMILES and target columns.
    val_df : pd.DataFrame or None
        Validation data.
    test_df : pd.DataFrame or None
        Test data.
    smiles_column : str
        Name of SMILES column.
    target_columns : list[str]
        Names of target columns.
    batch_size : int, default=64
        Batch size for data loaders.

    Returns
    -------
    dict[str, float]
        Same format as compute_final_trial_metrics.
    """
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    def df_to_loader(df: pd.DataFrame):
        """Convert DataFrame to data loader using chemprop v2 API."""
        smiles = df[smiles_column].tolist()
        targets = df[target_columns].values.tolist()
        datapoints = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, targets)]
        dataset = MoleculeDataset(datapoints)
        return build_dataloader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = df_to_loader(train_df)
    val_loader = df_to_loader(val_df) if val_df is not None else None
    test_loader = df_to_loader(test_df) if test_df is not None else None

    return compute_final_trial_metrics(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_columns=target_columns,
    )
