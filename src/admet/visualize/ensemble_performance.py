"""Ensemble-specific visualizations wrapping existing utilities.

Provides plotting utilities to render parity plots, metric bars, and blind
histograms for ensemble predictions. The functions assume the prediction
DataFrames follow the standard schema produced by
`admet.evaluate.ensemble` helpers (columns like `pred_{endpoint}_ensemble_log`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars
from admet.visualize.plots import plot_numeric_distributions

logger = logging.getLogger(__name__)


def _endpoints_from_metrics(metrics_df: pd.DataFrame | None) -> list[str]:
    """Return ordered endpoint names extracted from an aggregated metrics DataFrame."""
    if metrics_df is None or metrics_df.empty or "endpoint" not in metrics_df.columns:
        return []
    endpoints: list[str] = []
    for ep in metrics_df["endpoint"].tolist():
        if not isinstance(ep, str):
            continue
        if ep.lower() == "macro":
            continue
        if ep not in endpoints:
            endpoints.append(ep)
    return endpoints


def _extract_true_and_pred_arrays_from_dfs(
    df_true: pd.DataFrame, preds_df: pd.DataFrame, endpoints: Sequence[str], prefix: str = "pred"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, y_pred, mask for the arrays expected by plotting functions.

    y_true : shape (N, D)
    y_pred : shape (N, D) from ensemble predictions columns
    mask : boolean int array shape (N, D) extracted from y_true NaNs
    """
    y_true = df_true[list(endpoints)].to_numpy(dtype=float)
    y_pred_cols = [f"{prefix}_{ep}_ensemble_log" for ep in endpoints]
    y_pred = preds_df[y_pred_cols].to_numpy(dtype=float)
    mask = (~np.isnan(y_true)).astype(int)
    return y_true, y_pred, mask


def plot_labeled_ensemble(
    df_true: pd.DataFrame,
    preds_log_df: pd.DataFrame,
    preds_lin_df: pd.DataFrame,
    endpoints: Sequence[str],
    save_root: Path,
    n_jobs: int = 1,
    dpi: int = 150,
    metrics_by_endpoint: pd.DataFrame | None = None,
) -> None:
    """Plot parity plots and metric bars for a labeled dataset using the
    ensemble predictions and ground truth.

    Outputs two subfolders: `eval/plots/parity/{log,linear}/` and
    `eval/plots/metrics/{log,linear}/` with PNGs for each endpoint and
    metric respectively.
    """
    save_root = Path(save_root)

    resolved_endpoints = _endpoints_from_metrics(metrics_by_endpoint) or list(endpoints)
    available_endpoints = []
    for ep in resolved_endpoints:
        pred_col = f"pred_{ep}_ensemble_log"
        if ep in df_true.columns and pred_col in preds_log_df.columns:
            available_endpoints.append(ep)
    if not available_endpoints:
        available_endpoints = list(endpoints)
    if not available_endpoints:
        inferred = []
        for col in preds_log_df.columns:
            if col.startswith("pred_") and col.endswith("_ensemble_log"):
                inferred.append(col[len("pred_") : -len("_ensemble_log")])
        available_endpoints = inferred
    if not available_endpoints:
        logger.warning("No endpoints available for labeled ensemble plots; skipping visualization.")
        return
    resolved_endpoints = available_endpoints

    # Build arrays
    y_true_log, y_pred_log, mask_log = _extract_true_and_pred_arrays_from_dfs(
        df_true, preds_log_df, resolved_endpoints, prefix="pred"
    )
    y_true_lin, y_pred_lin, mask_lin = _extract_true_and_pred_arrays_from_dfs(
        df_true, preds_lin_df, resolved_endpoints, prefix="pred_linear"
    )

    # Use 'validation' split to mimic a single-split evaluation
    y_true_dict_log = {
        "train": np.zeros_like(y_true_log),
        "validation": y_true_log,
        "test": np.zeros_like(y_true_log),
    }
    y_pred_dict_log = {
        "train": np.zeros_like(y_pred_log),
        "validation": y_pred_log,
        "test": np.zeros_like(y_pred_log),
    }
    mask_dict_log = {
        "train": np.zeros_like(mask_log),
        "validation": mask_log,
        "test": np.zeros_like(mask_log),
    }

    y_true_dict_lin = {
        "train": np.zeros_like(y_true_lin),
        "validation": y_true_lin,
        "test": np.zeros_like(y_true_lin),
    }
    y_pred_dict_lin = {
        "train": np.zeros_like(y_pred_lin),
        "validation": y_pred_lin,
        "test": np.zeros_like(y_pred_lin),
    }
    mask_dict_lin = {
        "train": np.zeros_like(mask_lin),
        "validation": mask_lin,
        "test": np.zeros_like(mask_lin),
    }

    # Parity plots for log and linear
    parity_dir_log = save_root / "plots" / "parity" / "log"
    parity_dir_lin = save_root / "plots" / "parity" / "linear"

    logger.info("Plotting parity plots to %s and %s", parity_dir_log, parity_dir_lin)
    plot_parity_grid(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_log,
        resolved_endpoints,
        space="log",
        save_dir=parity_dir_log,
        dpi=dpi,
        n_jobs=n_jobs,
    )
    plot_parity_grid(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_lin,
        resolved_endpoints,
        space="linear",
        save_dir=parity_dir_lin,
        dpi=dpi,
        n_jobs=n_jobs,
    )

    # Metric bar charts: we'll reuse model_performance's plot_metric_bars
    metrics_dir_log = save_root / "plots" / "metrics" / "log"
    metrics_dir_lin = save_root / "plots" / "metrics" / "linear"

    # Provide representative save paths; function will create the parent dirs
    logger.info("Plotting metric bars to %s and %s", metrics_dir_log, metrics_dir_lin)
    plot_metric_bars(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_log,
        resolved_endpoints,
        space="log",
        save_path_r2=metrics_dir_log / "metrics_r2.png",
        save_path_spr2=metrics_dir_log / "metrics_spr2.png",
        n_jobs=n_jobs,
        dpi=dpi,
    )
    plot_metric_bars(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_lin,
        resolved_endpoints,
        space="linear",
        save_path_r2=metrics_dir_lin / "metrics_r2.png",
        save_path_spr2=metrics_dir_lin / "metrics_spr2.png",
        n_jobs=n_jobs,
        dpi=dpi,
    )


def plot_blind_distributions(
    preds_log_df: pd.DataFrame,
    preds_lin_df: pd.DataFrame,
    endpoints: Sequence[str],
    save_root: Path,
    metrics_by_endpoint: pd.DataFrame | None = None,
) -> None:
    """Plot histogram + KDE per endpoint for the blind ensemble predictions.

    Produces files under `blind/plots/distributions/{log,linear}/`.
    """
    save_root = Path(save_root)
    distributions_log_dir = save_root / "plots" / "distributions" / "log"
    distributions_lin_dir = save_root / "plots" / "distributions" / "linear"
    # Build a tidy DataFrame with endpoint columns
    # Derived column name patterns for clarity; selection is per-endpoint below

    # `predictions_to_dataframe` may use 'pred_linear_{ep}_ensemble_log'. Attempt
    # to handle both cases when selecting columns below.

    resolved_endpoints = _endpoints_from_metrics(metrics_by_endpoint) or list(endpoints)
    if not resolved_endpoints:
        inferred = []
        for col in preds_log_df.columns:
            if col.startswith("pred_") and col.endswith("_ensemble_log"):
                inferred.append(col[len("pred_") : -len("_ensemble_log")])
        resolved_endpoints = inferred

    filtered_endpoints = []
    for ep in resolved_endpoints:
        if f"pred_{ep}_ensemble_log" in preds_log_df.columns:
            filtered_endpoints.append(ep)
    if not filtered_endpoints:
        logger.warning("No endpoints available for blind distribution plots; skipping visualization.")
        return
    resolved_endpoints = filtered_endpoints

    for ep in resolved_endpoints:
        col_log = f"pred_{ep}_ensemble_log"
        # Use short DF to plot single column
        df_log = preds_log_df[[col_log]].rename(columns={col_log: ep})
        prefix_lin = f"pred_linear_{ep}_ensemble_log"
        prefix_fallback = f"pred_{ep}_ensemble_log"
        col_lin_name = prefix_lin if prefix_lin in preds_lin_df.columns else prefix_fallback
        df_lin = preds_lin_df[[col_lin_name]]
        df_lin.columns = [ep]

        logger.info("Plotting blind distributions for endpoint %s", ep)
        plot_numeric_distributions(
            df_log,
            columns=[ep],
            n_cols=1,
            title=f"Blind distribution - {ep} (log)",
            save_path=distributions_log_dir / f"hist_kde_{ep.replace(' ', '_').replace('/', '_')}.png",
        )
        plot_numeric_distributions(
            df_lin,
            columns=[ep],
            n_cols=1,
            title=f"Blind distribution - {ep} (linear)",
            save_path=distributions_lin_dir / f"hist_kde_{ep.replace(' ', '_').replace('/', '_')}.png",
        )


__all__ = [
    "plot_labeled_ensemble",
    "plot_blind_distributions",
]
