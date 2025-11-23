"""Ensemble-specific visualizations wrapping existing utilities.

Provides plotting utilities to render parity plots, metric bars, and blind
histograms for ensemble predictions. The functions assume the prediction
DataFrames follow the standard schema produced by
`admet.evaluate.ensemble` helpers (columns like `pred_{endpoint}_ensemble_log`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars
from admet.visualize.plots import plot_numeric_distributions


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
):
    """Plot parity plots and metric bars for a labeled dataset using the
    ensemble predictions and ground truth.

    Outputs two subfolders: `eval/plots/parity/{log,linear}/` and
    `eval/plots/metrics/{log,linear}/` with PNGs for each endpoint and
    metric respectively.
    """
    save_root = Path(save_root)
    # Build arrays
    y_true_log, y_pred_log, mask_log = _extract_true_and_pred_arrays_from_dfs(
        df_true, preds_log_df, endpoints, prefix="pred"
    )
    y_true_lin, y_pred_lin, mask_lin = _extract_true_and_pred_arrays_from_dfs(
        df_true, preds_lin_df, endpoints, prefix="pred_linear"
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
    plot_parity_grid(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_log,
        endpoints,
        space="log",
        save_dir=parity_dir_log,
        dpi=dpi,
        n_jobs=n_jobs,
    )
    plot_parity_grid(
        y_true_dict_lin,
        y_pred_dict_lin,
        mask_dict_lin,
        endpoints,
        space="linear",
        save_dir=parity_dir_lin,
        dpi=dpi,
        n_jobs=n_jobs,
    )

    # Metric bar charts: we'll reuse model_performance's plot_metric_bars
    metrics_dir_log = save_root / "plots" / "metrics" / "log"
    metrics_dir_lin = save_root / "plots" / "metrics" / "linear"
    # Provide representative save paths; function will create the parent dirs
    plot_metric_bars(
        y_true_dict_log,
        y_pred_dict_log,
        mask_dict_log,
        endpoints,
        space="log",
        save_path_r2=metrics_dir_log / "metrics_r2.png",
        save_path_spr2=metrics_dir_log / "metrics_spr2.png",
        n_jobs=n_jobs,
        dpi=dpi,
    )
    plot_metric_bars(
        y_true_dict_lin,
        y_pred_dict_lin,
        mask_dict_lin,
        endpoints,
        space="linear",
        save_path_r2=metrics_dir_lin / "metrics_r2.png",
        save_path_spr2=metrics_dir_lin / "metrics_spr2.png",
        n_jobs=n_jobs,
        dpi=dpi,
    )


def plot_blind_distributions(
    preds_log_df: pd.DataFrame, preds_lin_df: pd.DataFrame, endpoints: Sequence[str], save_root: Path
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

    for ep in endpoints:
        col_log = f"pred_{ep}_ensemble_log"
        # explicit col name for linear predictions; will be resolved below
        # Some codepaths may have `pred_{ep}_ensemble_log` for linear as well; fall back accordingly
        if col_log not in preds_log_df.columns:
            continue
        # Use short DF to plot single column
        df_log = preds_log_df[[col_log]].rename(columns={col_log: ep})
        prefix_lin = f"pred_linear_{ep}_ensemble_log"
        prefix_fallback = f"pred_{ep}_ensemble_log"
        col_lin_name = prefix_lin if prefix_lin in preds_lin_df.columns else prefix_fallback
        df_lin = preds_lin_df[[col_lin_name]]
        df_lin.columns = [ep]

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
