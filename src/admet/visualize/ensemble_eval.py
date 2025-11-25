"""Visualization helpers for ensemble evaluation CLI outputs.

This module bridges the CSV artifacts written by ``ensemble_eval`` and the
existing plotting utilities under :mod:`admet.visualize`. It renders:

* Parity plots (log10 + linear) for labeled evaluation datasets.
* Metric bar charts for ensemble vs. aggregated-model metrics, including
  std-error bars for the aggregate statistics.
* Histograms with KDE overlays for blind prediction distributions in both
  log10 and linear space.

Functions here accept the in-memory DataFrames returned from
``admet.evaluate.ensemble`` to avoid re-reading CSVs in the CLI and to keep
logic reusable for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from admet.visualize import GLASBEY_PALETTE
from admet.visualize.model_performance import _apply_transform_space, plot_parity_grid

_METRICS = ["mae", "rmse", "R2", "pearson_r2", "spearman_rho2", "kendall_tau"]


def _safe_endpoint_slug(endpoint: str) -> str:
    """Return a filesystem-safe slug for an endpoint name."""
    replacements = {" ": "_", "/": "_", "<": "lt", ">": "gt"}
    out = endpoint
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


def _resolve_endpoints_from_preds(preds_df: pd.DataFrame) -> list[str]:
    """Return ordered endpoint names by inspecting prediction columns."""
    cols = [c for c in preds_df.columns if c not in ("Molecule Name", "SMILES")]
    return cols


def _build_split_dicts(
    df_eval: pd.DataFrame, preds_df: pd.DataFrame, endpoints: Sequence[str], active_split: str = "validation"
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Construct split dictionaries compatible with plot_parity_grid.

    Parameters
    ----------
    df_eval : pandas.DataFrame
        DataFrame containing true values for endpoints.
    preds_df : pandas.DataFrame
        DataFrame containing predicted values for endpoints.
    endpoints : Sequence[str]
        Ordered endpoint names.
    active_split : str, optional
        Which split should carry the true/pred data. Remaining splits are filled
        with zeros so the plotting utilities receive the expected keys.
    """
    if active_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split name '{active_split}' for parity plotting")
    df_true = df_eval[endpoints].apply(pd.to_numeric, errors="coerce")
    df_preds = preds_df[endpoints].apply(pd.to_numeric, errors="coerce")
    y_true = df_true.to_numpy(dtype=float)
    y_pred = df_preds.to_numpy(dtype=float)
    mask = (~np.isnan(y_true)).astype(int)

    zeros = np.zeros_like(y_true)
    zeros_pred = np.zeros_like(y_pred)
    zeros_mask = np.zeros_like(mask)
    y_true_dict = {"train": zeros.copy(), "validation": zeros.copy(), "test": zeros.copy()}
    y_pred_dict = {"train": zeros_pred.copy(), "validation": zeros_pred.copy(), "test": zeros_pred.copy()}
    mask_dict = {"train": zeros_mask.copy(), "validation": zeros_mask.copy(), "test": zeros_mask.copy()}

    y_true_dict[active_split] = y_true
    y_pred_dict[active_split] = y_pred
    mask_dict[active_split] = mask
    return y_true_dict, y_pred_dict, mask_dict


def _ensure_endpoint_order(
    endpoints: Sequence[str], include_macro: bool, metrics_df: Optional[pd.DataFrame]
) -> List[str]:
    """Return endpoints enforcing requested ordering (plus macro if present)."""
    ordered = list(endpoints)
    if include_macro and metrics_df is not None and "endpoint" in metrics_df.columns:
        if "macro" in metrics_df["endpoint"].unique() and "macro" not in ordered:
            ordered.append("macro")
    return ordered


def _select_metric_values(
    metrics_df: pd.DataFrame,
    metric_type: str,
    metric_name: str,
    endpoints: Sequence[str],
) -> np.ndarray:
    """Return values for a metric type ordered by endpoints."""
    subset = metrics_df[(metrics_df["type"] == metric_type) & (metrics_df["metric"] == metric_name)]
    lookup = subset.set_index("endpoint")["value"]
    values = []
    for ep in endpoints:
        val = pd.to_numeric(lookup.get(ep, np.nan), errors="coerce")
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _metric_bar_plot(
    values: np.ndarray,
    errors: Optional[np.ndarray],
    endpoints: Sequence[str],
    metric_name: str,
    title_prefix: str,
    space: str,
    save_path: Path,
    dpi: int = 600,
) -> None:
    """Render a single bar chart with optional error bars."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(endpoints))
    fig_w = max(6.0, len(endpoints) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=dpi)
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=6 if errors is not None else 0,
        color=GLASBEY_PALETTE[0],
        edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=45, ha="right")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{title_prefix} - {metric_name.upper()} ({space})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # Annotate NaNs to clarify missing metrics
    for rect, val in zip(bars, values):
        if not np.isfinite(val):
            ax.text(rect.get_x() + rect.get_width() / 2, 0.02, "NaN", rotation=90, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_metric_family(
    metrics_df: pd.DataFrame,
    metric_type: str,
    error_type: Optional[str],
    endpoints: Sequence[str],
    space: str,
    out_dir: Path,
    dpi: int,
) -> None:
    """Plot all metric variants for a specific ``type`` entry."""
    for metric_name in _METRICS:
        values = _select_metric_values(metrics_df, metric_type, metric_name, endpoints)
        errors = None
        if error_type is not None:
            errors = _select_metric_values(metrics_df, error_type, metric_name, endpoints)
        fname = f"{metric_type}_{metric_name}.png"
        _metric_bar_plot(
            values,
            errors,
            endpoints,
            metric_name,
            title_prefix=metric_type.replace("_", " ").title(),
            space=space,
            save_path=out_dir / fname,
            dpi=dpi,
        )


def _plot_true_vs_pred_distribution(
    true_series: pd.Series,
    pred_with_true: pd.Series,
    pred_without_true: pd.Series,
    total_count: int,
    endpoint: str,
    space_label: str,
    save_dir: Path,
    *,
    dpi: int = 600,
) -> None:
    true_clean = pd.to_numeric(true_series, errors="coerce").dropna()
    pred_with_true_clean = pd.to_numeric(pred_with_true, errors="coerce").dropna()
    pred_without_true_clean = pd.to_numeric(pred_without_true, errors="coerce").dropna()
    if true_clean.empty and pred_with_true_clean.empty and pred_without_true_clean.empty:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    true_color = GLASBEY_PALETTE[0]
    pred_with_true_color = GLASBEY_PALETTE[1]
    pred_without_true_color = GLASBEY_PALETTE[2]
    if not true_clean.empty:
        sns.histplot(
            true_clean,
            kde=True,
            stat="density",
            color=true_color,
            alpha=0.45,
            ax=ax,
            label=f"True (n={len(true_clean)})",
        )
    if not pred_with_true_clean.empty:
        percent_true = (len(pred_with_true_clean) / total_count) * 100 if total_count else 0.0
        sns.histplot(
            pred_with_true_clean,
            kde=True,
            stat="density",
            color=pred_with_true_color,
            alpha=0.45,
            ax=ax,
            label=f"Predicted (with true) (n={len(pred_with_true_clean)}, {percent_true:.1f}%)",
        )
    if not pred_without_true_clean.empty:
        sns.histplot(
            pred_without_true_clean,
            kde=True,
            stat="density",
            color=pred_without_true_color,
            alpha=0.45,
            ax=ax,
            label=f"Predicted (no true) (n={len(pred_without_true_clean)})",
        )
    ax.set_title(f"{endpoint} distribution ({space_label})")
    ax.set_xlabel(endpoint)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / f"{_safe_endpoint_slug(endpoint)}_{space_label}.png", dpi=dpi)
    plt.close(fig)


def _plot_labeled_distribution_family(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    endpoints: Sequence[str],
    save_dir: Path,
    space_label: str,
    dpi: int,
) -> None:
    if df_true is None or df_pred is None:
        return
    for ep in endpoints:
        if ep not in df_pred.columns:
            continue
        true_series = df_true[ep] if ep in df_true.columns else pd.Series(dtype=float)
        mask_true = true_series.notna()
        mask_true_aligned = mask_true.reindex(df_pred.index, fill_value=False)
        pred_series = df_pred[ep]
        pred_with_true = pred_series[mask_true_aligned]
        pred_without_true = pred_series[~mask_true_aligned]
        overlap = pred_with_true.index.intersection(pred_without_true.index)
        if not overlap.empty:
            msg = f"Prediction partitions overlap for endpoint '{ep}' in {space_label} space: " f"{overlap.tolist()}"
            raise ValueError(msg)
        combined = len(pred_with_true) + len(pred_without_true)
        if combined != len(pred_series):
            raise ValueError(
                f"Prediction partitions do not cover dataset for endpoint '{ep}' "
                f"in {space_label} space (got {combined} vs expected {len(pred_series)})"
            )
        _plot_true_vs_pred_distribution(
            true_series[mask_true],
            pred_with_true,
            pred_without_true,
            total_count=len(pred_series),
            endpoint=ep,
            space_label=space_label,
            save_dir=save_dir,
            dpi=dpi,
        )


def plot_labeled_eval_outputs(
    df_eval: pd.DataFrame,
    preds_log_df: pd.DataFrame,
    preds_linear_df: Optional[pd.DataFrame],
    metrics_log_df: Optional[pd.DataFrame],
    metrics_linear_df: Optional[pd.DataFrame],
    figures_dir: Path,
    *,
    dpi: int = 600,
    n_jobs: int = 1,
    parity_split: str = "validation",
) -> None:
    """Render parity plots and metric bar charts for labeled ensemble evals.

    Parameters
    ----------
    parity_split : str, optional
        Which split label to attribute the provided data to when rendering
        parity plots (train|validation|test). Defaults to ``"validation"`` to
        match the historical behavior for external eval sets.
    """
    if preds_log_df is None or df_eval is None:
        return
    endpoints = _resolve_endpoints_from_preds(preds_log_df)
    if not endpoints:
        return

    figures_dir = Path(figures_dir)
    df_true_log = df_eval[endpoints].apply(pd.to_numeric, errors="coerce")
    preds_log_numeric = preds_log_df[endpoints].apply(pd.to_numeric, errors="coerce")
    df_true_linear = pd.DataFrame(
        _apply_transform_space(df_true_log.to_numpy(dtype=float), endpoints, space="linear"),
        columns=endpoints,
        index=df_eval.index,
    )
    if preds_linear_df is not None:
        preds_linear_numeric = preds_linear_df[endpoints].apply(pd.to_numeric, errors="coerce")
    else:
        preds_linear_numeric = pd.DataFrame(
            _apply_transform_space(preds_log_numeric.to_numpy(dtype=float), endpoints, space="linear"),
            columns=endpoints,
            index=preds_log_numeric.index,
        )

    # Parity plots (log + linear)
    y_true_dict, y_pred_dict, mask_dict = _build_split_dicts(
        df_eval, preds_log_df, endpoints, active_split=parity_split
    )
    parity_root = figures_dir / "parity"
    plot_parity_grid(
        y_true_dict,
        y_pred_dict,
        mask_dict,
        endpoints,
        space="log",
        save_dir=parity_root / "log",
        dpi=dpi,
        n_jobs=n_jobs,
    )
    plot_parity_grid(
        y_true_dict,
        y_pred_dict,
        mask_dict,
        endpoints,
        space="linear",
        save_dir=parity_root / "linear",
        dpi=dpi,
        n_jobs=n_jobs,
    )

    # Metric bar plots
    metrics_root = figures_dir / "metrics"
    metrics_for_order = metrics_log_df if metrics_log_df is not None else metrics_linear_df
    endpoints_with_macro = _ensure_endpoint_order(endpoints, include_macro=True, metrics_df=metrics_for_order)

    if metrics_log_df is not None and not metrics_log_df.empty:
        _plot_metric_family(
            metrics_log_df,
            metric_type="models_agg_mean_log",
            error_type="models_agg_stderr_log",
            endpoints=endpoints_with_macro,
            space="log",
            out_dir=metrics_root / "models_agg_mean_log",
            dpi=dpi,
        )
        _plot_metric_family(
            metrics_log_df,
            metric_type="ensemble_log",
            error_type=None,
            endpoints=endpoints_with_macro,
            space="log",
            out_dir=metrics_root / "ensemble_log",
            dpi=dpi,
        )

    if metrics_linear_df is not None and not metrics_linear_df.empty:
        _plot_metric_family(
            metrics_linear_df,
            metric_type="models_agg_mean_linear",
            error_type="models_agg_stderr_linear",
            endpoints=endpoints_with_macro,
            space="linear",
            out_dir=metrics_root / "models_agg_mean_linear",
            dpi=dpi,
        )
        _plot_metric_family(
            metrics_linear_df,
            metric_type="ensemble_linear",
            error_type=None,
            endpoints=endpoints_with_macro,
            space="linear",
            out_dir=metrics_root / "ensemble_linear",
            dpi=dpi,
        )

    distributions_root = figures_dir / "distributions"
    _plot_labeled_distribution_family(
        df_true=df_true_log,
        df_pred=preds_log_numeric,
        endpoints=endpoints,
        save_dir=distributions_root / "log",
        space_label="log",
        dpi=dpi,
    )
    _plot_labeled_distribution_family(
        df_true=df_true_linear,
        df_pred=preds_linear_numeric,
        endpoints=endpoints,
        save_dir=distributions_root / "linear",
        space_label="linear",
        dpi=dpi,
    )


def _plot_endpoint_distribution(
    preds_df: pd.DataFrame,
    endpoint: str,
    space_label: str,
    save_dir: Path,
    *,
    dpi: int = 600,
) -> None:
    """Plot histogram + KDE for a single endpoint column."""
    series = pd.to_numeric(preds_df[endpoint], errors="coerce").dropna()
    if series.empty:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_safe_endpoint_slug(endpoint)}_{space_label}.png"
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    sns.histplot(
        series,
        kde=True,
        ax=ax,
        color=GLASBEY_PALETTE[3],
        label=f"Predicted (n={len(series)})",
    )
    ax.set_title(f"{endpoint} distribution ({space_label})")
    ax.set_xlabel(endpoint)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / fname, dpi=dpi)
    plt.close(fig)


def plot_blind_distributions(
    preds_log_df: pd.DataFrame,
    preds_linear_df: Optional[pd.DataFrame],
    figures_dir: Path,
    *,
    dpi: int = 600,
) -> None:
    """Render histograms for blind predictions (log + linear)."""
    if preds_log_df is None:
        return
    endpoints = _resolve_endpoints_from_preds(preds_log_df)
    if not endpoints:
        return

    figures_dir = Path(figures_dir)
    log_dir = figures_dir / "distributions" / "log"
    lin_dir = figures_dir / "distributions" / "linear"

    for ep in endpoints:
        _plot_endpoint_distribution(preds_log_df, ep, "log", log_dir, dpi=dpi)
        if preds_linear_df is not None and ep in preds_linear_df.columns:
            _plot_endpoint_distribution(preds_linear_df, ep, "linear", lin_dir, dpi=dpi)


__all__ = [
    "plot_labeled_eval_outputs",
    "plot_blind_distributions",
]
