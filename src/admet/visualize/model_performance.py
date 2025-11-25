"""admet.visualize.model_performance
====================================

Visualization utilities for multi‑endpoint model performance.

Provides:

* Per‑endpoint parity plots across ``train``/``validation``/``test`` splits
    in log10 and (optionally transformed) linear spaces.
* Grouped bar charts for metrics (MAE, RMSE, R², Pearson r², Spearman ρ²,
    Kendall τ) per endpoint and split.

Input Data Schema
-----------------
Functions expect dictionaries keyed by split name containing 2‑D numpy arrays
with shape ``(N_samples, N_endpoints)`` plus a parallel boolean/int mask of the
same shape indicating presence (1/True) of a target.

Assumptions
-----------
* Endpoints are ordered consistently across all arrays.
* Log space values are log10‑transformed for all endpoints except ``LogD``
    which is left linear; conversions to linear space exponentiate non‑``LogD``
    columns.
* Metric calculations reuse :func:`admet.evaluate.metrics.compute_metrics`
    ensuring consistency with training/evaluation pipelines.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from admet.evaluate import metrics as eval_metrics


# _safe_endpoint_slug is defined in ensemble_eval; redefine lightweight here to avoid circular deps.
def _safe_endpoint_slug(endpoint: str) -> str:
    """Return a filesystem-safe slug for an endpoint name."""
    replacements = {" ": "_", "/": "_", "<": "lt", ">": "gt"}
    out = endpoint
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


_MAX_LOG10_BEFORE_OVERFLOW = 308.0  # log10(np.finfo(float).max) ~ 308.254


def _safe_power10(values: np.ndarray) -> np.ndarray:
    """Exponentiate base-10 values with clipping to avoid overflow."""

    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, a_min=None, a_max=_MAX_LOG10_BEFORE_OVERFLOW)
    result = np.power(10.0, clipped)
    # Replace any residual infs/NaNs with NaN for downstream handling
    result[~np.isfinite(result)] = np.nan
    return result


def to_linear(values: np.ndarray, endpoint: str) -> np.ndarray:
    """Convert log10 values to linear space for non-``LogD`` endpoints.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array of log10 values.
    endpoint : str
        Endpoint name (``LogD`` is exempt from exponentiation).

    Returns
    -------
    numpy.ndarray
        Transformed 1-D array (unchanged for ``LogD``).
    """
    if endpoint == "LogD":
        return values
    return np.power(10.0, values)


def _apply_transform_space(y: np.ndarray, endpoints: Sequence[str], space: str) -> np.ndarray:
    """Return array in requested plotting space.

    Parameters
    ----------
    y : numpy.ndarray
        Array of shape ``(N, D)`` in log space.
    endpoints : Sequence[str]
        Endpoint names ordered along ``D`` axis.
    space : str
        Either ``'log'`` or ``'linear'``.

    Returns
    -------
    numpy.ndarray
        Transformed array (copy); exponentiated for non‑``LogD`` endpoints
        when ``space='linear'``.
    """
    if space == "log":
        return y
    out = y.copy()
    for j, ep in enumerate(endpoints):
        if ep != "LogD":
            out[:, j] = _safe_power10(out[:, j])
    return out


def _masked_arrays(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract valid true/pred arrays for endpoint index ``j``.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Arrays with shape ``(N, D)``.
    mask : numpy.ndarray
        Presence mask (1/True -> present) same shape.
    j : int
        Endpoint column index.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y_true_valid, y_pred_valid)`` 1-D arrays.
    """
    valid = mask[:, j].astype(bool)
    return y_true[valid, j], y_pred[valid, j]


def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics for a single endpoint column.

    Reuses :func:`admet.evaluate.metrics.compute_metrics` (called on a single
    pseudo-endpoint) to ensure numeric parity with training evaluation.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        1-D arrays of true and predicted values.

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``R2``, ``pearson_r2``, ``spearman_rho2``,
        ``kendall_tau`` (NaNs if insufficient data).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    n = y_true.shape[0]
    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "R2": float("nan"),
            "pearson_r2": float("nan"),
            "spearman_rho2": float("nan"),
            "kendall_tau": float("nan"),
        }

    mask = np.ones((n, 1), dtype=int)
    metrics = eval_metrics.compute_metrics(y_true.reshape(-1, 1), y_pred.reshape(-1, 1), mask, endpoints=["ep"])
    return {
        "mae": metrics["ep"]["mae"],
        "rmse": metrics["ep"]["rmse"],
        "R2": metrics["ep"]["R2"],
        "pearson_r2": metrics["ep"]["pearson_r2"],
        "spearman_rho2": metrics["ep"]["spearman_rho2"],
        "kendall_tau": metrics["ep"]["kendall_tau"],
    }


def _plot_parity_worker(
    j: int,
    ep: str,
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, np.ndarray],
    space: str,
    save_dir: Path,
    dpi: int = 600,
) -> None:
    """Render a parity plot for a single endpoint.

    Picklable worker enabling multi-process execution.

    Parameters
    ----------
    j : int
        Endpoint index.
    ep : str
        Endpoint name.
    y_true_dict, y_pred_dict, mask_dict : dict[str, numpy.ndarray]
        Split keyed arrays of shape ``(N, D)``.
    space : str
        Plotting space (``'log'`` or ``'linear'``).
    save_dir : pathlib.Path
        Directory to write figure PNG.
    dpi : int, optional
        Render DPI (default 600).
    """
    splits = ["train", "validation", "test"]
    title_space = "(log10)" if space == "log" else "(linear)"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    fig.suptitle(f"Parity Plots - {ep} {title_space}", fontsize=18)

    # Prepare pooled axis limits for consistent scale
    pooled_vals = []
    for s in splits:
        y_t = y_true_dict[s][:, j]
        y_p = y_pred_dict[s][:, j]
        mask = mask_dict[s][:, j]
        valid = mask.astype(bool)
        if np.any(valid):
            t = y_t[valid]
            p = y_p[valid]
            if space == "linear" and ep != "LogD":
                t = _safe_power10(t)
                p = _safe_power10(p)
            pooled_vals.append(t)
            pooled_vals.append(p)
    if pooled_vals:
        combined = np.concatenate(pooled_vals)
        lo = float(np.nanpercentile(combined, 1))
        hi = float(np.nanpercentile(combined, 99))
        pad = 0.1 * (hi - lo) if hi > lo else 0.1 * abs(lo) if lo != 0 else 1.0
        x_min, x_max = lo - pad, hi + pad
    else:
        x_min, x_max = 0.0, 1.0

    for ax, s in zip(axes, splits):
        y_t = y_true_dict[s][:, j]
        y_p = y_pred_dict[s][:, j]
        mask = mask_dict[s][:, j]
        y_t_valid, y_p_valid = _masked_arrays(y_t.reshape(-1, 1), y_p.reshape(-1, 1), mask.reshape(-1, 1), 0)
        if space == "linear" and ep != "LogD":
            y_t_valid = _safe_power10(y_t_valid)
            y_p_valid = _safe_power10(y_p_valid)

        if y_t_valid.size == 0:
            ax.text(
                0.5,
                0.5,
                "No valid points",
                fontsize=10,
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.3",
                    "linewidth": 0.6,
                    "alpha": 0.7,
                },
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel("True", fontsize=14)
            ax.set_ylabel("Predicted", fontsize=14)
            continue

        ax.scatter(y_t_valid, y_p_valid, alpha=0.7, s=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.plot([x_min, x_max], [x_min, x_max], ls="--", color="gray")
        ax.set_xlabel("True", fontsize=14)
        ax.set_ylabel("Predicted", fontsize=14)
        ax.set_title(f"Dataset Split: {s.capitalize()}", fontsize=16)
        ax.grid(True)

        stats = _compute_stats(y_t_valid, y_p_valid)
        stats_text = (
            f"n: {y_t_valid.shape[0]}\n"
            f"MAE: {stats['mae']:.3g}\n"
            f"RMSE: {stats['rmse']:.3g}\n"
            f"$R^2$: {stats['R2']:.3g}\n"
            f"Pearson $r^2$: {stats['pearson_r2']:.3g}\n"
            f"Spearman $\\rho^2$: {stats['spearman_rho2']:.3g}\n"
            f"Kendall $\\tau$: {stats['kendall_tau']:.3g}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.3",
                "linewidth": 0.6,
                "alpha": 0.7,
            },
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fpath = Path(save_dir) / f"parity_{_safe_endpoint_slug(ep)}.png"
    fig.savefig(fpath, dpi=dpi)
    plt.close(fig)


def _metric_plot_worker(
    metric_key: str,
    per_split_metrics: Dict[str, Dict[str, Dict[str, float]]],
    endpoints: Sequence[str],
    save_path: Path,
    space: str,
    dpi: int = 600,
) -> None:
    """Render a grouped bar chart for a single metric across splits.

    Parameters
    ----------
    metric_key : str
        Metric name (e.g. ``'R2'``).
    per_split_metrics : dict
        Mapping split -> endpoint -> metric dict.
    endpoints : Sequence[str]
        Endpoint names.
    save_path : pathlib.Path
        Output PNG file path.
    space : str
        Plotting space label for title.
    dpi : int, optional
        Figure DPI.
    """
    splits = ["train", "validation", "test"]
    labels = [ep.replace(" ", "\n") for ep in endpoints]
    x = np.arange(len(endpoints))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, len(endpoints) * 1.0), 6), dpi=dpi)
    for i, s in enumerate(splits):
        vals = np.array(
            [per_split_metrics[s][ep].get(metric_key, float("nan")) for ep in endpoints],
            dtype=float,
        )
        ax.bar(x + (i - 1) * width, vals, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    key_lower = metric_key.lower()
    if key_lower == "r2":
        ax.set_ylabel(r"$R^2$", fontsize=14)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"$R^2$ by Endpoint ({space})", fontsize=16)
    elif key_lower == "pearson_r2":
        ax.set_ylabel(r"Pearson $r^2$", fontsize=14)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Pearson $r^2$ by Endpoint ({space})", fontsize=16)
    elif key_lower == "spearman_rho2":
        ax.set_ylabel(r"Spearman $\rho^2$", fontsize=14)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Spearman $\\rho^2$ by Endpoint ({space})", fontsize=16)
    elif key_lower == "kendall_tau":
        ax.set_ylabel(r"Kendall $\tau$", fontsize=14)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"Kendall $\\tau$ by Endpoint ({space})", fontsize=16)
    else:
        ax.set_ylabel(metric_key.upper(), fontsize=14)
        ax.set_title(f"{metric_key.upper()} by Endpoint ({space})", fontsize=16)

    ax.grid(axis="y")

    # add legend with border box
    ax.legend(
        title="Dataset Split",
        fontsize=12,
        title_fontsize=12,
        frameon=True,
        edgecolor="black",
        framealpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_parity_grid(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, np.ndarray],
    endpoints: Sequence[str],
    *,
    space: str,
    save_dir: Path,
    dpi: int = 600,
    n_jobs: int = 1,
) -> None:
    """Generate and persist parity plots for each endpoint.

    Parameters
    ----------
    y_true_dict, y_pred_dict, mask_dict : dict[str, numpy.ndarray]
        Split keyed arrays with shape ``(N, D)``.
    endpoints : Sequence[str]
        Endpoint names.
    space : str
        ``'log'`` or ``'linear'``.
    save_dir : pathlib.Path
        Directory to write PNG files (created if missing).
    dpi : int, optional
        Figure DPI (default 600).
    n_jobs : int, optional
        Parallel processes; values <=1 run sequentially.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # nothing to precompute here; workers handle splits per endpoint
    # Run plotting sequentially or in parallel with progress bar
    if n_jobs is None or n_jobs <= 1:
        for j, ep in enumerate(tqdm(list(endpoints), desc="Parity plots")):
            _plot_parity_worker(j, ep, y_true_dict, y_pred_dict, mask_dict, space, save_dir, dpi)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_plot_parity_worker, j, ep, y_true_dict, y_pred_dict, mask_dict, space, save_dir, dpi)
                for j, ep in enumerate(endpoints)
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parity plots"):
                # re-raise exceptions from workers
                fut.result()


def plot_metric_bars(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, np.ndarray],
    endpoints: Sequence[str],
    *,
    space: str,
    save_path_r2: Path,
    save_path_spr2: Path,
    dpi: int = 600,
    n_jobs: int = 1,
) -> None:
    """Generate grouped bar charts for multiple metrics.

    Parameters
    ----------
    y_true_dict, y_pred_dict, mask_dict : dict[str, numpy.ndarray]
        Split keyed arrays (``(N, D)``).
    endpoints : Sequence[str]
        Endpoint names.
    space : str
        Plot space (``'log'`` or ``'linear'``) used for transformations.
    save_path_r2, save_path_spr2 : pathlib.Path
        Representative output paths; all metric plots saved adjacent to
        ``save_path_r2`` directory.
    dpi : int, optional
        DPI for all figures.
    n_jobs : int, optional
        Parallelism level for metric plotting (ProcessPoolExecutor).
    """
    save_path_r2 = Path(save_path_r2)
    save_path_r2.parent.mkdir(parents=True, exist_ok=True)
    save_path_spr2 = Path(save_path_spr2)
    save_path_spr2.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]
    # Compute per-split metrics for all endpoints at once using compute_metrics
    per_split_metrics: dict[str, dict] = {}
    for s in splits:
        y_t = y_true_dict[s]
        y_p = y_pred_dict[s]
        mask = mask_dict[s]
        # If requested, transform to linear space for non-LogD endpoints
        if space == "linear":
            y_t_plot = _apply_transform_space(y_t, endpoints, space)
            y_p_plot = _apply_transform_space(y_p, endpoints, space)
        else:
            y_t_plot = y_t
            y_p_plot = y_p
        per_split_metrics[s] = eval_metrics.compute_metrics(y_t_plot, y_p_plot, mask, endpoints)

    metrics_to_plot = ["mae", "rmse", "R2", "pearson_r2", "spearman_rho2", "kendall_tau"]

    # Save all metrics to separate PNGs in save_path_r2.parent
    base_dir = Path(save_path_r2).parent
    tasks = []
    for metric_key in metrics_to_plot:
        out_path = base_dir / f"metrics_{metric_key.lower().replace(' ', '_')}.png"
        tasks.append((metric_key, str(out_path)))

    if n_jobs is None or n_jobs <= 1:
        for metric_key, out_str in tqdm(tasks, desc="Metric bars"):
            _metric_plot_worker(metric_key, per_split_metrics, endpoints, Path(out_str), space, dpi)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_metric_plot_worker, metric_key, per_split_metrics, endpoints, Path(out_str), space, dpi)
                for metric_key, out_str in tasks
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Metric bars"):
                fut.result()
