"""Visualization utilities for XGBoost model performance.

Implements parity plots (per endpoint across splits) and grouped bar charts for
R^2 and Spearman rho^2 across train/val/test splits in log and linear spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import concurrent.futures
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import spearmanr

from admet.evaluate import metrics as eval_metrics


def to_linear(values: np.ndarray, endpoint: str) -> np.ndarray:
    """Convert log10-values to linear for all endpoints except `LogD`.

    Args:
        values: Array with shape (n_samples,). Values expected in log10 form.
        endpoint: Endpoint name.
    Returns:
        Linear-space values as numpy array.
    """
    if endpoint == "LogD":
        return values
    return np.power(10.0, values)


def _apply_transform_space(y: np.ndarray, endpoints: Sequence[str], space: str) -> np.ndarray:
    if space == "log":
        return y
    out = y.copy()
    for j, ep in enumerate(endpoints):
        if ep != "LogD":
            out[:, j] = np.power(10.0, out[:, j])
    return out


def _masked_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, j: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask[:, j] == 1
    return y_true[valid, j], y_pred[valid, j]


def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute display statistics for a single endpoint column.

    Uses `compute_metrics` for MAE, RMSE, and R² to ensure parity with the
    evaluation pipeline. Spearman rho (not squared) is returned via
    `scipy.stats.spearmanr`.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.shape[0]
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "spearman": float("nan")}
    mask = np.ones((n, 1), dtype=int)
    metrics = eval_metrics.compute_metrics(
        y_true.reshape(-1, 1), y_pred.reshape(-1, 1), mask, endpoints=["ep"]
    )
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
    """Top-level worker that draws a parity plot for a single endpoint.

    This function is picklable and suitable for running in a multi-process pool.
    """
    # Use local matplotlib state inside worker to avoid backend/style issues in
    # multiprocessing child processes.
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    try:
        plt.style.use("science")
    except Exception:
        pass

    splits = ["train", "val", "test"]
    title_space = "(log10)" if space == "log" else "(linear)"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    fig.suptitle(f"Parity Plots - {ep} {title_space}")

    # Prepare pooled axis limits for consistent scale
    pooled_vals = []
    for s in splits:
        y_t = y_true_dict[s][:, j]
        y_p = y_pred_dict[s][:, j]
        mask = mask_dict[s][:, j]
        valid = mask == 1
        if np.any(valid):
            t = y_t[valid]
            p = y_p[valid]
            if space == "linear" and ep != "LogD":
                t = np.power(10.0, t)
                p = np.power(10.0, p)
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
            y_t_valid = np.power(10.0, y_t_valid)
            y_p_valid = np.power(10.0, y_p_valid)

        if y_t_valid.size == 0:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            continue

        ax.scatter(y_t_valid, y_p_valid, alpha=0.7, s=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.plot([x_min, x_max], [x_min, x_max], ls="--", color="gray")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

        stats = _compute_stats(y_t_valid, y_p_valid)
        stats_text = (
            f"MAE: {stats['mae']:.3g}\n"
            f"RMSE: {stats['rmse']:.3g}\n"
            f"$R^2$: {stats['R2']:.3g}\n"
            f"Pearson $r^2$: {stats['pearson_r2']:.3g}\n"
            f"Spearman $\\rho^2$: {stats['spearman_rho2']:.3g}\n"
            f"Kendall $\\tau$: {stats['kendall_tau']:.3g}"
        )
        ax.text(0.02, 0.95, stats_text, ha="left", va="top", transform=ax.transAxes, fontsize=9)
        ax.grid(True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fpath = Path(save_dir) / f"parity_{ep.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(fpath, dpi=dpi)
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
    """Make per-endpoint parity figures across train/val/test and save them as PNG.

    Parameters:
        y_true_dict: {split: array (n_samples, n_endpoints)}
        y_pred_dict: {split: array (n_samples, n_endpoints)}
        mask_dict: {split: mask array}
        space: "log" or "linear"
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("science")
    except Exception:
        pass

    # nothing to precompute here; workers handle splits per endpoint
    # Run plotting sequentially or in parallel with progress bar
    if n_jobs is None or n_jobs <= 1:
        for j, ep in enumerate(tqdm(list(endpoints), desc="Parity plots")):
            _plot_parity_worker(j, ep, y_true_dict, y_pred_dict, mask_dict, space, save_dir, dpi)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _plot_parity_worker, j, ep, y_true_dict, y_pred_dict, mask_dict, space, save_dir, dpi
                )
                for j, ep in enumerate(endpoints)
            ]
            for fut in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc="Parity plots"
            ):
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
    """Plot grouped bar charts for R^2 and Spearman rho^2 across endpoints and splits.

    save_path_{r2,spr2}: file-like Path to save.
    """
    try:
        plt.style.use("science")
    except Exception:
        pass

    save_path_r2 = Path(save_path_r2)
    save_path_r2.parent.mkdir(parents=True, exist_ok=True)
    save_path_spr2 = Path(save_path_spr2)
    save_path_spr2.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    r2_vals = {s: [] for s in splits}
    spr2_vals = {s: [] for s in splits}

    for j, ep in enumerate(endpoints):
        for s in splits:
            y_t = y_true_dict[s][:, j]
            y_p = y_pred_dict[s][:, j]
            mask = mask_dict[s][:, j]
            valid = mask == 1
            if not np.any(valid):
                r2_vals[s].append(float("nan"))
                spr2_vals[s].append(float("nan"))
                continue
            y_t_v = y_t[valid]
            y_p_v = y_p[valid]
            if space == "linear" and ep != "LogD":
                y_t_v = np.power(10.0, y_t_v)
                y_p_v = np.power(10.0, y_p_v)
            try:
                mask_local = np.ones((y_t_v.shape[0], 1), dtype=int)
                mloc = eval_metrics.compute_metrics(
                    y_t_v.reshape(-1, 1), y_p_v.reshape(-1, 1), mask_local, endpoints=[ep]
                )
                r2_vals[s].append(mloc[ep].get("r2", float("nan")))
                spr2_vals[s].append(mloc[ep].get("spearman_rho2", float("nan")))
            except Exception:
                r2_vals[s].append(float("nan"))
                spr2_vals[s].append(float("nan"))

    # Convert lists to arrays for plotting
    labels = [ep.replace(" ", "\n") for ep in endpoints]
    x = np.arange(len(endpoints))
    width = 0.2

    def _save_r2():
        fig, ax = plt.subplots(figsize=(max(8, len(endpoints) * 0.6), 6), dpi=dpi)
        for i, s in enumerate(splits):
            vals = np.array(r2_vals[s], dtype=float)
            ax.bar(x + (i - 1) * width, vals, width, label=s)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(r"$R^2$")
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"$R^2$ by Endpoint ({space})")
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(save_path_r2, dpi=dpi)
        plt.close(fig)

    def _save_spr2():
        fig, ax = plt.subplots(figsize=(max(8, len(endpoints) * 0.6), 6), dpi=dpi)
        for i, s in enumerate(splits):
            vals = np.array(spr2_vals[s], dtype=float)
            # Replace NaN with 0 for plotting but we'll hatch bars that were NaN
            vals_plot = np.nan_to_num(vals, nan=0.0)
            bars = ax.bar(x + (i - 1) * width, vals_plot, width, label=s)
            for k, v in enumerate(vals):
                if np.isnan(v):
                    bars[k].set_hatch("//")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(r"Spearman $\rho^2$")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Spearman $\rho^2$ by Endpoint ({space})")
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(save_path_spr2, dpi=dpi)
        plt.close(fig)

    for fn in tqdm([_save_r2, _save_spr2], desc="Metric bars"):
        fn()
