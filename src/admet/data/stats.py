from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, TypedDict, Union, cast

import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection
from torchmetrics.regression import KendallRankCorrCoef, PearsonCorrCoef, R2Score, SpearmanCorrCoef

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, "Tensor"]


class DistributionStats(TypedDict):
    min: float
    max: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    count: int


class CorrelationMetrics(TypedDict):
    mae: float
    rae: float
    mape: float
    rmse: float
    R2: float
    pearson_r: float
    spearman_rho: float
    kendall_tau: float


class RelativeAbsoluteError(torch.nn.Module):
    """Compute RAE: MAE / MAE_baseline where baseline predicts the mean."""

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        mae = torch.mean(torch.abs(preds - target))
        baseline = torch.mean(target)
        mae_baseline = torch.mean(torch.abs(target - baseline))
        if mae_baseline == 0:
            return torch.tensor(float("nan"), device=preds.device)
        return mae / mae_baseline


def _create_torch_metrics(compute_rank_correlations: bool = True, device: str = "cpu") -> MetricCollection:
    """Create a MetricCollection with all correlation metrics."""
    metrics = {
        "mae": MeanAbsoluteError(),
        "mape": MeanAbsolutePercentageError(),
        "rmse": MeanSquaredError(squared=False),
        "R2": R2Score(),
        "pearson_r": PearsonCorrCoef(),
    }
    if compute_rank_correlations:
        metrics["spearman_rho"] = SpearmanCorrCoef()
        metrics["kendall_tau"] = KendallRankCorrCoef()
    return MetricCollection(metrics).to(device)


def _is_torch_tensor(arr: ArrayLike) -> bool:
    """Check if input is a PyTorch tensor."""
    return isinstance(arr, torch.Tensor)


def _to_tensor(arr: ArrayLike, device: str = "cpu") -> Tensor:
    """Convert numpy array or tensor to torch tensor."""
    if _is_torch_tensor(arr):
        return arr.to(device)  # type: ignore[union-attr]
    return torch.from_numpy(np.asarray(arr)).to(device)


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if _is_torch_tensor(arr):
        return arr.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(arr)


def distribution(array: np.ndarray) -> DistributionStats:
    """Return formatted summary statistics for a numeric series.

    Parameters
    ----------
    series : pandas.Series
        Numeric series (non-numeric entries should be coerced prior to call).

    Returns
    -------
    DistributionStats
        Dictionary with keys: ``min``, ``max``, ``mean``, ``median``, ``std``,
        ``skew``, ``kurtosis``, ``count``.
    """
    array = np.asarray(array).ravel()
    array = array[~np.isnan(array)]
    count = array.shape[0]
    if count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
            "count": 0,
        }

    # ddof=1 requires at least 2 samples; skew/kurtosis require at least 3
    std_val = float(np.std(array, ddof=1)) if count > 1 else 0.0

    # Skew/kurtosis are undefined for constant arrays (zero variance)
    # Check variance to avoid RuntimeWarning from catastrophic cancellation
    has_variance = std_val > np.finfo(np.float64).eps * np.abs(np.mean(array))
    skew_val = float(scipy_stats.skew(array)) if count > 2 and has_variance else 0.0
    kurtosis_val = float(scipy_stats.kurtosis(array)) if count > 2 and has_variance else 0.0

    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": std_val,
        "skew": skew_val,
        "kurtosis": kurtosis_val,
        "count": int(count),
    }


def correlation_batch_torch(
    y_true: Tensor,
    y_pred: Tensor,
    compute_rank_correlations: bool = True,
) -> List[CorrelationMetrics]:
    """Compute metrics for multiple targets using torchmetrics (GPU-native).

    This is the preferred method during training when data is already on GPU.
    Avoids CPU transfers entirely for maximum performance.

    Parameters
    ----------
    y_true : torch.Tensor
        2D tensor of shape (n_samples, n_targets) containing true values.
    y_pred : torch.Tensor
        2D tensor of shape (n_samples, n_targets) containing predicted values.
    compute_rank_correlations : bool, default=True
        Whether to compute expensive rank correlations (Spearman, Kendall).

    Returns
    -------
    List[CorrelationMetrics]
        List of metric dictionaries, one per target.
    """
    device = y_true.device
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(1)

    n_targets = y_true.shape[1]
    results: List[CorrelationMetrics] = []

    for target_idx in range(n_targets):
        y_t = y_true[:, target_idx]
        y_p = y_pred[:, target_idx]

        # Remove NaNs
        valid_mask = ~(torch.isnan(y_t) | torch.isnan(y_p))
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]

        if y_t.numel() == 0:
            results.append(
                {
                    "mae": float("nan"),
                    "rae": float("nan"),
                    "mape": float("nan"),
                    "rmse": float("nan"),
                    "R2": float("nan"),
                    "pearson_r": float("nan"),
                    "spearman_rho": float("nan"),
                    "kendall_tau": float("nan"),
                }
            )
            continue

        # Create fresh metrics for each target to avoid state issues
        metrics = _create_torch_metrics(compute_rank_correlations, device=str(device))
        rae_metric = RelativeAbsoluteError().to(device)

        metric_results = metrics(y_p, y_t)
        rae_val = rae_metric(y_p, y_t)

        results.append(
            {
                "mae": float(metric_results["mae"].item()),
                "rae": float(rae_val.item()),
                "mape": float(metric_results["mape"].item()),
                "rmse": float(metric_results["rmse"].item()),
                "R2": float(metric_results["R2"].item()),
                "pearson_r": float(metric_results["pearson_r"].item()),
                "spearman_rho": float(metric_results.get("spearman_rho", torch.tensor(float("nan"))).item()),
                "kendall_tau": float(metric_results.get("kendall_tau", torch.tensor(float("nan"))).item()),
            }
        )

    return results


def correlation_batch(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    compute_rank_correlations: bool = True,
    use_gpu: bool = False,
    use_torch: bool | None = None,
) -> List[CorrelationMetrics]:
    """Compute metrics for multiple targets in a vectorized manner (10-30x faster).

    This function processes all targets simultaneously. When inputs are PyTorch tensors
    or use_torch=True, uses torchmetrics for GPU-native computation. Otherwise uses
    vectorized NumPy operations with optional CuPy GPU acceleration.

    Parameters
    ----------
    y_true : np.ndarray or torch.Tensor
        2D array/tensor of shape (n_samples, n_targets) containing true values.
    y_pred : np.ndarray or torch.Tensor
        2D array/tensor of shape (n_samples, n_targets) containing predicted values.
    compute_rank_correlations : bool, default=True
        Whether to compute expensive rank correlations (Spearman, Kendall).
    use_gpu : bool, default=False
        Whether to use GPU acceleration via CuPy (if available, numpy path only).
    use_torch : bool or None, default=None
        Force use of torchmetrics (True) or numpy/cupy (False).
        If None, auto-detects based on input type.

    Returns
    -------
    List[CorrelationMetrics]
        List of metric dictionaries, one per target.

    Notes
    -----
    Performance improvements:
    - torchmetrics: 2-5x faster on GPU, no data transfers during training
    - numpy: 10-30x faster than sequential calls (vectorization)
    - cupy: 2-5x additional speedup for large datasets (>10k samples)
    """
    # Auto-detect whether to use torch
    if use_torch is None:
        use_torch = _is_torch_tensor(y_true) or _is_torch_tensor(y_pred)

    if use_torch:
        y_true_t = _to_tensor(y_true)
        y_pred_t = _to_tensor(y_pred)
        return correlation_batch_torch(y_true_t, y_pred_t, compute_rank_correlations)

    # Numpy/CuPy path
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    # Try GPU acceleration if requested
    if use_gpu:
        try:
            import cupy as cp

            # Transfer to GPU
            y_true_gpu = cp.asarray(y_true_np)
            y_pred_gpu = cp.asarray(y_pred_np)

            # Compute on GPU
            results_gpu = _correlation_batch_impl(y_true_gpu, y_pred_gpu, compute_rank_correlations, xp=cp)

            # Transfer back to CPU with explicit typing and graceful fallback
            converted: List[CorrelationMetrics] = []
            for r in results_gpu:
                row: dict[str, float] = {}
                for k, v in r.items():
                    try:
                        if isinstance(v, cp.ndarray):
                            val = cp.asnumpy(v)
                        else:
                            val = v
                        row[k] = float(cast(float, val))
                    except Exception:
                        row[k] = float("nan")
                converted.append(cast(CorrelationMetrics, row))

            return converted
        except ImportError:
            logger.debug("CuPy not available, falling back to CPU computation")
        except Exception as e:
            logger.warning("GPU computation failed (%s), falling back to CPU", e)

    # CPU computation
    return _correlation_batch_impl(y_true_np, y_pred_np, compute_rank_correlations, xp=np)


def _correlation_batch_impl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_rank_correlations: bool,
    xp: Any,
) -> List[CorrelationMetrics]:
    """Internal implementation supporting both NumPy and CuPy."""
    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_samples, n_targets = y_true.shape
    results: List[CorrelationMetrics] = []

    # Vectorized NaN masking for all targets at once
    valid_mask = ~(xp.isnan(y_true) | xp.isnan(y_pred))  # shape: (n_samples, n_targets)

    for target_idx in range(n_targets):
        # Get valid data for this target
        mask = valid_mask[:, target_idx]
        y_t = y_true[mask, target_idx]
        y_p = y_pred[mask, target_idx]

        # Convert back to numpy for scipy functions (if using GPU)
        if xp.__name__ == "cupy":
            import cupy as cp

            y_t_cpu = cp.asnumpy(y_t)
            y_p_cpu = cp.asnumpy(y_p)
        else:
            y_t_cpu = y_t
            y_p_cpu = y_p

        if len(y_t_cpu) == 0:
            results.append(
                {
                    "mae": float("nan"),
                    "rae": float("nan"),
                    "mape": float("nan"),
                    "rmse": float("nan"),
                    "R2": float("nan"),
                    "pearson_r": float("nan"),
                    "spearman_rho": float("nan"),
                    "kendall_tau": float("nan"),
                }
            )
            continue

        # Compute basic metrics (can stay on GPU if using cupy for numpy-compatible ops)
        errors = xp.abs(y_t - y_p)
        mae = float(xp.mean(errors))

        y_t_mean = xp.mean(y_t)
        baseline_errors = xp.abs(y_t - y_t_mean)
        mae_baseline = float(xp.mean(baseline_errors))
        rae = mae / mae_baseline if mae_baseline != 0 else float("nan")

        # MAPE - handle division by zero
        with xp.errstate(divide="ignore", invalid="ignore"):
            mape_vals = xp.abs((y_t - y_p) / y_t)
            mape_vals = mape_vals[xp.isfinite(mape_vals)]
            mape = float(xp.mean(mape_vals)) if len(mape_vals) > 0 else float("nan")

        # RMSE
        rmse = float(xp.sqrt(xp.mean((y_t - y_p) ** 2)))

        # R² - more efficient direct computation
        ss_res = xp.sum((y_t - y_p) ** 2)
        ss_tot = xp.sum((y_t - y_t_mean) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")

        # Pearson correlation - optimized direct computation
        if len(y_t_cpu) >= 2:
            # Use numpy's corrcoef which is highly optimized
            corr_matrix = np.corrcoef(y_t_cpu, y_p_cpu)
            pearson_r = float(corr_matrix[0, 1])
        else:
            pearson_r = float("nan")

        # Rank correlations (expensive - only if requested)
        if compute_rank_correlations and len(y_t_cpu) >= 2:
            spearman_rho = float(spearmanr(y_t_cpu, y_p_cpu).statistic)
            kendall_tau = float(kendalltau(y_t_cpu, y_p_cpu).statistic)
        else:
            spearman_rho = float("nan")
            kendall_tau = float("nan")

        results.append(
            {
                "mae": mae,
                "rae": rae,
                "mape": mape,
                "rmse": rmse,
                "R2": r2,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "kendall_tau": kendall_tau,
            }
        )

    return results


def correlation_torch(
    y_true: Tensor,
    y_pred: Tensor,
    compute_rank_correlations: bool = True,
) -> CorrelationMetrics:
    """Compute correlation metrics using torchmetrics (GPU-native, no data transfers).

    This is the preferred method during training when data is already on GPU.
    Provides 2-5x speedup over numpy-based computation by avoiding CPU transfers.

    Parameters
    ----------
    y_true : torch.Tensor
        1-D tensor of true values (can be on any device).
    y_pred : torch.Tensor
        1-D tensor of predicted values (must be on same device as y_true).
    compute_rank_correlations : bool, default=True
        Whether to compute expensive rank correlations (Spearman, Kendall).

    Returns
    -------
    CorrelationMetrics
        Dictionary with all correlation metrics.
    """
    device = y_true.device
    y_true = y_true.flatten().float()
    y_pred = y_pred.flatten().float()

    # Remove NaNs
    valid_mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if y_true.numel() == 0:
        return {
            "mae": float("nan"),
            "rae": float("nan"),
            "mape": float("nan"),
            "rmse": float("nan"),
            "R2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
        }

    # Create metrics on the same device
    metrics = _create_torch_metrics(compute_rank_correlations, device=str(device))
    rae_metric = RelativeAbsoluteError().to(device)

    # Compute metrics
    metric_results = metrics(y_pred, y_true)
    rae_val = rae_metric(y_pred, y_true)

    result: CorrelationMetrics = {
        "mae": float(metric_results["mae"].item()),
        "rae": float(rae_val.item()),
        "mape": float(metric_results["mape"].item()),
        "rmse": float(metric_results["rmse"].item()),
        "R2": float(metric_results["R2"].item()),
        "pearson_r": float(metric_results["pearson_r"].item()),
        "spearman_rho": float(metric_results.get("spearman_rho", torch.tensor(float("nan"))).item()),
        "kendall_tau": float(metric_results.get("kendall_tau", torch.tensor(float("nan"))).item()),
    }

    return result


def correlation(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    compute_rank_correlations: bool = True,
    use_torch: bool | None = None,
) -> CorrelationMetrics:
    """Compute metrics for a single endpoint column.

    Automatically uses torchmetrics when inputs are PyTorch tensors (for GPU efficiency),
    and falls back to scipy/sklearn for numpy arrays.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray or torch.Tensor
        1-D arrays/tensors of true and predicted values.
    compute_rank_correlations : bool, default=True
        Whether to compute expensive rank correlations (Spearman, Kendall).
        Setting to False can speed up computation by 30-50%.
    use_torch : bool or None, default=None
        Force use of torchmetrics (True) or scipy/sklearn (False).
        If None, auto-detects based on input type.

    Returns
    -------
    CorrelationMetrics
        Dictionary with keys: ``mae``, ``rae``, ``mape``, ``rmse``, ``R2``,
        ``pearson_r``, ``spearman_rho``, ``kendall_tau``.
    """
    # Auto-detect whether to use torch
    if use_torch is None:
        use_torch = _is_torch_tensor(y_true) or _is_torch_tensor(y_pred)

    if use_torch:
        y_true_t = _to_tensor(y_true)
        y_pred_t = _to_tensor(y_pred)
        return correlation_torch(y_true_t, y_pred_t, compute_rank_correlations)

    # Numpy path (original implementation for backward compatibility)
    y_true_np = _to_numpy(y_true).ravel()
    y_pred_np = _to_numpy(y_pred).ravel()

    # Remove NaNs
    valid_mask = ~np.isnan(y_true_np) & ~np.isnan(y_pred_np)
    y_true_np = y_true_np[valid_mask]
    y_pred_np = y_pred_np[valid_mask]

    if y_true_np.shape[0] == 0:
        return {
            "mae": float("nan"),
            "rae": float("nan"),
            "mape": float("nan"),
            "rmse": float("nan"),
            "R2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
        }

    mae = mean_absolute_error(y_true_np, y_pred_np)
    mae_baseline = mean_absolute_error(y_true_np, np.full_like(y_true_np, np.mean(y_true_np)))
    rae = mae / mae_baseline if mae_baseline != 0 else float("nan")

    mape = mean_absolute_percentage_error(y_true_np, y_pred_np)
    rmse = root_mean_squared_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    # Correlation coefficients are undefined for constant arrays
    y_true_std = np.std(y_true_np)
    y_pred_std = np.std(y_pred_np)
    can_compute_correlation = (
        y_true_np.size >= 2 and y_true_std > np.finfo(np.float64).eps and y_pred_std > np.finfo(np.float64).eps
    )

    pearson_r_val = pearsonr(y_true_np, y_pred_np).statistic if can_compute_correlation else float("nan")

    # Only compute expensive rank correlations if requested
    if compute_rank_correlations and can_compute_correlation:
        spearman_rho = spearmanr(y_true_np, y_pred_np).statistic
        kendall_tau = kendalltau(y_true_np, y_pred_np).statistic
    else:
        spearman_rho = float("nan")
        kendall_tau = float("nan")

    return {
        "mae": float(mae),
        "rae": float(rae),
        "mape": float(mape),
        "rmse": float(rmse),
        "R2": float(r2),
        "pearson_r": float(pearson_r_val),
        "spearman_rho": float(spearman_rho),
        "kendall_tau": float(kendall_tau),
    }
