from typing import Dict, List
import numpy as np
from sklearn.metrics import mean_squared_error


def per_task_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute RMSE for each task independently."""
    n_tasks = y_true.shape[1]
    rmses = []
    for t in range(n_tasks):
        rmse_t = mean_squared_error(y_true[:, t], y_pred[:, t], squared=False)
        rmses.append(rmse_t)
    return np.array(rmses)


def aggregate_metric(per_task: np.ndarray, task_weights: List[float]) -> float:
    """Weighted sum of per-task metrics."""
    w = np.array(task_weights, dtype=float)
    w = w / w.sum()
    return float(np.sum(w * per_task))


def build_metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_weights: List[float],
    prefix: str = "val",
) -> Dict[str, float]:
    """Build a dict of per-task and aggregated metrics."""
    rmse_tasks = per_task_rmse(y_true, y_pred)
    agg = aggregate_metric(rmse_tasks, task_weights)

    metrics = {f"{prefix}_rmse_task_{i}": float(rmse) for i, rmse in enumerate(rmse_tasks)}
    metrics[f"{prefix}_rmse_weighted"] = agg
    return metrics
