"""Shared helpers for MLflow logging."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import mlflow
import numpy as np


def _clean_key(part: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", str(part))


def flatten_params(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config objects into a dict suitable for mlflow.log_params."""

    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_part = _clean_key(key)
            new_prefix = f"{prefix}.{key_part}" if prefix else key_part
            flat.update(flatten_params(value, new_prefix))
    elif isinstance(obj, (list, tuple)):
        if not obj and prefix:
            flat[prefix] = "[]"
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flat.update(flatten_params(value, new_prefix))
    else:
        if prefix:
            flat[prefix] = "null" if obj is None else obj
    return flat


def _coerce_metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        numeric = float(value)
    except Exception:  # noqa: BLE001
        return None
    return float(numeric) if math.isfinite(float(numeric)) else None


def flatten_metrics(
    run_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], *, prefix: str = "metrics"
) -> Dict[str, float]:
    """Flatten nested metric structure produced by trainers."""

    flat: Dict[str, float] = {}
    for split, endpoints in run_metrics.items():
        for endpoint, spaces in endpoints.items():
            if not isinstance(spaces, dict):
                continue
            endpoint_key = _clean_key(endpoint)
            for space_name, metrics in spaces.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, metric_value in metrics.items():
                    numeric_value = _coerce_metric(metric_value)
                    if numeric_value is None:
                        continue
                    key_parts = [prefix, split, endpoint_key, space_name, metric_name]
                    key = ".".join(_clean_key(part) for part in key_parts if part)
                    flat[key] = numeric_value
    return flat


def set_mlflow_tracking(tracking_uri: Optional[str], experiment_name: Optional[str]) -> None:
    """Set MLflow tracking URI and experiment name.

    Args:
        tracking_uri: MLflow tracking URI (e.g., file:///path/to/mlruns)
        experiment_name: Name of the MLflow experiment
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)


__all__ = ["flatten_params", "flatten_metrics", "set_mlflow_tracking"]
