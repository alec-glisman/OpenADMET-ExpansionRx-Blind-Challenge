"""Training orchestration for XGBoost multi-endpoint model (initial version)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Optional
import logging
import numpy as np

from admet.data.load import LoadedDataset
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.utils import set_global_seeds
from admet.evaluate.metrics import compute_metrics_log_and_linear

logger = logging.getLogger(__name__)


def _extract_features(df, fingerprint_cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, fingerprint_cols].to_numpy(dtype=float)


def _extract_targets(df, endpoints: Sequence[str]) -> np.ndarray:
    return df[endpoints].to_numpy(dtype=float)


def _target_mask(y: np.ndarray) -> np.ndarray:
    return (~np.isnan(y)).astype(int)


def train_xgb_models(
    dataset: LoadedDataset,
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Train XGBoost models per endpoint and compute metrics.

    Returns metrics dict keyed by split (train/val/test).
    """
    # Seed global RNGs early for reproducibility
    set_global_seeds(seed)
    endpoints = dataset.endpoints
    fp_cols = dataset.fingerprint_cols
    # Prepare data
    X_train = _extract_features(dataset.train, fp_cols)
    X_val = _extract_features(dataset.val, fp_cols)
    X_test = _extract_features(dataset.test, fp_cols)
    Y_train = _extract_targets(dataset.train, endpoints)
    Y_val = _extract_targets(dataset.val, endpoints)
    Y_test = _extract_targets(dataset.test, endpoints)
    mask_train = _target_mask(Y_train)
    mask_val = _target_mask(Y_val)
    mask_test = _target_mask(Y_test)

    # Sample weights
    sample_weight = None
    if sample_weight_mapping:
        sw = []
        for ds_name in dataset.train["Dataset"].astype(str):
            sw.append(sample_weight_mapping.get(ds_name, sample_weight_mapping.get("default", 1.0)))
        sample_weight = np.array(sw, dtype=float)

    model = XGBoostMultiEndpoint(endpoints=endpoints, model_params=model_params, random_state=seed)
    model.fit(
        X_train,
        Y_train,
        Y_mask=mask_train,
        X_val=X_val,
        Y_val=Y_val,
        Y_val_mask=mask_val,
        sample_weight=sample_weight,
        early_stopping_rounds=early_stopping_rounds,
    )
    # Predictions
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics = {
        "train": compute_metrics_log_and_linear(Y_train, pred_train, mask_train, endpoints),
        "val": compute_metrics_log_and_linear(Y_val, pred_val, mask_val, endpoints),
        "test": compute_metrics_log_and_linear(Y_test, pred_test, mask_test, endpoints),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir / "model"))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


__all__ = ["train_xgb_models"]
