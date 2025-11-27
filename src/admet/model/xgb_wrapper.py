"""XGBoost per-endpoint multi-output wrapper implementation.

Provides a thin abstraction that trains one ``XGBRegressor`` per endpoint,
handling missing labels via a mask and persisting individual model JSON files.

Contents
--------
Classes
    XGBoostMultiEndpoint : Train & infer per-endpoint models with masking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm  # type: ignore[import-not-found]
from xgboost import XGBRegressor

from admet.data.fingerprinting import DEFAULT_FINGERPRINT_CONFIG, FingerprintConfig

from .base import BaseModel


class XGBoostMultiEndpoint(BaseModel):
    """Train one XGBoost regressor per endpoint with missing-value masking.

    Each endpoint model is fit only on rows where its target is present. During
    inference all endpoint predictions are generated; endpoints lacking any
    training data return ``NaN`` vectors.

    Parameters
    ----------
    endpoints : Sequence[str]
        Ordered list of endpoint names.
    model_params : dict, optional
        Hyperparameters passed to each ``XGBRegressor`` instance.
    random_state : int, optional
        Seed for model reproducibility.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        model_params: Optional[Dict[str, object]] = None,
        random_state: Optional[int] = 123,
    ) -> None:
        # Explicit attribute type annotations to satisfy ModelProtocol structural typing.
        self.endpoints: Sequence[str]
        self.input_type: str
        self.endpoints = list(endpoints)
        self.input_type = "fingerprint"
        self.model_params = model_params or {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        }
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.n_features_: Optional[int] = None
        self.fingerprint_config: Optional[FingerprintConfig] = DEFAULT_FINGERPRINT_CONFIG
        # Dict of endpoint -> model (None if no training data for that endpoint)
        self.models = {}  # type: Dict[str, Optional[XGBRegressor]]

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        Y_mask: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        Y_val_mask: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        early_stopping_rounds: int | None = 50,
    ) -> None:
        """Fit one model per endpoint.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (n_samples, n_features).
        Y_train : np.ndarray
            Training target matrix (n_samples, n_endpoints).
        Y_mask : np.ndarray
            Mask indicating valid target entries (same shape as ``Y_train``).
        X_val, Y_val, Y_val_mask : np.ndarray, optional
            Validation data for early stopping / evaluation; if provided each
            endpoint model uses only rows with valid targets.
        sample_weight : np.ndarray, optional
            Per-row sample weights applied to all endpoints.
        early_stopping_rounds : int, optional
            Early stopping rounds forwarded to XGBoost.
        """
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D numpy array")
        if Y_train.ndim != 2:
            raise ValueError("Y_train must be a 2D numpy array")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have the same number of rows")
        # record feature count for inference time validation
        self.n_features_ = X_train.shape[1]
        _, n_endpoints = Y_train.shape
        if n_endpoints != len(self.endpoints):
            raise ValueError(
                "Mismatch in endpoints: "
                f"model expects {len(self.endpoints)} endpoints, "
                f"but Y_train has {n_endpoints} columns (shape={Y_train.shape})."
            )
        self.logger.info("Training XGBoost models for %d endpoints", n_endpoints)
        for j, ep in enumerate(tqdm(self.endpoints, desc="Training endpoints")):
            # Accept both boolean and integer masks
            mask_j = Y_mask[:, j].astype(bool)
            if not np.any(mask_j):
                # No data for this endpoint; skip but create placeholder model (None)
                self.models[ep] = None  # type: ignore
                continue
            X_tr_ep = X_train[mask_j]
            y_tr_ep = Y_train[mask_j, j]
            sw_ep = sample_weight[mask_j] if sample_weight is not None else None
            eval_set = None
            eval_sample_weight = None
            if X_val is not None and Y_val is not None and Y_val_mask is not None:
                mask_val_j = Y_val_mask[:, j].astype(bool)
                if np.any(mask_val_j):
                    eval_set = [(X_val[mask_val_j], Y_val[mask_val_j, j])]
                    if sample_weight is not None:
                        eval_sample_weight = [sample_weight[mask_val_j]]
            params = {
                **self.model_params,
                "random_state": self.random_state,
                "early_stopping_rounds": early_stopping_rounds,
            }
            model = XGBRegressor(**params)
            fit_kwargs = {
                "sample_weight": sw_ep,
                "eval_set": eval_set,
                "verbose": False,
            }
            if eval_sample_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = eval_sample_weight
            try:
                model.fit(X_tr_ep, y_tr_ep, **fit_kwargs)
            except Exception as exc:  # pragma: no cover - fallback when GPU not available  # noqa: BLE001
                self.logger.warning("Initial model.fit failed (likely GPU issue), retrying with CPU params: %s", exc)
                cpu_params = {k: v for k, v in params.items() if k not in ("device", "device_id")}
                cpu_params["tree_method"] = "hist"
                model = XGBRegressor(**cpu_params)
                model.fit(X_tr_ep, y_tr_ep, **fit_kwargs)
            self.models[ep] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all endpoints for provided feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Prediction matrix shape ``(n_samples, n_endpoints)`` with ``NaN``
            for endpoints that lacked training data.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D numpy array for prediction")
        if self.n_features_ is None:
            # model was not fit
            raise ValueError("Model has not been fit yet; call `.fit` before `predict`.")
        if X.shape[1] != self.n_features_:
            raise ValueError("Input feature dimension mismatch with trained model")
        preds: List[np.ndarray] = []
        for ep in self.endpoints:
            mdl = self.models.get(ep)
            if mdl is None:
                preds.append(np.full(X.shape[0], np.nan))
            else:
                preds.append(mdl.predict(X))
        return np.vstack(preds).T

    def save(self, path: str) -> None:
        """Persist configuration and trained endpoint models.

        Parameters
        ----------
        path : str
            Output directory path; created if missing.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save each endpoint model
        meta = {
            "endpoints": self.endpoints,
            "model_params": self.model_params,
            "random_state": self.random_state,
            "n_features": self.n_features_,
        }
        fp_cfg = getattr(self, "fingerprint_config", None)
        if fp_cfg is not None:
            if isinstance(fp_cfg, FingerprintConfig):
                meta["fingerprint"] = fp_cfg.to_dict()
            elif isinstance(fp_cfg, dict):
                meta["fingerprint"] = fp_cfg
        (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
        for ep, mdl in self.models.items():
            if mdl is None:
                continue
            mdl.save_model(str(out_dir / f"{ep}.json"))

    @classmethod
    def load(cls, path: str) -> "XGBoostMultiEndpoint":
        """Load a previously saved endpoint model collection.

        Parameters
        ----------
        path : str
            Directory containing ``config.json`` and per-endpoint JSON files.

        Returns
        -------
        XGBoostMultiEndpoint
            Reconstructed multi-endpoint model.
        """
        in_dir = Path(path)
        meta = json.loads((in_dir / "config.json").read_text())
        obj = cls(
            endpoints=meta["endpoints"],
            model_params=meta["model_params"],
            random_state=meta["random_state"],
        )
        # restore recorded feature dimension
        obj.n_features_ = meta.get("n_features")
        fp_cfg_raw = meta.get("fingerprint")
        if fp_cfg_raw is not None:
            obj.fingerprint_config = FingerprintConfig.from_mapping(fp_cfg_raw, default=DEFAULT_FINGERPRINT_CONFIG)
        else:
            obj.fingerprint_config = DEFAULT_FINGERPRINT_CONFIG
        for ep in obj.endpoints:
            ep_path = in_dir / f"{ep}.json"
            if not ep_path.exists():
                obj.models[ep] = None
                continue
            mdl = XGBRegressor()
            mdl.load_model(str(ep_path))
            obj.models[ep] = mdl
        return obj

    def get_config(self) -> Dict[str, object]:  # pragma: no cover - trivial
        """Return high-level configuration dictionary."""
        return {
            "type": "xgboost_multi_endpoint",
            "model_params": self.model_params,
            "endpoints": self.endpoints,
        }

    def get_metadata(self) -> Dict[str, object]:  # pragma: no cover - trivial
        """Return lightweight metadata (counts & seeds)."""
        return {
            "random_state": self.random_state,
            "n_models": len([m for m in self.models.values() if m is not None]),
        }


__all__ = ["XGBoostMultiEndpoint"]
