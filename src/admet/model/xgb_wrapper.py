"""XGBoost per-endpoint multi-output wrapper implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional
import logging
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor

from .base import BaseModel


class XGBoostMultiEndpoint(BaseModel):
    """Train one XGBRegressor per endpoint, handling missing values via masks.

    Each endpoint model is fit on rows where its target is present. Predictions
    are generated for all compounds (rows) for all endpoints.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        model_params: Optional[Dict[str, object]] = None,
        random_state: Optional[int] = 123,
    ) -> None:
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
        _, n_endpoints = Y_train.shape
        assert n_endpoints == len(self.endpoints)
        self.logger.info(f"Training XGBoost models for {n_endpoints} endpoints")
        for j, ep in enumerate(tqdm(self.endpoints, desc="Training endpoints")):
            mask_j = Y_mask[:, j] == 1
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
                mask_val_j = Y_val_mask[:, j] == 1
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
            model.fit(X_tr_ep, y_tr_ep, **fit_kwargs)
            self.models[ep] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds: List[np.ndarray] = []
        for ep in self.endpoints:
            mdl = self.models.get(ep)
            if mdl is None:
                preds.append(np.full(X.shape[0], np.nan))
            else:
                preds.append(mdl.predict(X))
        return np.vstack(preds).T

    def save(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save each endpoint model
        meta = {
            "endpoints": self.endpoints,
            "model_params": self.model_params,
            "random_state": self.random_state,
        }
        (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
        for ep, mdl in self.models.items():
            if mdl is None:
                continue
            mdl.save_model(str(out_dir / f"{ep}.json"))

    @classmethod
    def load(cls, path: str) -> "XGBoostMultiEndpoint":
        in_dir = Path(path)
        meta = json.loads((in_dir / "config.json").read_text())
        obj = cls(
            endpoints=meta["endpoints"],
            model_params=meta["model_params"],
            random_state=meta["random_state"],
        )
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
        return {
            "type": "xgboost_multi_endpoint",
            "model_params": self.model_params,
            "endpoints": self.endpoints,
        }

    def get_metadata(self) -> Dict[str, object]:  # pragma: no cover - trivial
        return {
            "random_state": self.random_state,
            "n_models": len([m for m in self.models.values() if m is not None]),
        }


__all__ = ["XGBoostMultiEndpoint"]
