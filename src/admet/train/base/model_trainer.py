"""Single‑dataset trainer abstractions."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from enum import Enum
import json
import logging
import multiprocessing

import numpy as np

from admet.data.load import LoadedDataset
from admet.model.base import BaseModel, ModelProtocol
from admet.utils import set_global_seeds
from admet.evaluate.metrics import AllMetrics, compute_metrics_log_and_linear
from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars

from .utils import _extract_features, _extract_targets, _target_mask

logger = logging.getLogger(__name__)


class FeaturizationMethod(str, Enum):
    """Enumeration of supported featurization methods.

    Using ``Enum`` ensures static type checkers treat members as instances of
    ``FeaturizationMethod`` instead of raw string literals, removing protocol
    / assignment errors in tests while keeping lightweight behavior (members
    still subclass ``str`` and compare equal to their raw values).
    """

    SMILES = "smiles"
    MORGAN_FP = "morgan_fp"
    NONE = "none"


@dataclass
class RunSummary:
    featurization: FeaturizationMethod
    endpoints: List[str]
    model: Any
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray
    mask_train: np.ndarray
    mask_val: np.ndarray
    mask_test: np.ndarray
    pred_train: np.ndarray
    pred_val: np.ndarray
    pred_test: np.ndarray


class BaseModelTrainer(ABC):
    def __init__(
        self,
        model_cls: Type[ModelProtocol],
        *,
        model_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        mixed_precision: bool = False,
    ) -> None:
        self.model_cls = model_cls
        self.model_params = model_params or {}
        self.seed = seed
        self.device = device
        self.mixed_precision = mixed_precision
        self.featurization: FeaturizationMethod = FeaturizationMethod.NONE
        self.model: Optional[BaseModel] = None

    def prepare_features(self, dataset: "LoadedDataset") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.featurization == FeaturizationMethod.MORGAN_FP:
            feature_cols = dataset.fingerprint_cols
        elif self.featurization == FeaturizationMethod.SMILES:
            feature_cols = [dataset.smiles_col]
        else:
            raise ValueError(f"Unsupported featurization method: {self.featurization}")
        if not feature_cols:
            raise ValueError("No feature columns found in dataset; cannot prepare features.")
        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing fingerprint columns in {split_name} split: {missing[:10]}")
        return (
            _extract_features(dataset.train, feature_cols),
            _extract_features(dataset.val, feature_cols),
            _extract_features(dataset.test, feature_cols),
        )

    def prepare_targets(self, dataset: "LoadedDataset") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        endpoints = dataset.endpoints
        if not endpoints:
            raise ValueError("No endpoints found in dataset; cannot prepare targets.")
        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in endpoints if c not in df.columns]
            if missing:
                raise ValueError(f"Missing endpoint columns in {split_name} split: {missing}")
        return (
            _extract_targets(dataset.train, endpoints),
            _extract_targets(dataset.val, endpoints),
            _extract_targets(dataset.test, endpoints),
        )

    def prepare_masks(self, Y_train, Y_val, Y_test):
        return _target_mask(Y_train), _target_mask(Y_val), _target_mask(Y_test)

    def build_sample_weights(
        self,
        dataset: "LoadedDataset",
        sample_weight_mapping: Optional[Dict[str, float]],
    ):
        if not sample_weight_mapping:
            return None
        if "Dataset" not in dataset.train.columns:
            raise ValueError("Training split missing 'Dataset' column needed for sample weight mapping")
        default = sample_weight_mapping.get("default", 1.0)
        sw_series = dataset.train["Dataset"].astype(str).map(lambda x: sample_weight_mapping.get(x, default))
        return sw_series.to_numpy(dtype=float)

    def build_model(self, endpoints: List[str]) -> BaseModel:
        return self.model_cls(endpoints, self.model_params, self.seed)  # type: ignore[call-arg]

    def compute_metrics(
        self,
        Y_train,
        Y_val,
        Y_test,
        pred_train,
        pred_val,
        pred_test,
        mask_train,
        mask_val,
        mask_test,
        endpoints: List[str],
    ) -> AllMetrics:
        return {
            "train": compute_metrics_log_and_linear(Y_train, pred_train, mask_train, endpoints),
            "validation": compute_metrics_log_and_linear(Y_val, pred_val, mask_val, endpoints),
            "test": compute_metrics_log_and_linear(Y_test, pred_test, mask_test, endpoints),
        }

    def save_artifacts(
        self,
        model: BaseModel,
        run_metrics: AllMetrics,
        output_dir: Path,
        summary: RunSummary,
        *,
        dataset: "LoadedDataset",
        extra_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir / "model"))
        (output_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2))
        n_cpus = multiprocessing.cpu_count()
        endpoints = summary.endpoints
        y_true = {"train": summary.Y_train, "validation": summary.Y_val, "test": summary.Y_test}
        y_pred = {"train": summary.pred_train, "validation": summary.pred_val, "test": summary.pred_test}
        y_mask = {"train": summary.mask_train, "validation": summary.mask_val, "test": summary.mask_test}
        fig_root = output_dir / "figures"
        for space in ["log", "linear"]:
            space_dir = fig_root / space
            space_dir.mkdir(parents=True, exist_ok=True)
            plot_parity_grid(
                y_true,
                y_pred,
                y_mask,
                endpoints,
                space=space,
                save_dir=space_dir,
                n_jobs=n_cpus,
            )
            plot_metric_bars(
                y_true,
                y_pred,
                y_mask,
                endpoints,
                space=space,
                save_path_r2=space_dir / "metrics_r2.png",
                save_path_spr2=space_dir / "metrics_spearman_rho2.png",
                n_jobs=n_cpus,
            )

    def fit(
        self,
        dataset: LoadedDataset,
        *,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        early_stopping_rounds: Optional[int] = None,
        output_dir: Optional[Path] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], Optional[RunSummary]]:
        set_global_seeds(self.seed)
        endpoints: List[str] = list(dataset.endpoints)
        if dry_run:
            return ({"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}}, None)
        if not endpoints:
            raise ValueError("No endpoints found in dataset; cannot train model.")
        if not (hasattr(dataset, "train") and hasattr(dataset, "val") and hasattr(dataset, "test")):
            raise ValueError("LoadedDataset must provide 'train','val','test' splits")
        X_train, X_val, X_test = self.prepare_features(dataset)
        Y_train, Y_val, Y_test = self.prepare_targets(dataset)
        mask_train, mask_val, mask_test = self.prepare_masks(Y_train, Y_val, Y_test)
        sample_weight = self.build_sample_weights(dataset, sample_weight_mapping)
        self.model = self.build_model(endpoints)
        self.model.fit(
            X_train,
            Y_train,
            Y_mask=mask_train,
            X_val=X_val,
            Y_val=Y_val,
            Y_val_mask=mask_val,
            sample_weight=sample_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)
        pred_test = self.model.predict(X_test)
        run_metrics = self.compute_metrics(
            Y_train,
            Y_val,
            Y_test,
            pred_train,
            pred_val,
            pred_test,
            mask_train,
            mask_val,
            mask_test,
            endpoints,
        )
        summary = RunSummary(
            featurization=self.featurization,
            model=self.model,
            endpoints=endpoints,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            Y_train=Y_train,
            Y_val=Y_val,
            Y_test=Y_test,
            mask_train=mask_train,
            mask_val=mask_val,
            mask_test=mask_test,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
        )
        if output_dir is not None:
            self.save_artifacts(
                self.model,
                run_metrics,
                output_dir,
                summary,
                dataset=dataset,
                extra_meta=extra_meta,
            )
        return run_metrics, summary


__all__ = ["BaseModelTrainer", "RunSummary", "FeaturizationMethod"]
