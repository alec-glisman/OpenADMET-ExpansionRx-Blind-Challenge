"""Single‑dataset trainer abstractions."""

from __future__ import annotations

import json
import logging
import multiprocessing
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from admet.data.fingerprinting import DEFAULT_FINGERPRINT_CONFIG, FingerprintConfig, MorganFingerprintGenerator
from admet.data.load import LoadedDataset
from admet.evaluate.metrics import AllMetrics, compute_metrics_log_and_linear
from admet.model.base import BaseModel, ModelProtocol
from admet.utils import get_git_commit_hash, set_global_seeds
from admet.visualize.model_performance import plot_metric_bars, plot_parity_grid

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
    fingerprint_config: FingerprintConfig
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
        fingerprint_config: Optional[FingerprintConfig] = None,
    ) -> None:
        self.model_cls = model_cls
        self.model_params = model_params or {}
        self.seed = seed
        self.device = device
        self.mixed_precision = mixed_precision
        self.featurization: FeaturizationMethod = FeaturizationMethod.NONE
        self.model: Optional[BaseModel] = None
        self.git_commit: Optional[str] = None
        self.fingerprint_config = fingerprint_config or DEFAULT_FINGERPRINT_CONFIG
        self._fingerprint_generator: Optional[MorganFingerprintGenerator] = None

    def prepare_features(self, dataset: "LoadedDataset") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices for train/validation/test splits.

        For fingerprint featurization, this delegates to ``_extract_features`` which
        returns ``float32`` NumPy arrays without unnecessary copies.

        For SMILES featurization, this returns the configured SMILES column as
        an ``object``-dtype 2D array of shape ``(n_samples, 1)`` for each split.
        Keeping SMILES as objects prevents pandas from attempting a lossy string
        -> float conversion while still satisfying the expected shapes in
        unit tests and downstream featurizers.
        """

        if self.featurization == FeaturizationMethod.MORGAN_FP:
            feature_cols = list(dataset.fingerprint_cols)
        elif self.featurization == FeaturizationMethod.SMILES:
            feature_cols = [dataset.smiles_col]
        else:
            raise ValueError(f"Unsupported featurization method: {self.featurization}")

        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in feature_cols if c not in df.columns]
            if feature_cols and missing:
                raise ValueError(f"Missing fingerprint columns in {split_name} split: {missing[:10]}")

        # Numeric fingerprint features use the optimized float32 extractor when precomputed.
        if self.featurization == FeaturizationMethod.MORGAN_FP and feature_cols:
            return (
                _extract_features(dataset.train, feature_cols),
                _extract_features(dataset.val, feature_cols),
                _extract_features(dataset.test, feature_cols),
            )

        if self.featurization == FeaturizationMethod.MORGAN_FP and not feature_cols:
            if dataset.smiles_col not in dataset.train.columns:
                raise ValueError("SMILES column missing; cannot generate fingerprints on the fly.")
            fp_cfg = getattr(dataset, "fingerprint_config", None) or self.fingerprint_config
            if self._fingerprint_generator is None or getattr(self._fingerprint_generator, "config", None) != fp_cfg:
                self._fingerprint_generator = MorganFingerprintGenerator(
                    radius=fp_cfg.radius,
                    count_simulation=fp_cfg.use_counts,
                    include_chirality=fp_cfg.include_chirality,
                    fp_size=fp_cfg.n_bits,
                    config=fp_cfg,
                )

            def _featurize(df: "pd.DataFrame") -> np.ndarray:  # type: ignore[name-defined]
                if dataset.smiles_col not in df.columns:
                    raise ValueError("SMILES column missing from split; cannot featurize.")
                fps = self._fingerprint_generator.calculate_fingerprints(df[dataset.smiles_col])
                return fps.to_numpy(dtype=np.float32, copy=False)

            return (
                _featurize(dataset.train),
                _featurize(dataset.val),
                _featurize(dataset.test),
            )

        # SMILES featurization keeps raw SMILES strings as object arrays.
        # Each split is returned as shape (n_samples, 1) to match tests and
        # provide a consistent interface to downstream featurizers.
        if self.featurization == FeaturizationMethod.SMILES:
            X_train = dataset.train[feature_cols].to_numpy(dtype=object)
            X_val = dataset.val[feature_cols].to_numpy(dtype=object)
            X_test = dataset.test[feature_cols].to_numpy(dtype=object)
            return X_train, X_val, X_test

        # The above branches should be exhaustive, but keep mypy satisfied.
        raise ValueError(f"Unsupported featurization method: {self.featurization}")

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

    def prepare_masks(
        self,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        Y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build presence masks (1 if target present else 0) for each split.

        Parameters
        ----------
        Y_train, Y_val, Y_test : numpy.ndarray
            2-D target arrays (``(N_samples, N_endpoints)``) for each split.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of ``(mask_train, mask_val, mask_test)`` with ``int`` dtype.
        """
        return (
            _target_mask(Y_train),
            _target_mask(Y_val),
            _target_mask(Y_test),
        )

    def build_sample_weights(
        self,
        dataset: "LoadedDataset",
        sample_weight_mapping: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        """Return per-sample weights for the training split.

        Parameters
        ----------
        dataset : LoadedDataset
            Loaded dataset object containing a training split with a ``Dataset`` column.
        sample_weight_mapping : dict[str, float] | None
            Mapping of dataset label -> weight. Special key ``"default"`` is used when
            a label is missing.

        Returns
        -------
        numpy.ndarray | None
            1-D float array of length ``len(dataset.train)`` or ``None`` if mapping absent.
        """
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

        git_commit = self.git_commit or get_git_commit_hash()
        if git_commit:
            logger.info("Saving artifacts for git commit %s", git_commit)

        # Standardized run metadata used by downstream evaluation/ensembling.
        run_meta = {
            "model_type": type(model).__name__,
            "endpoints": summary.endpoints,
            "featurization": summary.featurization.value,
            "fingerprint": (
                summary.fingerprint_config.to_dict() if summary.featurization == FeaturizationMethod.MORGAN_FP else None
            ),
            "model_path": "model",
            "seed": self.seed,
            "extra_meta": extra_meta or {},
            "git_commit": git_commit,
        }
        (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
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
    ) -> Tuple[AllMetrics, Optional[RunSummary]]:
        set_global_seeds(self.seed)
        if self.git_commit is None:
            self.git_commit = get_git_commit_hash()
            if self.git_commit:
                logger.info("Starting training with git commit %s", self.git_commit)
            else:
                logger.debug("Git commit hash unavailable; proceeding without commit metadata.")
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
        try:
            setattr(self.model, "fingerprint_config", self.fingerprint_config)
        except Exception:  # pragma: no cover - defensive attribute set
            logger.debug("Model does not accept fingerprint_config attribute; continuing without it.")
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
            fingerprint_config=self.fingerprint_config,
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
