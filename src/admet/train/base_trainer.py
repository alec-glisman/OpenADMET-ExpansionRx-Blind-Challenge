"""admet.train.base_trainer
===========================

Core abstractions for single‑dataset and Ray‑based multi‑dataset training.

Overview
--------
Two layered interfaces are provided:

``BaseModelTrainer``
    Implements canonical train → predict → evaluate → save flow for a
    single :class:`admet.data.load.LoadedDataset` using backend‑specific
    hooks (feature/target extraction, model construction, metrics, artifacts).

``BaseRayMultiDatasetTrainer``
    Discovers multiple HF dataset directories and schedules remote training
    tasks via Ray using a provided concrete trainer class.

Training Flow (Single Dataset)
------------------------------
The orchestration implemented by :meth:`BaseModelTrainer.run` follows this
sequence:

1. Seed global RNGs.
2. Prepare feature matrices (``prepare_features``).
3. Prepare target matrices (``prepare_targets``) + masks (``prepare_masks``).
4. Optional sample weight vector (``build_sample_weights``).
5. Instantiate backend model (``build_model``) & train (``model.fit``).
6. Generate predictions for train/validation/test.
7. Compute metrics (``compute_metrics``).
8. Persist artifacts (``save_artifacts``) if an output directory is supplied.

Remote Payload Schema
---------------------
Each Ray worker returns ``(rel_key, payload)`` where ``payload`` contains:

``metrics``
    Nested split → metric dict (or ``None`` on failure).
``meta``
    Split/path metadata inferred from directory layout.
``status``
    ``'ok'``, ``'partial'``, ``'error'``, or ``'timeout'``.
``start_time`` / ``end_time`` / ``duration_seconds``
    Timing information in ISO format and seconds.

Artifacts
---------
Concrete trainers typically write (under the supplied output directory):
``model/`` (backend‑specific serialization), ``metrics.json``, ``figures/``
containing plots, plus optional additional metadata (e.g. summary CSV for
multi‑dataset runs).
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import datetime
import logging
import multiprocessing

import json
import numpy as np
import ray
import pandas as pd

from admet.data.load import LoadedDataset
from admet.model.base import BaseModel, ModelProtocol
from admet.utils import set_global_seeds
from admet.evaluate.metrics import AllMetrics, compute_metrics_log_and_linear
from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars

logger = logging.getLogger(__name__)


def _extract_features(df, fingerprint_cols: Sequence[str]) -> np.ndarray:
    """Extract fingerprint features from a dataframe split.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data split containing fingerprint columns.
    fingerprint_cols : Sequence[str]
        Names of fingerprint columns to select.

    Returns
    -------
    numpy.ndarray
        2-D array of fingerprint features with dtype float.
    """
    return df.loc[:, fingerprint_cols].to_numpy(dtype=float)


def _extract_targets(df, endpoints: Sequence[str]) -> np.ndarray:
    """Extract endpoint target columns from a split.

    Parameters
    ----------
    df : pandas.DataFrame
        Input split containing endpoint columns.
    endpoints : Sequence[str]
        Names of endpoint columns to select.

    Returns
    -------
    numpy.ndarray
        2-D float array with targets and NaNs where missing.
    """
    return df[endpoints].to_numpy(dtype=float)


def _target_mask(y: np.ndarray) -> np.ndarray:
    """Build a binary mask indicating presence of target values.

    Parameters
    ----------
    y : numpy.ndarray
        Target matrix with NaNs used to signal missing values.

    Returns
    -------
    numpy.ndarray
        Integer mask array where 1 indicates a present target and 0 indicates
        a missing target.
    """
    # Return a boolean mask for presence of targets; callers can convert to int if needed
    return ~np.isnan(y)


def infer_split_metadata(hf_path: Path, root: Path) -> Dict[str, object]:
    """Infer dataset layout metadata from an HF dataset path.

    The function looks for common directory layouts under ``root`` and
    returns a mapping that includes the relative path plus best-effort
    cluster/split/fold and quality information.

    Parameters
    ----------
    hf_path : pathlib.Path
        Absolute or relative path pointing to a ``hf_dataset`` directory.
    root : pathlib.Path
        Directory used as a reference to calculate relative paths.

    Returns
    -------
    dict
        A dictionary containing keys such as ``relative_path``, ``full_path``,
        ``cluster_method``, ``cluster``, ``quality``, ``split`` and ``fold``
        when these can be inferred.
    """
    try:
        rel = hf_path.relative_to(root)
        rel_parts: List[str] = [p for p in rel.parts if p]
        meta: Dict[str, object] = {
            "relative_path": str(rel),
            "absolute_path": str(hf_path.resolve()),
        }
    except Exception as e:
        logger.warning("Failed to infer relative path for %s: %s", hf_path, e)
        rel = None
        rel_parts = []
        meta = {"relative_path": str(hf_path)}

    try:
        full_path = str(hf_path.resolve())
    except Exception as e:
        logger.warning("Failed to resolve full path for %s: %s", hf_path, e)
        full_path = str(hf_path)
    full_parts: List[str] = [p for p in Path(full_path).parts if p and p != "/"]
    meta["full_path"] = full_path

    for part in rel_parts:
        if part.startswith("split_"):
            meta["split"] = part.replace("split_", "")
        elif part.startswith("fold_"):
            meta["fold"] = part.replace("fold_", "")

    # Prefer cluster extraction from relative parts, fallback to full path parts
    if len(full_parts) > 5:
        cluster_method = full_parts[-4]
        quality = full_parts[-5]
        meta["quality"] = quality
        meta["cluster"] = f"{quality}/{cluster_method}"

    # Logging to help debug any future layout mismatches
    logger.info("Inferred metadata for %s: %s", rel, meta)
    return meta


@dataclass
class SplitMetadata:
    """Typed representation of dataset split metadata.

    This dataclass mirrors the keys returned by :func:`infer_split_metadata`
    and is provided to help avoid dictionary-key typos in user code.
    """

    relative_path: str
    absolute_path: str
    full_path: str
    quality: Optional[str] = None
    cluster: Optional[str] = None
    split: Optional[str] = None
    fold: Optional[str] = None


def metadata_from_dict(d: Dict[str, object]) -> SplitMetadata:
    def _opt_str(k: str) -> Optional[str]:
        v = d.get(k)
        return str(v) if v is not None else None

    return SplitMetadata(
        relative_path=str(d.get("relative_path", "")),
        absolute_path=str(d.get("absolute_path", "")),
        full_path=str(d.get("full_path", "")),
        quality=_opt_str("quality"),
        cluster=_opt_str("cluster"),
        split=_opt_str("split"),
        fold=_opt_str("fold"),
    )


class FeaturizationMethod(Enum):
    """Simple enum to track dataset featurization method.

    Two basic methods are supported at present:
    - SMILES: raw SMILES string column(s) to be used by a backend that
        consumes SMILES directly (for example a graph-based model loader).
    - MORGAN_FP: precomputed Morgan fingerprint columns (e.g. 'fp_0',
        'fp_1', ...) holding numeric fingerprint bits.
    """

    SMILES = "smiles"
    MORGAN_FP = "morgan_fp"
    NONE = None


@dataclass
class RunOutputs:
    """Container for arrays computed during a single run.

    Attributes
    ----------
    endpoints: list of str
        Endpoint names for the output columns.
    X_train, X_val, X_test: numpy arrays
    Y_train, Y_val, Y_test: numpy arrays with NaNs for missing targets
    mask_train, mask_val, mask_test: numpy arrays (bool or int) masks
    pred_train, pred_val, pred_test: numpy arrays (predictions)
    """

    # Input and output column names
    featurization: FeaturizationMethod
    endpoints: List[str]

    # Predictive model
    model: Any

    # Datasets
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


@ray.remote
def _train_single_dataset_remote(
    trainer_cls: Type[BaseModelTrainer],
    trainer_kwargs: Dict[str, Any],
    hf_path: str,
    root_dir: str,
    early_stopping_rounds: int = 50,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_root: Optional[str] = None,
    seed: Optional[int] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_json: bool = False,
    dry_run: bool = False,
    max_duration_seconds: Optional[float] = None,
    n_fingerprint_bits: Optional[int] = None,
    featurization: FeaturizationMethod = FeaturizationMethod.MORGAN_FP,
) -> Tuple[str, Dict[str, object]]:
    """Run a single dataset trainer in a Ray worker.

    Parameters
    ----------
    trainer_cls : type
        A concrete trainer class that implements :class:`BaseModelTrainer`.
    trainer_kwargs : dict
        Keyword arguments used to construct ``trainer_cls``.
    hf_path : str
        Path to the HF ``hf_dataset`` directory for this task.
    root_dir : str
        Path of the dataset root used for relative metadata inference.
    early_stopping_rounds : int, optional
        Number of early stopping rounds to use during training.
    sample_weight_mapping : dict, optional
        Optional mapping of dataset names to sample weights.
    output_root : str, optional
        Root directory where artifacts should be written.
    seed : int, optional
        Seed used as a base for per-run randomization.
    featurization : :class:`FeaturizationMethod`, optional
        Which type of featurization to expect/validate for the dataset. The
        loader currently defaults to Morgan fingerprints when present.

    Returns
    -------
    tuple
        ``(rel_key, payload)`` where ``payload`` includes keys ``metrics``,
        ``meta`` (split metadata), ``status``, and timing fields.
    """
    hf_dir = Path(hf_path)
    root = Path(root_dir)
    # Local import for use inside Ray worker
    from admet.data.load import load_dataset as _load_dataset

    meta = infer_split_metadata(hf_dir, root)
    rel_key = str(meta.get("relative_path", hf_dir.name))

    start_ts = datetime.datetime.now()
    start_time = start_ts.isoformat()
    try:
        # Configure logging inside the worker to match the originating process
        from admet.logging import configure_logging

        if log_level or log_file or log_json:
            configure_logging(level=log_level or "INFO", file=log_file, structured=log_json)
    except Exception:
        # ignore logging configuration failures in remote worker
        pass

    try:
        # Load dataset based on requested featurization; currently the
        # loader accepts `n_fingerprint_bits` and will validate fingerprint
        # columns when requested. For SMILES featurization no fingerprint
        # columns are required and we fall-back to the same loader here.
        # Future loaders may accept `featurization` as a parameter.
        logger.info("Loading dataset '%s' with featurization=%s", hf_dir, featurization)
        if featurization == FeaturizationMethod.MORGAN_FP:
            if n_fingerprint_bits is not None:
                dataset = _load_dataset(hf_dir, n_fingerprint_bits=n_fingerprint_bits)
            else:
                dataset = _load_dataset(hf_dir)
        else:
            # For SMILES featurization we don't require fingerprint columns.
            dataset = _load_dataset(hf_dir)
        out_dir: Optional[Path] = None
        if output_root is not None:
            base = Path(output_root)
            cluster = str(meta.get("cluster", "unknown_method"))
            split = str(meta.get("split", "unknown_split"))
            fold = str(meta.get("fold", "unknown_fold"))
            out_dir = base / cluster / f"split_{split}" / f"fold_{fold}"

        # Construct trainer instance and run
        # Ensure per-run seed is passed into trainer kwargs so BaseModelTrainer.run uses it
        if seed is not None:
            trainer_kwargs = dict(trainer_kwargs)
            trainer_kwargs.setdefault("seed", seed)
        trainer = trainer_cls(**trainer_kwargs)
        metrics = trainer.fit(
            dataset,
            sample_weight_mapping=sample_weight_mapping,
            early_stopping_rounds=early_stopping_rounds,
            output_dir=out_dir,
            dry_run=dry_run,
        )
        # Compute end time / duration
        end_ts = datetime.datetime.utcnow()
        end_time = end_ts.isoformat()
        duration_seconds = (end_ts - start_ts).total_seconds()

        # Choose status heuristically
        status = "ok"

        if dry_run:
            status = "skipped"

        # If metrics is None, treat as error
        if metrics is None:
            status = "error"

        # If any expected split is missing or a macro is missing, mark partial
        elif isinstance(metrics, dict):
            expected_splits = {"train", "validation", "test"}
            present = set(metrics.keys())
            if not expected_splits.issubset(present):
                status = "partial"
            else:
                # Check if all macro fields present
                for split in expected_splits:
                    if not isinstance(metrics.get(split, {}), dict) or "macro" not in metrics[split]:
                        status = "partial"
                        break

        if max_duration_seconds is not None and duration_seconds > max_duration_seconds:
            status = "timeout"

        return rel_key, {
            "metrics": metrics,
            "meta": meta,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
        }

    except Exception as exc:  # pragma: no cover - can't reason about ray worker internals
        logger.exception("Dataset training failed for %s: %s", rel_key, exc)
        end_ts = datetime.datetime.now()
        end_time = end_ts.isoformat()
        duration_seconds = (end_ts - start_ts).total_seconds()
        return rel_key, {
            "metrics": None,
            "meta": meta,
            "error": str(exc),
            "status": "error",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
        }


class BaseModelTrainer(ABC):
    """Abstract base class for single-dataset training orchestration.

    Concrete implementations (e.g. XGBoostTrainer) supply backend‑specific
    logic for feature/target extraction, model construction, metrics, and
    artifact persistence. The :meth:`run` method (implemented here) performs
    the end‑to‑end train‑evaluate‑save cycle.

    Attributes
    ----------
    model_cls : type
        The model class (subclass of :class:`admet.model.base.BaseModel`) used
        to construct the backend model.
    model_params : dict
        Backend-specific hyperparameters passed to the model upon
        instantiation.
    seed : int or None
        Random seed for deterministic training.
    device : str or None
        Device selection (e.g., ``'cpu'`` or ``'cuda'``) used by backends that
        execute on accelerators.
    mixed_precision : bool
        Whether to use mixed-precision training for accelerated backends.
    """

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

        # NOTE: Implemented by subclasses
        self.featurization = FeaturizationMethod.NONE
        self.model = None

    def prepare_features(self, dataset: LoadedDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare fingerprint feature matrices for each split.

        Parameters
        ----------
        dataset : LoadedDataset
            A loaded HF dataset produced by :func:`admet.data.load.load_dataset`.

        Returns
        -------
        tuple
            ``(X_train, X_val, X_test)`` arrays suitable for training.
        """
        # Get feature columns based on featurization method (corrected logic)
        if self.featurization == FeaturizationMethod.MORGAN_FP:
            feature_cols = dataset.fingerprint_cols
        elif self.featurization == FeaturizationMethod.SMILES:
            feature_cols = [dataset.smiles_col]
        else:
            raise ValueError(f"Unsupported featurization method: {self.featurization}")

        if not feature_cols:
            raise ValueError("No feature columns found in dataset; cannot prepare features.")

        # Check that all fingerprint columns are present in each split
        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing fingerprint columns in {split_name} split: {missing[:10]}")

        X_train = _extract_features(dataset.train, feature_cols)
        X_val = _extract_features(dataset.val, feature_cols)
        X_test = _extract_features(dataset.test, feature_cols)
        return X_train, X_val, X_test

    def prepare_targets(self, dataset: LoadedDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract endpoint target arrays for each split.

        Parameters
        ----------
        dataset : LoadedDataset
            Loaded dataset with endpoint columns.

        Returns
        -------
        tuple
            ``(Y_train, Y_val, Y_test)`` numpy arrays with NaNs for missing targets.
        """
        endpoints = dataset.endpoints
        if not endpoints:
            raise ValueError("No endpoints found in dataset; cannot prepare targets.")
        # Check that all endpoint columns are present in each split
        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in endpoints if c not in df.columns]
            if missing:
                raise ValueError(f"Missing endpoint columns in {split_name} split: {missing}")
        Y_train = _extract_targets(dataset.train, endpoints)
        Y_val = _extract_targets(dataset.val, endpoints)
        Y_test = _extract_targets(dataset.test, endpoints)
        return Y_train, Y_val, Y_test

    def prepare_masks(
        self,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        Y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct per‑split missing‑target masks.

        Parameters
        ----------
        Y_train, Y_val, Y_test : numpy.ndarray
            Target matrices with ``NaN`` for missing labels.

        Returns
        -------
        tuple
            Boolean masks (train, validation, test) where ``True`` indicates
            presence of a label.
        """
        mask_train = _target_mask(Y_train)
        mask_val = _target_mask(Y_val)
        mask_test = _target_mask(Y_test)
        return mask_train, mask_val, mask_test

    def build_sample_weights(
        self,
        dataset: LoadedDataset,
        sample_weight_mapping: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        """Create a per-row sample-weight vector for the training split.

        This function maps the dataset's ``Dataset`` column values through
        ``sample_weight_mapping`` (with optional ``default`` key). When not
        supplied, returns ``None``.
        """
        if not sample_weight_mapping:
            return None
        mapping = sample_weight_mapping
        # Ensure the 'Dataset' column is present when building sample weights
        if "Dataset" not in dataset.train.columns:
            raise ValueError("Training split missing 'Dataset' column needed for sample weight mapping")
        default = mapping.get("default", 1.0)
        # Use pandas vectorized mapping for performance
        sw_series = dataset.train["Dataset"].astype(str).map(lambda x: mapping.get(x, default))
        return sw_series.to_numpy(dtype=float)

    def build_model(self, endpoints: List[str]) -> BaseModel:
        """Instantiate the backend model with endpoints and configuration.

        Parameters
        ----------
        endpoints : list of str
            Endpoints the model should predict.

        Returns
        -------
        BaseModel
            The instantiated model ready to be fit.
        """
        return self.model_cls(endpoints, self.model_params, self.seed)  # type: ignore[call-arg]

    def compute_metrics(
        self,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        Y_test: np.ndarray,
        pred_train: np.ndarray,
        pred_val: np.ndarray,
        pred_test: np.ndarray,
        mask_train: np.ndarray,
        mask_val: np.ndarray,
        mask_test: np.ndarray,
        endpoints: List[str],
    ) -> AllMetrics:
        """Compute macro and per-endpoint metrics for all splits.

        Parameters
        ----------
        Y_train, Y_val, Y_test : numpy.ndarray
            True target arrays for each split.
        pred_train, pred_val, pred_test : numpy.ndarray
            Predicted arrays for each split.
        mask_train, mask_val, mask_test : numpy.ndarray
            Masks indicating observed targets for each split.
        endpoints : list of str
            Endpoint columns corresponding to the final axis of ``Y``.

        Returns
        -------
        dict
            Nested metrics dictionary keyed by split name.
        """
        return {
            "train": compute_metrics_log_and_linear(Y_train, pred_train, mask_train, endpoints),
            "validation": compute_metrics_log_and_linear(Y_val, pred_val, mask_val, endpoints),
            "test": compute_metrics_log_and_linear(Y_test, pred_test, mask_test, endpoints),
        }

    def save_artifacts(
        self,
        model: "BaseModel",
        metrics: AllMetrics,
        output_dir: Path,
        outputs: "RunOutputs",
        *,
        dataset: LoadedDataset,
        extra_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        """Save model, metrics, and generated figures to ``output_dir``.

        Parameters
        ----------
        model : BaseModel
            The already-trained model instance.
        metrics : dict
            Nested metrics dictionary as returned by :meth:`compute_metrics`.
        output_dir : pathlib.Path
            Directory to write artifacts under.
        dataset : LoadedDataset
            The dataset used for training; required for plotting.
        extra_meta : dict, optional
            Optional extra metadata written alongside metrics (currently not
            persisted directly; reserved for future extension).
        """
        assert isinstance(model, BaseModel)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir / "model"))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # get number of CPUs for parallel plotting
        n_cpus = multiprocessing.cpu_count()
        logger.info("Using %d CPU cores for plotting.", n_cpus)

        # Use precomputed outputs to avoid re-extracting/recomputing
        endpoints = outputs.endpoints
        Y_train, Y_val, Y_test = outputs.Y_train, outputs.Y_val, outputs.Y_test
        mask_train, mask_val, mask_test = outputs.mask_train, outputs.mask_val, outputs.mask_test
        pred_train, pred_val, pred_test = outputs.pred_train, outputs.pred_val, outputs.pred_test

        y_true = {"train": Y_train, "validation": Y_val, "test": Y_test}
        y_pred = {"train": pred_train, "validation": pred_val, "test": pred_test}
        y_mask = {"train": mask_train, "validation": mask_val, "test": mask_test}
        fig_root = output_dir / "figures"
        for space in ["log", "linear"]:
            space_dir = fig_root / space
            space_dir.mkdir(parents=True, exist_ok=True)
            # Parity plots: one file per endpoint
            plot_parity_grid(
                y_true,
                y_pred,
                y_mask,
                endpoints,
                space=space,
                save_dir=space_dir,
                n_jobs=n_cpus,
            )
            # Metric bars
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

    # --- High-level orchestration ---
    def fit(
        self,
        dataset: LoadedDataset,
        *,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        early_stopping_rounds: Optional[int] = None,
        output_dir: Optional[Path] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Dict[str, Any]]:  # pragma: no cover - abstract
        """End-to-end training and evaluation on a single dataset.

        Implements the canonical training flow:
        ``prepare_features`` -> ``prepare_targets`` -> ``prepare_masks`` ->
        ``build_model`` -> ``fit`` -> ``predict`` -> ``compute_metrics`` ->
        ``save_artifacts`` (if requested).

        Parameters
        ----------
        dataset : LoadedDataset
            The dataset used for training and evaluation.
        sample_weight_mapping : dict or None
            Optional mapping for training sample weights.
        early_stopping_rounds : int
            Early stopping parameter for the model.
        output_dir : pathlib.Path or None
            Directory where artifacts will be saved; if none, artifacts are
            not written to disk.
        extra_meta : dict, optional
            Any additional metadata to include with saved artifacts.

        Returns
        -------
        dict
            Nested metrics dictionary keyed by split; each split should
            expose at least a ``macro`` metrics dict.
        """
        # Default orchestration for typical trainers: prepare -> build -> fit -> predict -> metrics -> save
        # Seed global RNGs early for reproducibility
        set_global_seeds(self.seed)
        logger.info("Using seed=%s for training", self.seed)

        endpoints: List[str] = list(dataset.endpoints)
        # Basic validation
        if dry_run:
            logger.info("Dry-run enabled; will not fit models.")
        if not endpoints:
            raise ValueError("No endpoints found in dataset; cannot train model.")
        # Ensure dataset has required splits
        if not (hasattr(dataset, "train") and hasattr(dataset, "val") and hasattr(dataset, "test")):
            raise ValueError("LoadedDataset must provide 'train','val','test' splits")

        logger.info("Preparing features/targets for endpoints: %s", endpoints)
        X_train, X_val, X_test = self.prepare_features(dataset)
        Y_train, Y_val, Y_test = self.prepare_targets(dataset)
        mask_train, mask_val, mask_test = self.prepare_masks(Y_train, Y_val, Y_test)
        logger.info(
            "Prepared dataset splits: X_train=%s, X_val=%s, X_test=%s; Y_train=%s; masks=%s",
            getattr(X_train, "shape", None),
            getattr(X_val, "shape", None),
            getattr(X_test, "shape", None),
            getattr(Y_train, "shape", None),
            getattr(mask_train, "shape", None),
        )

        sample_weight = self.build_sample_weights(dataset, sample_weight_mapping)
        logger.info("Built sample_weight with shape=%s", getattr(sample_weight, "shape", None))
        logger.info("Built sample_weight with shape=%s", getattr(sample_weight, "shape", None))

        if dry_run:
            logger.info("Dry-run: skip model construction and training")
            # Return minimal metrics structure
            return {
                "train": {"macro": {}},
                "validation": {"macro": {}},
                "test": {"macro": {}},
            }

        self.model = self.build_model(endpoints)
        logger.info("Built model: %s", type(self.model).__name__)
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

        # Predictions
        pred_train = self.model.predict(X_train)
        pred_val = self.model.predict(X_val)
        pred_test = self.model.predict(X_test)

        metrics = self.compute_metrics(
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

        outputs = RunOutputs(
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
                self.model, metrics, output_dir, outputs, dataset=dataset, extra_meta=extra_meta
            )

        return metrics


class BaseEnsembleTrainer:
    """Ray-based ensemble trainer for multiple datasets.

    This class encapsulates Ray orchestration for running a single-dataset
    trainer over many discovered dataset directories.
    """

    def __init__(
        self,
        *,
        trainer_cls: Type[BaseModelTrainer],
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trainer_cls = trainer_cls
        self.trainer_kwargs = trainer_kwargs or {}

    def discover_datasets(self, root: Path) -> List[Path]:
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def infer_metadata(self, hf_path: Path, root: Path) -> Dict[str, object]:
        return infer_split_metadata(hf_path, root)

    def build_output_dir(self, base: Path, meta: Dict[str, object]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"

    def fit_ensemble(
        self,
        root: Path,
        *,
        output_root: Optional[Path] = None,
        early_stopping_rounds: Optional[int] = None,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        num_cpus: Optional[int] = None,
        ray_address: Optional[str] = None,
        seed: Optional[int] = None,
        dry_run: bool = False,
        max_duration_seconds: Optional[float] = None,
        n_fingerprint_bits: Optional[int] = None,
        featurization: FeaturizationMethod = FeaturizationMethod.MORGAN_FP,
    ) -> Dict[str, Dict[str, object]]:
        """Default Ray orchestration for all datasets discovered under `root`.

        This method schedules a remote training job for each dataset using the
        generic `_train_single_dataset_remote` helper and aggregates results.
        """
        if output_root is None:
            output_root = Path("models")

        hf_paths = list(self.discover_datasets(root))
        if not hf_paths:
            raise ValueError(f"No 'hf_dataset' directories found under {root}.")

        logger.info("Discovered %d hf_dataset directories under %s", len(hf_paths), root)

        started_local_ray = False
        if not ray.is_initialized():
            if ray_address and ray_address.lower() != "local":
                logger.info("Connecting to existing Ray cluster at %s", ray_address)
                ray.init(address=ray_address, ignore_reinit_error=True)
            else:
                cpu_count = num_cpus or multiprocessing.cpu_count()
                logger.info("Starting local Ray runtime with %d CPUs", cpu_count)
                ray.init(num_cpus=cpu_count, ignore_reinit_error=True)
                started_local_ray = True

        sorted_hf_paths = sorted(hf_paths, key=lambda p: str(p.relative_to(root)))
        base_seed = seed if seed is not None else 0

        # Decide logging config for workers based on our current logging root
        try:
            from admet.logging import get_logging_config

            log_cfg = get_logging_config()
        except Exception:
            log_cfg = {"level": "INFO", "file": None, "structured": False}

        tasks = []
        for idx, hf_path in enumerate(sorted_hf_paths):
            model_seed = base_seed + idx
            tasks.append(
                _train_single_dataset_remote.remote(
                    self.trainer_cls,
                    self.trainer_kwargs,
                    str(hf_path),
                    str(root),
                    early_stopping_rounds=early_stopping_rounds,
                    sample_weight_mapping=sample_weight_mapping,
                    output_root=str(output_root),
                    seed=model_seed,
                    log_level=log_cfg.get("level"),
                    log_file=log_cfg.get("file"),
                    log_json=log_cfg.get("structured") or False,
                    dry_run=dry_run,
                    max_duration_seconds=max_duration_seconds,
                    n_fingerprint_bits=n_fingerprint_bits,
                    featurization=featurization,
                )
            )

        remaining = set(tasks)
        results: List[Tuple[str, Dict[str, object]]] = []
        total = len(tasks)
        completed = 0
        logger.info("Starting Ray training for %d models", total)
        while remaining:
            done, remaining = ray.wait(list(remaining), num_returns=1)
            for d in done:
                rel_key, payload = ray.get(d)
                results.append((rel_key, payload))
                completed += 1
                logger.info(
                    "Completed %d/%d models (%s) - %d still in progress",
                    completed,
                    total,
                    rel_key,
                    len(remaining),
                )

        aggregated = {rel_key: payload for rel_key, payload in results}

        # Build and persist a summary of macro metrics across all datasets.
        summary_rows: List[Dict[str, object]] = []
        success_count = 0
        failure_count = 0
        for rel_key, payload in aggregated.items():
            status = str(payload.get("status", "ok"))
            if status == "ok":
                success_count += 1
            else:
                failure_count += 1
            metrics = payload.get("metrics") or {}  # type: ignore[assignment]
            meta = payload.get("meta") or {}  # type: ignore[assignment]
            # include timing information in the summary rows
            start_time = payload.get("start_time")
            end_time = payload.get("end_time")
            duration_seconds = payload.get("duration_seconds")
            for split_name in ["train", "validation", "test"]:
                split_metrics = metrics.get(split_name, {})  # type: ignore[assignment]
                macro = split_metrics.get("macro", {})
                row: Dict[str, object] = {"dataset": rel_key, "split": split_name}
                row["status"] = status
                row["start_time"] = start_time
                row["end_time"] = end_time
                row["duration_seconds"] = duration_seconds
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        row[f"meta_{k}"] = v
                if isinstance(macro, dict):
                    for k, v in macro.items():
                        row[k] = v
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_root = output_root
            summary_root.mkdir(parents=True, exist_ok=True)
            summary_csv = summary_root / "metrics_summary.csv"
            summary_json = summary_root / "metrics_summary.json"
            summary_df.to_csv(summary_csv, index=False)
            summary_json.write_text(summary_df.to_json(orient="records", indent=2))
            logger.info("Wrote metrics summary to %s and %s", summary_csv, summary_json)
            logger.info("Ray training summary: %d succeeded, %d failed", success_count, failure_count)

        # shutdown ray runtime if we started it locally
        if started_local_ray:
            try:
                ray.shutdown()
            except Exception:
                logger.warning("ray.shutdown() raised an exception during cleanup", exc_info=True)

        return aggregated


def train_model(
    dataset: LoadedDataset,
    trainer_cls: Type[BaseModelTrainer],
    model_cls: Type[ModelProtocol],
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: Optional[int] = None,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> AllMetrics:
    """Convenience wrapper to train a single HF dataset using XGBoost.

    Parameters
    ----------
    dataset : LoadedDataset
        The dataset to train on.
    model_params : dict, optional
        Hyperparameters for the XGBoost backend. If ``None``, default
        parameters are used.
    early_stopping_rounds : int, optional
        Early stopping parameter forwarded to the model.
    sample_weight_mapping : dict, optional
        Mapping of dataset names to sample weights.
    output_dir : pathlib.Path, optional
        Path to write artifacts under.
    seed : int, optional
        Seed for deterministic model fit.

    Returns
    -------
    dict
        Nested metrics dictionary keyed by split name (train, validation, test).

    Notes
    -----
    For XGBoost: trainer_cls=XGBoostTrainer, model_cls=XGBoostMultiEndpoint,
    """

    trainer = trainer_cls(
        model_cls=model_cls,
        model_params=model_params,
        seed=seed,
    )
    return trainer.fit(
        dataset,
        sample_weight_mapping=sample_weight_mapping,
        early_stopping_rounds=early_stopping_rounds,
        output_dir=output_dir,
    )


def train_ensemble(
    root: Path,
    *,
    ensemble_trainer_cls: Type[BaseEnsembleTrainer],
    trainer_cls: Type[BaseModelTrainer],
    model_cls: Type[ModelProtocol],
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: Optional[int] = None,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_root: Optional[Path] = None,
    seed: Optional[int] = None,
    n_fingerprint_bits: Optional[int] = None,
    num_cpus: Optional[int] = None,
    ray_address: Optional[str] = None,
    dry_run: bool = False,
    max_duration_seconds: Optional[float] = None,
) -> Dict[str, Dict[str, object]]:
    """Train XGBoost models in parallel via Ray across multiple HF datasets.

    Parameters
    ----------
    root : pathlib.Path
        Root directory to search for hf_dataset directories.
    trainer_cls : type
        A concrete trainer class that implements :class:`BaseEnsembleTrainer`.
    model_cls : type
        The model class (subclass of :class:`admet.model.base.BaseModel`) used
        to construct the backend model.
    model_params : dict, optional
        Hyperparameter dictionary forwarded to the underlying model.
    early_stopping_rounds : int, optional
        Early stopping value per run.
    sample_weight_mapping : dict, optional
        Sample weights mapping for each dataset.
    output_root : pathlib.Path, optional
        Root directory where per-dataset artifacts are written.
    seed : int, optional
        Base seed for deterministic per-dataset seeds.
    num_cpus : int, optional
        Number of CPUs to use when starting a local Ray instance.
    ray_address : str, optional
        Optional Ray address to connect to; if ``'local'``, a local Ray
        instance is started if not already running.

    Returns
    -------
    dict
        Mapping of dataset relative key to payload dict containing metrics,
        metadata, status, and timing fields.

    Notes
    -----
    For XGBoost: ensemble_trainer_cls=BaseEnsembleTrainer, trainer_cls=XGBoostTrainer, model_cls=XGBoostMultiEndpoint,
    """

    ray_trainer = ensemble_trainer_cls(
        trainer_cls=trainer_cls,
        trainer_kwargs={
            "model_cls": model_cls,
            "model_params": model_params,
            "seed": seed,
        },
    )
    aggregated = ray_trainer.fit_ensemble(
        root,
        output_root=output_root,
        early_stopping_rounds=early_stopping_rounds,
        sample_weight_mapping=sample_weight_mapping,
        num_cpus=num_cpus,
        ray_address=ray_address,
        dry_run=dry_run,
        max_duration_seconds=max_duration_seconds,
        n_fingerprint_bits=n_fingerprint_bits,
        seed=seed,
    )

    return aggregated


__all__ = [
    "BaseModelTrainer",
    "BaseEnsembleTrainer",
    "FeaturizationMethod",
    "RunOutputs",
    "train_model",
    "train_ensemble",
]
