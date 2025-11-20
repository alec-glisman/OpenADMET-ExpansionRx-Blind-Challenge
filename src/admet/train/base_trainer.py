"""admet.train.base_trainer
==========================

Core training abstractions and helpers used across model backends.

This module defines the minimal API and several concrete utilities that
implement multi-dataset training orchestration for backend-agnostic
trainers. The primary abstractions are:

``BaseModelTrainer``
    Abstract base class that encapsulates the single-dataset training flow.

``BaseRayMultiDatasetTrainer``
    Higher-level orchestrator providing a default Ray-based multi-dataset
    execution engine and a convenient remote helper for single-dataset tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
from dataclasses import dataclass

import numpy as np
import ray
import pandas as pd
import multiprocessing

from admet.data.load import LoadedDataset
from admet.model.base import BaseModel
import logging
from admet.utils import set_global_seeds

logger = logging.getLogger(__name__)


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

    Returns
    -------
    tuple
        A 2-tuple: (rel_key, payload). ``rel_key`` is a string identifier for
        the dataset (typically a relative path). ``payload`` is a mapping
        containing ``metrics`` and ``meta`` for the run.
    """
    hf_dir = Path(hf_path)
    root = Path(root_dir)
    # Local import for use inside Ray worker
    from admet.data.load import load_dataset as _load_dataset

    meta = infer_split_metadata(hf_dir, root)
    rel_key = str(meta.get("relative_path", hf_dir.name))

    try:
        # Configure logging inside the worker to match the originating process
        from admet.logging import configure_logging

        if log_level or log_file or log_json:
            configure_logging(level=log_level or "INFO", file=log_file, structured=log_json)
    except Exception:
        # ignore logging configuration failures in remote worker
        pass

    try:
        dataset = _load_dataset(hf_dir)
        out_dir: Optional[Path] = None
        if output_root is not None:
            base = Path(output_root)
            cluster = str(meta.get("cluster", "unknown_method"))
            split = str(meta.get("split", "unknown_split"))
            fold = str(meta.get("fold", "unknown_fold"))
            out_dir = base / cluster / f"split_{split}" / f"fold_{fold}"

        # Construct trainer instance and run
        trainer = trainer_cls(**trainer_kwargs)
        metrics = trainer.run(
            dataset,
            sample_weight_mapping=sample_weight_mapping,
            early_stopping_rounds=early_stopping_rounds,
            output_dir=out_dir,
        )

        return rel_key, {"metrics": metrics, "meta": meta}
    except Exception as exc:  # pragma: no cover - can't reason about ray worker internals
        logger.exception("Dataset training failed for %s: %s", rel_key, exc)
        return rel_key, {"metrics": None, "meta": meta, "error": str(exc)}


class BaseModelTrainer(ABC):
    """Abstract base class for single-dataset training orchestration.

    This class defines the minimal operation points a backend trainer must
    provide to be usable by higher level orchestration. Concrete
    implementations (e.g., XGBoostTrainer) should implement the hooks and
    the :meth:`run` method which performs a full train-evaluate-save cycle.

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
        model_cls: Type[BaseModel],
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

    # --- Hooks that concrete trainers must implement ---

    @abstractmethod
    def prepare_features(self, dataset: LoadedDataset) -> Tuple[Any, Any, Any]:
        """Convert dataset splits into backend-ready feature representations.

        Parameters
        ----------
        dataset : LoadedDataset
            Loaded dataset with train/validation/test splits and fingerprint columns.

        Returns
        -------
        tuple
            Tuple containing backend-ready feature arrays for train/val/test.
        """

        raise NotImplementedError

    @abstractmethod
    def prepare_targets(self, dataset: LoadedDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract endpoint targets per split as numpy arrays.

        Parameters
        ----------
        dataset : LoadedDataset
            Dataset providing endpoint columns in each split.

        Returns
        -------
        tuple
            Numpy arrays for Y_train, Y_val, Y_test with NaNs for missing targets.
        """

        raise NotImplementedError

    @abstractmethod
    def prepare_masks(
        self,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        Y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build missing-target masks from targets for each split.

        Parameters
        ----------
        Y_train, Y_val, Y_test : numpy.ndarray
            Target arrays for each split with NaNs for missing values.

        Returns
        -------
        tuple
            Mask arrays with 1 for present targets and 0 for missing targets.
        """

        raise NotImplementedError

    @abstractmethod
    def build_sample_weights(
        self,
        dataset: LoadedDataset,
        sample_weight_mapping: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        """Construct a 1D sample weight array for the training split.

        Parameters
        ----------
        dataset : LoadedDataset
            Dataset used to extract sample identifiers from the training split.
        sample_weight_mapping : dict, optional
            Mapping of dataset names to weights with optional ``default``.

        Returns
        -------
        numpy.ndarray or None
            1-D array of sample weights aligned with the training split or
            ``None`` when not applicable.
        """

        raise NotImplementedError

    @abstractmethod
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

        raise NotImplementedError

    @abstractmethod
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
    ) -> Dict[str, Dict[str, Any]]:
        """Compute all metrics for the provided true and predicted values.

        Parameters
        ----------
        Y_train, Y_val, Y_test : numpy.ndarray
            True target arrays for each split.
        pred_train, pred_val, pred_test : numpy.ndarray
            Predicted values for each split.
        mask_train, mask_val, mask_test : numpy.ndarray
            Masks indicating present/absent values for each target.
        endpoints : list of str
            Endpoint names corresponding to target columns.

        Returns
        -------
        dict
            Nested dict keyed by split name (``train``, ``validation``,
            ``test``) containing metric dictionaries.
        """

        raise NotImplementedError

    @abstractmethod
    def save_artifacts(
        self,
        model: BaseModel,
        metrics: Dict[str, Dict[str, Any]],
        output_dir: Path,
        *,
        dataset: LoadedDataset,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist model, metrics, and optional plots to disk.

        Parameters
        ----------
        model : BaseModel
            The trained model instance to persist.
        metrics : dict
            Nested metrics as returned by :meth:`compute_metrics`.
        output_dir : pathlib.Path
            Output directory to write artifacts under.
        dataset : LoadedDataset
            Dataset used for generating plots and saving metadata.
        extra_meta : dict, optional
            Additional metadata to store alongside metrics.
        """

        raise NotImplementedError

    # --- High-level orchestration ---

    @abstractmethod
    def run(
        self,
        dataset: LoadedDataset,
        *,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        early_stopping_rounds: int = 50,
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
            Nested metrics dictionary keyed by split; identical to the
            prior API of :func:`train_xgb_models`.
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

        model = self.build_model(endpoints)
        logger.info("Built model: %s", type(model).__name__)
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

        if output_dir is not None:
            self.save_artifacts(model, metrics, output_dir, dataset=dataset, extra_meta=extra_meta)

        return metrics


class BaseRayMultiDatasetTrainer(ABC):
    """Abstract Ray-based multi-dataset trainer.

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

    @abstractmethod
    def discover_datasets(self, root: Path) -> Sequence[Path]:
        """Return a sequence of dataset directories to train on."""

    def infer_metadata(self, hf_path: Path, root: Path) -> Dict[str, Any]:
        """Default implementation that calls :func:`infer_split_metadata`.

        Backends may override this if they desire a different parsing heuristic.
        """
        return infer_split_metadata(hf_path, root)

    @abstractmethod
    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        """Compose an output directory path from base and metadata."""
        raise NotImplementedError

    def run_all(
        self,
        root: Path,
        *,
        output_root: Optional[Path] = None,
        early_stopping_rounds: int = 50,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        num_cpus: Optional[int] = None,
        ray_address: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Default Ray orchestration for all datasets discovered under `root`.

        This method schedules a remote training job for each dataset using the
        generic `_train_single_dataset_remote` helper and aggregates results.
        """
        if output_root is None:
            output_root = Path("xgb_artifacts")

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
        for rel_key, payload in aggregated.items():
            metrics = payload.get("metrics") or {}  # type: ignore[assignment]
            meta = payload.get("meta") or {}  # type: ignore[assignment]
            for split_name in ["train", "validation", "test"]:
                split_metrics = metrics.get(split_name, {})  # type: ignore[assignment]
                macro = split_metrics.get("macro", {})
                row: Dict[str, object] = {"dataset": rel_key, "split": split_name}
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

        # shutdown ray runtime if we started it locally
        if started_local_ray:
            try:
                ray.shutdown()
            except Exception:
                logger.warning("ray.shutdown() raised an exception during cleanup", exc_info=True)

        return aggregated


__all__ = ["BaseModelTrainer", "BaseRayMultiDatasetTrainer"]
