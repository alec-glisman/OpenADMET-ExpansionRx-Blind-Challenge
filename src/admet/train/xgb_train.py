"""admet.train.xgb_train
=========================

XGBoost‑specific trainer implementations for multi‑endpoint ADMET regression.

Components
----------
``XGBoostTrainer``
    Concrete :class:`admet.train.base_trainer.BaseModelTrainer` implementing
    fingerprint feature extraction, endpoint target handling with masks, and
    metric computation via :func:`admet.evaluate.metrics.compute_metrics_log_and_linear`.

``XGBoostRayMultiDatasetTrainer``
    Concrete :class:`admet.train.base_trainer.BaseRayMultiDatasetTrainer` that
    discovers ``hf_dataset`` directories and orchestrates Ray‑based parallel
    training jobs.

Convenience wrappers (``train_xgb_models`` / ``train_xgb_models_ray``) support
single‑dataset and multi‑dataset execution respectively.

Artifacts Written
-----------------
Under an output directory passed to ``run`` / high‑level wrappers, the trainer
persists: ``model/`` (XGBoost booster per endpoint), ``metrics.json`` (nested
split metrics), and ``figures/`` containing parity plots and metric bar charts
for log and linear target spaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple, cast
import logging
import multiprocessing

import numpy as np

from admet.data.load import LoadedDataset
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.train.base_trainer import RunOutputs
from admet.model.base import BaseModel

# Note: seed setting is handled in BaseModelTrainer.run
from admet.evaluate.metrics import compute_metrics_log_and_linear, AllMetrics
from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars
from admet.train.base_trainer import BaseModelTrainer, BaseRayMultiDatasetTrainer, infer_split_metadata

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


class XGBoostTrainer(BaseModelTrainer):
    """Single-dataset trainer implementation using XGBoost.

    This class implements :class:`~admet.train.base_trainer.BaseModelTrainer`
    for XGBoost backend models, delegates per-endpoint training to
    :class:`admet.model.xgb_wrapper.XGBoostMultiEndpoint`, and preserves
    existing metrics and artifact formats used by the higher-level
    orchestration logic.

    Notes
    -----
    The class uses dataset fingerprints as features extracted from
    :class:`admet.data.load.LoadedDataset` and predicts all endpoints using
    independent XGBoost regressors per endpoint.
    """

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
        fp_cols = dataset.fingerprint_cols
        if not fp_cols:
            raise ValueError("No fingerprint columns found in dataset; cannot prepare features.")
        # Check that all fingerprint columns are present in each split
        for split_name, df in [
            ("train", dataset.train),
            ("validation", dataset.val),
            ("test", dataset.test),
        ]:
            missing = [c for c in fp_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing fingerprint columns in {split_name} split: {missing[:10]}")
        # Reuse existing helpers to keep behavior identical.
        X_train = _extract_features(dataset.train, fp_cols)
        X_val = _extract_features(dataset.val, fp_cols)
        X_test = _extract_features(dataset.test, fp_cols)
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
        """Build and return an XGBoostMultiEndpoint instance.

        Parameters
        ----------
        endpoints : list of str
            The endpoint names to predict.

        Returns
        -------
        XGBoostMultiEndpoint
            Backend model instance configured with ``self.model_params``.
        """
        # Instantiate model using self.model_cls to support dependency injection/testability.
        # We type-hint the return as XGBoostMultiEndpoint for callers expecting the XGBoost wrapper,
        # but construct via `self.model_cls` so other model classes can be injected in tests.
        # Construct model via the injected model class; some model classes may
        # accept different initialization signatures, so we forward positional
        # args for the common XGBoostMultiEndpoint constructor.
        from typing import cast, Any

        return cast(Any, self.model_cls)(endpoints, self.model_params, self.seed)

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

    def run(
        self,
        dataset: LoadedDataset,
        *,
        sample_weight_mapping: Optional[Dict[str, float]] = None,
        early_stopping_rounds: int = 50,
        output_dir: Optional[Path] = None,
        extra_meta: Optional[Dict[str, object]] = None,
        dry_run: bool = False,
    ) -> AllMetrics:
        """Perform the complete train -> predict -> eval flow.

        High-level orchestration that delegates to
        :meth:`BaseModelTrainer.run`; documented here for convenience.
        """
        # Delegate full orchestration to BaseModelTrainer.run
        return super().run(
            dataset,
            sample_weight_mapping=sample_weight_mapping,
            early_stopping_rounds=early_stopping_rounds,
            output_dir=output_dir,
            extra_meta=extra_meta,
            dry_run=dry_run,
        )


class XGBoostRayMultiDatasetTrainer(BaseRayMultiDatasetTrainer):
    """Multi-dataset trainer using Ray for XGBoost runs.

    This class implements the small set of dataset-discovery and output path
    formation hooks required by :class:`BaseRayMultiDatasetTrainer`. The
    actual Ray orchestration and scheduling is implemented by the base
    class method :meth:`BaseRayMultiDatasetTrainer.run_all` which calls the
    generic remote helper for each dataset.
    """

    def discover_datasets(self, root: Path) -> List[Path]:
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def infer_metadata(self, hf_path: Path, root: Path) -> Dict[str, object]:
        return infer_split_metadata(hf_path, root)

    def build_output_dir(self, base: Path, meta: Dict[str, object]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


def train_xgb_models(
    dataset: LoadedDataset,
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
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
    """

    trainer = XGBoostTrainer(
        model_cls=cast(type, XGBoostMultiEndpoint),  # type: ignore[arg-type]
        model_params=model_params,
        seed=seed,
    )
    return trainer.run(
        dataset,
        sample_weight_mapping=sample_weight_mapping,
        early_stopping_rounds=early_stopping_rounds,
        output_dir=output_dir,
    )


def train_xgb_models_ray(
    root: Path,
    *,
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
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
    """

    ray_trainer = XGBoostRayMultiDatasetTrainer(
        trainer_cls=XGBoostTrainer,
        trainer_kwargs={
            "model_cls": cast(type, XGBoostMultiEndpoint),  # type: ignore[arg-type]
            "model_params": model_params,
            "seed": seed,
        },
    )
    return ray_trainer.run_all(
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


__all__ = ["train_xgb_models", "train_xgb_models_ray"]
