"""Ensemble evaluation helpers for labeled and blind CSVs.

This module provides helpers to:

- Load serialized models from directories (XGBoostMultiEndpoint supported by
  model config inference),
- Convert SMILES to features expected by models (fingerprints / SMILES),
- Run per-model predictions across an array of inputs,
- Aggregate predictions across models (mean/median),
- Provide simple helpers to produce standardized prediction DataFrames and
  metric tables compatible with the existing evaluation pipeline.

The module intentionally avoids direct plotting / CLI binding so it can be
unit-tested easily. Visualizations and CLI integration live in separate
modules. Where applicable, existing conversion functions from
`admet.evaluate.metrics` and `admet.visualize.model_performance` are reused
for parity with the project's conventions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from admet.data.fingerprinting import DEFAULT_FINGERPRINT_CONFIG, FingerprintConfig, MorganFingerprintGenerator
from admet.data.load import load_dataset
from admet.evaluate import metrics as eval_metrics
from admet.model.base import BaseModel
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.visualize.model_performance import _apply_transform_space

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePredictConfig:
    """Configuration for running ensemble predictions from a parent directory.

    Parameters
    ----------
    models_root
        Parent directory under which individual model run directories live.
        Each run directory must contain a ``run_meta.json`` file written by
        the training pipeline, or a legacy ``config.json`` for older
        XGBoost-only runs.
    eval_csv
        Optional path to a labeled evaluation CSV.
    blind_csv
        Optional path to a blind (unlabeled) CSV.
    agg_fn
        Aggregation function to combine per-model predictions.
    n_jobs
        Level of parallelism across models when running predictions.
    train_data_root
        Optional path to the Hugging Face datasets used during training. When
        provided, train/validation/test splits are evaluated and returned in
        the summary.
    """

    models_root: Path
    eval_csv: Optional[Path] = None
    blind_csv: Optional[Path] = None
    agg_fn: str = "mean"
    n_jobs: int = 1
    fingerprint_config: FingerprintConfig = DEFAULT_FINGERPRINT_CONFIG
    train_data_root: Optional[Path] = None


@dataclass
class EnsemblePredictSummary:
    """Summary of ensemble prediction outputs for eval, blind, and training datasets."""

    model_dirs: List[Path]  # (n_models,)
    endpoints: List[str]  # (n_endpoints,)

    # Eval dataset
    preds_log_eval: Optional[pd.DataFrame]
    preds_linear_eval: Optional[pd.DataFrame]
    metrics_log_eval: Optional[pd.DataFrame]
    metrics_linear_eval: Optional[pd.DataFrame]

    # Blind dataset: All model outputs
    preds_log_blind: Optional[pd.DataFrame]
    preds_linear_blind: Optional[pd.DataFrame]

    # Training split evaluations (optional, keyed by split name)
    train_split_evaluations: Optional[Dict[str, "SplitEvaluation"]] = None


@dataclass
class SplitEvaluation:
    """Container for a single split's evaluation artifacts."""

    df_true: pd.DataFrame
    preds_log: pd.DataFrame
    preds_linear: pd.DataFrame
    metrics_log: pd.DataFrame
    metrics_linear: pd.DataFrame


def aggregate_metrics_by_endpoint(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form metrics to wide form keyed by endpoint."""
    pivot_df = (
        metrics_df.drop(columns=["space"], errors="ignore")
        .pivot(index="endpoint", columns="metric", values="value")
        .reset_index()
        .sort_values("endpoint")
        .reset_index(drop=True)
    )
    pivot_df.columns.name = None
    return pivot_df


def log_metrics_to_linear(aggregated_log_df: pd.DataFrame, endpoints: Sequence[str]) -> pd.DataFrame:
    """Convert aggregated log metrics to linear space for non-LogD endpoints."""
    linear_df = aggregated_log_df.copy(deep=True)
    if "endpoint" not in linear_df.columns:
        return linear_df

    log_only_endpoints = set(endpoints)
    log_only_endpoints.discard("LogD")
    if not log_only_endpoints:
        return linear_df

    mask = linear_df["endpoint"].isin(log_only_endpoints)
    for metric_col in ("mae", "rmse"):
        if metric_col not in linear_df.columns:
            continue
        linear_df.loc[mask, metric_col] = np.power(10.0, linear_df.loc[mask, metric_col])
    return linear_df


def _detect_model_type_from_dir(model_dir: Path) -> Optional[str]:
    """Detects a supported model type from a saved model directory.

    Currently supports the XGBoost per-endpoint wrapper which persists
    ``config.json`` and per-endpoint json files.
    """
    cfg = model_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return None
    # Heuristic: presence of model_params + endpoints indicates xgboost
    if "endpoints" in data and "model_params" in data:
        return "xgboost_multi_endpoint"
    return None


def discover_model_runs(models_root: Path) -> List[Path]:
    """Discover model run directories beneath ``models_root``.

    Preference is given to directories containing ``run_meta.json`` written
    by the trainers. For backward compatibility we also support legacy
    XGBoost runs where only a ``config.json`` is present.
    """

    models_root = Path(models_root)
    run_dirs: List[Path] = []
    for meta_path in models_root.rglob("run_meta.json"):
        run_dirs.append(meta_path.parent)
    if not run_dirs:
        for cfg_path in models_root.rglob("config.json"):
            run_dirs.append(cfg_path.parent)
    # Deduplicate while preserving order
    seen: Dict[Path, bool] = {}
    unique_dirs: List[Path] = []
    for d in run_dirs:
        if d not in seen:
            seen[d] = True
            unique_dirs.append(d)
    return unique_dirs


def _parse_split_fold_from_model_dir(model_dir: Path) -> Tuple[int, int]:
    """Extract split and fold identifiers from a model directory path."""
    split_id: Optional[int] = None
    fold_id: Optional[int] = None
    for part in model_dir.parts:
        if part.startswith("split_"):
            try:
                split_id = int(part.replace("split_", ""))
            except ValueError:
                continue
        if part.startswith("fold_"):
            try:
                fold_id = int(part.replace("fold_", ""))
            except ValueError:
                continue
    if split_id is None or fold_id is None:
        raise ValueError(f"Could not infer split/fold from model directory {model_dir}")
    return split_id, fold_id


def _collect_training_splits(
    train_data_root: Path,
    model_dirs: Sequence[Path],
    fingerprint_config: FingerprintConfig,
) -> Dict[str, pd.DataFrame]:
    """Load and concatenate train/val/test splits for all discovered models."""
    combined_frames: Dict[str, List[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    seen_paths: Dict[Path, bool] = {}

    for md in model_dirs:
        try:
            split_id, fold_id = _parse_split_fold_from_model_dir(md)
        except ValueError as exc:
            logger.warning("%s; skipping training split collection for %s", exc, md)
            continue
        ds_path = Path(train_data_root) / f"split_{split_id}" / f"fold_{fold_id}" / "hf_dataset"
        if ds_path in seen_paths:
            continue
        if not ds_path.exists():
            logger.warning("Expected training dataset missing at %s (derived from %s)", ds_path, md)
            continue
        logger.debug("Loading training splits from %s", ds_path)
        ds = load_dataset(ds_path, fingerprint_config=fingerprint_config)
        seen_paths[ds_path] = True
        combined_frames["train"].append(ds.train)
        combined_frames["validation"].append(ds.val)
        combined_frames["test"].append(ds.test)

    concatenated: Dict[str, pd.DataFrame] = {}
    for split_name, frames in combined_frames.items():
        if frames:
            concatenated[split_name] = pd.concat(frames, ignore_index=True)
    return concatenated


def load_model_from_dir(model_dir: Path) -> BaseModel:
    """Load a model implementing the :class:`BaseModel` interface from a
    model directory.

    The loader attempts to infer the model type and instantiate the
    corresponding implementation. At present we support the
    ``XGBoostMultiEndpoint`` wrapper.
    """
    model_dir = Path(model_dir)

    # Preferred path: use run_meta.json written by trainers.
    meta_path = model_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            msg = f"Failed to parse run_meta.json in {model_dir}: {exc}"
            raise ValueError(msg) from exc
        model_type = meta.get("model_type")
        if model_type == "XGBoostMultiEndpoint":
            return XGBoostMultiEndpoint.load(str(model_dir / meta.get("model_path", "model")))

    # Backwards compatibility: infer type from legacy config.json.
    t = _detect_model_type_from_dir(model_dir)
    if t == "xgboost_multi_endpoint":
        return XGBoostMultiEndpoint.load(str(model_dir))

    raise ValueError(f"Unsupported or unknown model type at directory {model_dir}")


def load_models(model_dirs: Sequence[str]) -> List[BaseModel]:
    """Load multiple models from directories.

    Parameters
    ----------
    model_dirs : Sequence[str]
        Iterable of model directory paths.

    Returns
    -------
    List[BaseModel]
        List of instantiated models.
    """
    out: List[BaseModel] = []
    for d in model_dirs:
        mdl = load_model_from_dir(Path(d))
        out.append(mdl)
    return out


def run_ensemble_predictions_from_root(
    config: EnsemblePredictConfig,
    df_eval: Optional[pd.DataFrame] = None,
    df_blind: Optional[pd.DataFrame] = None,
) -> EnsemblePredictSummary:
    """High-level API to run ensemble predictions from a parent directory.

    This mirrors the training API style by accepting a configuration object
    and returning a summary of all relevant outputs.

    Parameters
    ----------
    config : EnsemblePredictConfig
        Configuration for the ensemble prediction run.
    df_eval : Optional[pandas.DataFrame]
        Optional labeled evaluation DataFrame.
    df_blind : Optional[pandas.DataFrame]
        Optional blind (unlabeled) DataFrame.

    Returns
    -------
    EnsemblePredictSummary
        Summary of ensemble prediction outputs.
    """

    run_dirs = discover_model_runs(config.models_root)
    if not run_dirs:
        raise ValueError(f"No model runs discovered under {config.models_root}")

    models: List[BaseModel] = []
    for d in run_dirs:
        models.append(load_model_from_dir(d))

    endpoints = models[0].endpoints

    # Eval dataset
    preds_log_eval: Optional[pd.DataFrame] = None
    preds_linear_eval: Optional[pd.DataFrame] = None
    metrics_log_eval: Optional[pd.DataFrame] = None
    metrics_linear_eval: Optional[pd.DataFrame] = None

    # Blind dataset: All model outputs
    preds_log_blind: Optional[pd.DataFrame] = None
    preds_linear_blind: Optional[pd.DataFrame] = None

    # Training datasets (train/validation/test) aggregated across model folds
    train_split_evaluations: Optional[Dict[str, SplitEvaluation]] = None

    if df_eval is not None:
        preds_log_eval, preds_linear_eval, metrics_log_eval, metrics_linear_eval = evaluate_labeled_dataset(
            models,
            df_eval,
            endpoints,
            config.agg_fn,
            n_jobs=config.n_jobs,
            fingerprint_config=config.fingerprint_config,
        )

    if config.train_data_root is not None:
        split_frames = _collect_training_splits(
            Path(config.train_data_root),
            run_dirs,
            fingerprint_config=config.fingerprint_config,
        )
        logger.info("Collected %d training splits from %s", len(split_frames), config.train_data_root)
        if split_frames:
            train_split_evaluations = {}
            for split_name, df_split in split_frames.items():
                logger.debug("Evaluating training split '%s' with %d samples", split_name, len(df_split))
                preds_log_split, preds_linear_split, metrics_log_split, metrics_linear_split = evaluate_labeled_dataset(
                    models,
                    df_split,
                    endpoints,
                    config.agg_fn,
                    n_jobs=config.n_jobs,
                    fingerprint_config=config.fingerprint_config,
                )
                train_split_evaluations[split_name] = SplitEvaluation(
                    df_true=df_split.reset_index(drop=True),
                    preds_log=preds_log_split,
                    preds_linear=preds_linear_split,
                    metrics_log=metrics_log_split,
                    metrics_linear=metrics_linear_split,
                )

    if df_blind is not None:
        preds_log_blind, preds_linear_blind = evaluate_blind_dataset(
            models,
            df_blind,
            endpoints,
            config.agg_fn,
            n_jobs=config.n_jobs,
            fingerprint_config=config.fingerprint_config,
        )

    return EnsemblePredictSummary(
        model_dirs=run_dirs,
        endpoints=list(endpoints),
        preds_log_eval=preds_log_eval,
        preds_linear_eval=preds_linear_eval,
        metrics_log_eval=metrics_log_eval,
        metrics_linear_eval=metrics_linear_eval,
        preds_log_blind=preds_log_blind,
        preds_linear_blind=preds_linear_blind,
        train_split_evaluations=train_split_evaluations,
    )


def _generate_features_for_models(
    models: List[BaseModel],
    smiles: pd.Series,
    fingerprint_config: Optional[FingerprintConfig] = None,
) -> Dict[int, np.ndarray]:
    """Prepare feature arrays per-model depending on model input type.

    Returns a mapping model_idx -> feature ndarray to be passed to
    ``model.predict``. All returned arrays are numeric numpy ndarrays
    (for fingerprints) or object arrays (for SMILES) depending on model.
    """
    logger.info(
        "Generating features for %d models with %d input SMILES and fingerprint_config=%s",
        len(models),
        len(smiles),
        fingerprint_config,
    )

    fingerprint_cache: Dict[Tuple[int, int, bool, bool], np.ndarray] = {}
    smiles_cache: Optional[np.ndarray] = None
    features: Dict[int, np.ndarray] = {}
    for i, m in tqdm(enumerate(models), total=len(models), desc="Generating features for models", dynamic_ncols=True):
        input_type = getattr(m, "input_type", "fingerprint")
        logger.debug("Preparing features for model %d with input_type %s", i, input_type)

        if input_type == "fingerprint":

            model_fp_cfg = getattr(m, "fingerprint_config", None)
            if not model_fp_cfg:
                logger.debug("Model %d missing fingerprint_config; defaulting to global or default config", i)
            elif not isinstance(model_fp_cfg, (FingerprintConfig, dict)):
                logger.warning(
                    "Model %d has unexpected fingerprint_config type %s; defaulting to global or default config",
                    i,
                    type(model_fp_cfg),
                )
                model_fp_cfg = None
            else:
                logger.debug("Model %d has fingerprint_config: %s", i, model_fp_cfg)
                model_fp_cfg = (
                    FingerprintConfig.from_mapping(model_fp_cfg) if isinstance(model_fp_cfg, dict) else model_fp_cfg
                )

            if model_fp_cfg:
                logger.debug("Using model %d fingerprint_config", i)
                fp_cfg = model_fp_cfg
            elif fingerprint_config:
                logger.debug("Using global fingerprint_config for model %d", i)
                fp_cfg = fingerprint_config
            else:
                logger.debug("Using default fingerprint_config for model %d", i)
                fp_cfg = DEFAULT_FINGERPRINT_CONFIG

            cache_key = (fp_cfg.radius, fp_cfg.n_bits, fp_cfg.use_counts, fp_cfg.include_chirality)
            if cache_key not in fingerprint_cache:
                logger.debug("Generating fingerprints for model %d with config: %s", i, fp_cfg)
                fg = MorganFingerprintGenerator(
                    radius=fp_cfg.radius,
                    count_simulation=fp_cfg.use_counts,
                    include_chirality=fp_cfg.include_chirality,
                    fp_size=fp_cfg.n_bits,
                    config=fp_cfg,
                )
                fp_df = fg.calculate_fingerprints(smiles)
                fingerprint_cache[cache_key] = fp_df.to_numpy(dtype=float)
            else:
                logger.debug("Using cached fingerprints for model %d", i)

            features[i] = fingerprint_cache[cache_key]

        elif input_type == "smiles":
            if smiles_cache is None:
                logger.debug("Generating smiles for model %d", i)
                smiles_cache = smiles.to_numpy(dtype=object).reshape(-1, 1)
            else:
                logger.debug("Using cached smiles for model %d", i)
            features[i] = smiles_cache

        else:
            raise ValueError(f"Unsupported model input_type {input_type}")

    return features


def predict_per_model(
    models: List[BaseModel],
    smiles: pd.Series,
    n_jobs: int = 1,
    fingerprint_config: Optional[FingerprintConfig] = None,
) -> np.ndarray:
    """Run predictions for each model on provided SMILES strings.

    Parameters
    ----------
    models : List[BaseModel]
        Sequence of models (must share same endpoints ordering and D).
    smiles : pandas.Series
        Series of SMILES values aligned to prediction rows.

    Returns
    -------
    numpy.ndarray
        3-D array of shape (n_models, n_samples, n_endpoints)
    """
    if not models:
        raise ValueError("`models` must contain at least one model")

    features_map = _generate_features_for_models(models, smiles, fingerprint_config=fingerprint_config)

    # Parallelize prediction across models when requested.
    pred_list: List[np.ndarray] = []
    if n_jobs != 1:

        def _predict_single(idx: int, mdl: BaseModel) -> np.ndarray:
            return mdl.predict(features_map[idx])  # type: ignore[no-any-return]

        parallel_results = Parallel(n_jobs=n_jobs)(delayed(_predict_single)(i, m) for i, m in enumerate(models))
        pred_list = [np.asarray(r) for r in parallel_results]

    else:
        for i, m in enumerate(models):
            x = features_map[i]
            pred_list.append(m.predict(x))

    return np.stack(pred_list, axis=0)


def aggregate_predictions(per_model_preds: np.ndarray, agg_fn: str = "mean") -> np.ndarray:
    """Aggregate per-model predictions into an ensemble prediction.

    Parameters
    ----------
    per_model_preds : np.ndarray
        Array of shape (n_models, N, D).
    agg_fn : str
        Aggregation function name ("mean" or "median").

    Returns
    -------
    numpy.ndarray
        Array of shape (N, D) with aggregated predictions in log10 space.
    """
    if per_model_preds.ndim != 3:
        raise ValueError("per_model_preds must be 3D (n_models, N, D)")
    if agg_fn == "mean":
        return np.nanmean(per_model_preds, axis=0)
    elif agg_fn == "median":
        return np.nanmedian(per_model_preds, axis=0)
    else:
        raise ValueError("agg_fn must be one of ['mean','median']")


def predictions_to_dataframe(
    df_input: pd.DataFrame,
    arr_ens_preds: np.ndarray,
    endpoints: Sequence[str],
) -> pd.DataFrame:
    """Build a standardized predictions DataFrame containing per-model and
    ensemble predictions for all endpoints

    The returned DataFrame contains at least: Molecule Name, SMILES and
    for each endpoint: ensemble prediction
    """
    if len(arr_ens_preds.shape) != 2:
        raise ValueError("ensemble_preds must be 2D (n_samples, n_endpoints)")
    if len(endpoints) != arr_ens_preds.shape[1]:
        raise ValueError("Mismatch between number of endpoints and ensemble predictions shape")

    df_output = df_input[["Molecule Name", "SMILES"]].copy()

    # output DataFrame
    for j, ep in enumerate(endpoints):
        df_output[ep] = arr_ens_preds[:, j]

    return df_output


def to_linear_space_array(log_array: np.ndarray, endpoints: Sequence[str]) -> np.ndarray:
    """Convert a 2D array of log10 predictions to linear space according to
    the repository convention (exponentiate all endpoints except `LogD`).
    """
    return _apply_transform_space(log_array, endpoints, space="linear")


def evaluate_dataset(
    models: List[BaseModel],
    df_input: pd.DataFrame,
    endpoints: Sequence[str],
    df_true_log: Optional[pd.DataFrame] = None,
    agg_fn: str = "mean",
    n_jobs: int = 1,
    fingerprint_config: Optional[FingerprintConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run ensemble predictions and optional evaluation on a dataset.

    Parameters
    ----------
    models : List[BaseModel]
        Sequence of models (must share same endpoints ordering and n_endpoints).
    df_input : pandas.DataFrame
        Input DataFrame containing at least "Molecule Name" and "SMILES" columns.
    endpoints : Sequence[str]
        Ordered list of endpoint names corresponding to model outputs.
    df_true_log : Optional[pandas.DataFrame]
        Optional DataFrame of true target values in log10 space for evaluation.
    agg_fn : str
        Aggregation function to combine per-model predictions ("mean" or "median").
    n_jobs : int
        Level of parallelism across models when running predictions.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, Optional[pandas.DataFrame], Optional[pandas.DataFrame]]
        Tuple of (ens_preds_log_df, ens_preds_lin_df, metrics_log_df, metrics_lin_df).
        If ``df_true_log`` is None, metrics DataFrames will be None.
    """

    # Validate endpoints
    if not models:
        raise ValueError("No models supplied")
    for m in models:
        if tuple(m.endpoints) != tuple(endpoints):
            raise ValueError("Mismatch between provided endpoint schema and model endpoints")

    smiles = df_input["SMILES"].astype(str)

    # predict outputs for all models
    model_preds_log = predict_per_model(
        models, smiles, n_jobs=n_jobs, fingerprint_config=fingerprint_config
    )  # (n_models, n_samples, n_endpoints)

    # aggregate predictions across models
    ens_preds_log = aggregate_predictions(model_preds_log, agg_fn)  # (n_samples, n_endpoints)

    # convert to linear space
    ens_preds_lin = to_linear_space_array(ens_preds_log, endpoints)  # (n_samples, n_endpoints)

    # build prediction DataFrames
    ens_preds_log_df = predictions_to_dataframe(df_input, ens_preds_log, endpoints)
    ens_preds_lin_df = predictions_to_dataframe(df_input, ens_preds_lin, endpoints)

    if df_true_log is None:
        return ens_preds_log_df, ens_preds_lin_df, None, None

    # Compute aggregated metrics in both spaces
    metrics: Dict[str, Any] = {}
    metrics["ensemble_log"] = evaluate_metrics(
        ens_preds_log_df,
        df_true_log,
        endpoints,
        transform="log",
    )  # Dict[str, Dict[str, float]], endpoint -> metric -> value
    metrics["ensemble_linear"] = evaluate_metrics(
        ens_preds_log_df,
        df_true_log,
        endpoints,
        transform="linear",
    )  # Dict[str, Dict[str, float]]

    # Compute per-model predictions and metrics
    for i, m in tqdm(
        enumerate(models),
        total=len(models),
        desc="Computing per-model predictions and metrics",
        dynamic_ncols=True,
    ):
        preds = model_preds_log[i]  # (N, D)
        for space in ["log", "linear"]:
            m_metrics = evaluate_metrics(
                predictions=predictions_to_dataframe(
                    df_input,
                    preds,
                    endpoints,
                ),
                true_values=df_true_log,
                endpoints=endpoints,
                transform=space,
            )
            metrics[f"model_{i}_{space}"] = m_metrics

    # Compute mean and standard error across models for each metric
    for space in ["log", "linear"]:
        per_model_metrics: List[Dict[str, Dict[str, float]]] = [
            cast(Dict[str, Dict[str, float]], metrics[f"model_{i}_{space}"]) for i in range(len(models))
        ]
        endpoint_names = list(per_model_metrics[0].keys())
        metric_names = list(per_model_metrics[0][endpoint_names[0]].keys())

        logger.debug("Aggregating per-model metrics for space '%s'", space)
        logger.debug("Collected per-model metrics for %d models", len(per_model_metrics))
        logger.debug("Endpoint names: %s", endpoint_names)
        logger.debug("Metric names: %s", metric_names)

        agg_mean_metrics: Dict[str, Dict[str, float]] = {}
        agg_stderr_metrics: Dict[str, Dict[str, float]] = {}
        for ep in endpoint_names:
            agg_mean_metrics[ep] = {}
            agg_stderr_metrics[ep] = {}

            for metric in metric_names:
                values = [float(m[ep][metric]) for m in per_model_metrics if not np.isnan(float(m[ep][metric]))]

                if values:
                    mean_val = float(np.nanmean(values))
                    std_err = float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))
                else:
                    logger.warning(
                        "No valid values to aggregate for metric '%s' at endpoint '%s' in space '%s'",
                        metric,
                        ep,
                        space,
                    )
                    mean_val = float("nan")
                    std_err = float("nan")

                agg_mean_metrics[ep][metric] = mean_val
                agg_stderr_metrics[ep][metric] = std_err

        metrics[f"models_agg_mean_{space}"] = agg_mean_metrics
        metrics[f"models_agg_stderr_{space}"] = agg_stderr_metrics

    # Build metrics DataFrames with columns: type, endpoint, metric, value
    def build_metrics_df(space: str) -> pd.DataFrame:
        rows = []
        for key, mdict in metrics.items():
            if not key.endswith(space):
                continue
            for ep in list(endpoints) + ["macro"]:
                d = mdict[ep]
                for k, v in d.items():
                    rows.append({"type": key, "endpoint": ep, "space": space, "metric": k, "value": v})
        return pd.DataFrame(rows)

    metrics_log_df = build_metrics_df("log")
    metrics_lin_df = build_metrics_df("linear")

    return ens_preds_log_df, ens_preds_lin_df, metrics_log_df, metrics_lin_df


def evaluate_metrics(
    predictions: pd.DataFrame,
    true_values: pd.DataFrame,
    endpoints: Sequence[str],
    transform: str = "log",
) -> eval_metrics.SplitMetrics:
    if not len(predictions) == len(true_values):
        raise ValueError("Predictions and true_values must have the same number of rows")
    if not all(ep in predictions.columns for ep in endpoints):
        raise ValueError("Not all endpoints are present in predictions DataFrame")
    if not all(ep in true_values.columns for ep in endpoints):
        raise ValueError("Not all endpoints are present in true_values DataFrame")

    # resort true values to match predictions
    true_values = true_values.loc[predictions.index]
    if not all(true_values["SMILES"] == predictions["SMILES"]):
        raise ValueError("SMILES in predictions and true_values do not match")

    # Cast to explicit numeric arrays of float64 to satisfy numpy typing
    # and avoid ambiguous 'object' dtype that can confuse mypy's ufunc
    # overload resolution.
    y_true: NDArray[np.float64] = np.asarray(true_values[list(endpoints)].to_numpy(dtype=float), dtype=float)
    y_pred: NDArray[np.float64] = np.asarray(predictions[list(endpoints)].to_numpy(dtype=float), dtype=float)
    mask = (~np.isnan(y_true)).astype(int)
    mask_counts = mask.sum(axis=0)

    # check finite
    if not np.all(np.isfinite(y_true[mask == 1])):
        raise ValueError("Non-finite values found in true values")
    if not np.all(np.isfinite(y_pred[mask == 1])):
        raise ValueError("Non-finite values found in predicted values")

    if transform == "linear":
        y_true = to_linear_space_array(y_true, endpoints)
        y_pred = to_linear_space_array(y_pred, endpoints)

    elif transform != "log":
        raise ValueError(f"Unsupported transform '{transform}'")

    invalid_indices = [j for j, count in enumerate(mask_counts) if count == 0]
    if invalid_indices:
        invalid_set = set(invalid_indices)
        for j in invalid_indices:
            logger.error("No valid true values found for endpoint '%s'", endpoints[j])

        valid_indices = [j for j, count in enumerate(mask_counts) if count > 0]
        nan_metrics: eval_metrics.EndpointMetrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "R2": float("nan"),
            "pearson_r2": float("nan"),
            "spearman_rho2": float("nan"),
            "kendall_tau": float("nan"),
        }

        # metrics_subset will always be a mapping from endpoint -> EndpointMetrics
        metrics_subset: eval_metrics.SplitMetrics
        if valid_indices:
            valid_endpoints = [endpoints[j] for j in valid_indices]
            metrics_subset = eval_metrics.compute_metrics(
                y_true[:, valid_indices],
                y_pred[:, valid_indices],
                mask[:, valid_indices],
                valid_endpoints,
            )
        else:
            metrics_subset = cast(eval_metrics.SplitMetrics, {"macro": cast(eval_metrics.EndpointMetrics, nan_metrics)})

        metrics: eval_metrics.SplitMetrics = {}
        for idx, ep in enumerate(endpoints):
            if idx in invalid_set:
                metrics[ep] = nan_metrics
            else:
                metrics[ep] = cast(eval_metrics.EndpointMetrics, metrics_subset[ep])
        metrics["macro"] = cast(eval_metrics.EndpointMetrics, metrics_subset.get("macro", nan_metrics))
    else:
        metrics = eval_metrics.compute_metrics(y_true, y_pred, mask, endpoints)

    for ep, mdict in metrics.items():
        for k, v in mdict.items():
            # Guard against non-numeric objects; convert to float for isfinite check.
            if isinstance(v, (int, float, np.floating)):
                _is_finite = bool(np.isfinite(v))
            else:
                _is_finite = False
            if not _is_finite:
                try:
                    y_true_vals = cast(
                        NDArray[np.float64],
                        y_true[mask[:, endpoints.index(ep)] == 1, endpoints.index(ep)],
                    )
                    y_pred_vals = cast(
                        NDArray[np.float64],
                        y_pred[mask[:, endpoints.index(ep)] == 1, endpoints.index(ep)],
                    )
                except (ValueError, IndexError):
                    y_true_vals = np.array([float("nan")], dtype=float)
                    y_pred_vals = np.array([float("nan")], dtype=float)

                if y_true_vals.size == 0:
                    y_true_vals = np.array([float("nan")], dtype=float)
                    y_pred_vals = np.array([float("nan")], dtype=float)

                logger.debug(
                    "y_true domain for endpoint '%s': min=%s, max=%s",
                    ep,
                    np.min(y_true_vals),
                    np.max(y_true_vals),
                )
                logger.debug(
                    "y_pred domain for endpoint '%s': min=%s, max=%s",
                    ep,
                    np.min(y_pred_vals),
                    np.max(y_pred_vals),
                )
                logger.warning(
                    "Non-finite metric '%s' for endpoint '%s' in space '%s'. True domain: min=%s, max=%s",
                    k,
                    ep,
                    transform,
                    np.min(y_true_vals),
                    np.max(y_true_vals),
                )

    return metrics


def evaluate_labeled_dataset(
    models: List[BaseModel],
    df_eval: pd.DataFrame,
    endpoints: Sequence[str],
    agg_fn: str = "mean",
    n_jobs: int = 1,
    fingerprint_config: Optional[FingerprintConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run ensemble evaluation for a labeled dataset.

    Parameters
    ----------
    models : List[BaseModel]
        Sequence of models (must share same endpoints ordering and n_endpoints).
    df_eval : pandas.DataFrame
        Input DataFrame containing at least "Molecule Name", "SMILES" and
        endpoint columns with true target values in log10 space.
    endpoints : Sequence[str]
        Ordered list of endpoint names corresponding to model outputs.
    agg_fn : str
        Aggregation function to combine per-model predictions ("mean" or "median").
    n_jobs : int
        Level of parallelism across models when running predictions.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Tuple of (ens_preds_log_df, ens_preds_lin_df, metrics_log_df, metrics_lin_df).
    """
    # Extract true values DataFrame in log space
    df_true_log = df_eval[["Molecule Name", "SMILES"] + list(endpoints)].copy()
    for ep in endpoints:
        df_true_log[ep] = pd.to_numeric(df_true_log[ep], errors="coerce")

    ens_preds_log, ens_preds_lin, metrics_log, metrics_linear = evaluate_dataset(
        models,
        df_eval,
        endpoints,
        df_true_log,
        agg_fn,
        n_jobs=n_jobs,
        fingerprint_config=fingerprint_config,
    )

    assert metrics_log is not None and metrics_linear is not None

    return ens_preds_log, ens_preds_lin, metrics_log, metrics_linear


def evaluate_blind_dataset(
    models: List[BaseModel],
    df_blind: pd.DataFrame,
    endpoints: Sequence[str],
    agg_fn: str = "mean",
    n_jobs: int = 1,
    fingerprint_config: Optional[FingerprintConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run ensemble inference on a blind (unlabeled) dataset.

    Parameters
    ----------
    models : List[BaseModel]
        Sequence of models (must share same endpoints ordering and n_endpoints).
    df_blind : pandas.DataFrame
        Input DataFrame containing at least "Molecule Name" and "SMILES" columns.
    endpoints : Sequence[str]
        Ordered list of endpoint names corresponding to model outputs.
    agg_fn : str
        Aggregation function to combine per-model predictions ("mean" or "median").
    n_jobs : int
        Level of parallelism across models when running predictions.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        Tuple of (ens_preds_log_df, ens_preds_lin_df)
    """
    ens_preds_log, ens_preds_lin, _, _ = evaluate_dataset(
        models,
        df_blind,
        endpoints,
        df_true_log=None,
        agg_fn=agg_fn,
        n_jobs=n_jobs,
        fingerprint_config=fingerprint_config,
    )
    return ens_preds_log, ens_preds_lin


__all__ = [
    "load_model_from_dir",
    "load_models",
    "discover_model_runs",
    "EnsemblePredictConfig",
    "EnsemblePredictSummary",
    "SplitEvaluation",
    "run_ensemble_predictions_from_root",
    "predict_per_model",
    "aggregate_predictions",
    "predictions_to_dataframe",
    "evaluate_labeled_dataset",
    "evaluate_blind_dataset",
    "aggregate_metrics_by_endpoint",
    "log_metrics_to_linear",
]
