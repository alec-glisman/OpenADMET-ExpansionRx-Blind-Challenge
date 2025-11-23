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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional
import json
import logging

import numpy as np
import pandas as pd

from admet.model.base import BaseModel
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.data.fingerprinting import MorganFingerprintGenerator
from admet.evaluate import metrics as eval_metrics
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
    """

    models_root: Path
    eval_csv: Optional[Path] = None
    blind_csv: Optional[Path] = None
    agg_fn: str = "mean"
    n_jobs: int = 1


@dataclass
class EnsemblePredictSummary:
    """Summary of ensemble prediction outputs for eval and blind datasets."""

    model_dirs: List[Path]
    preds_log_eval: Optional[pd.DataFrame]
    preds_linear_eval: Optional[pd.DataFrame]
    metrics_log: Optional[pd.DataFrame]
    metrics_linear: Optional[pd.DataFrame]
    preds_log_blind: Optional[pd.DataFrame]
    preds_linear_blind: Optional[pd.DataFrame]


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
    """

    run_dirs = discover_model_runs(config.models_root)
    if not run_dirs:
        raise ValueError(f"No model runs discovered under {config.models_root}")

    models: List[BaseModel] = []
    for d in run_dirs:
        models.append(load_model_from_dir(d))

    endpoints = models[0].endpoints
    preds_log_eval: Optional[pd.DataFrame] = None
    preds_lin_eval: Optional[pd.DataFrame] = None
    metrics_log: Optional[pd.DataFrame] = None
    metrics_linear: Optional[pd.DataFrame] = None
    preds_log_blind: Optional[pd.DataFrame] = None
    preds_lin_blind: Optional[pd.DataFrame] = None

    if df_eval is not None:
        (
            preds_log_eval,
            preds_lin_eval,
            metrics_log,
            metrics_linear,
            _,
        ) = evaluate_labeled_dataset(
            models,
            df_eval,
            endpoints,
            config.agg_fn,
            n_jobs=config.n_jobs,
        )

    if df_blind is not None:
        preds_log_blind, preds_lin_blind = evaluate_blind_dataset(
            models,
            df_blind,
            endpoints,
            config.agg_fn,
            n_jobs=config.n_jobs,
        )

    return EnsemblePredictSummary(
        model_dirs=run_dirs,
        preds_log_eval=preds_log_eval,
        preds_linear_eval=preds_lin_eval,
        metrics_log=metrics_log,
        metrics_linear=metrics_linear,
        preds_log_blind=preds_log_blind,
        preds_linear_blind=preds_lin_blind,
    )


def _generate_features_for_models(models: List[BaseModel], smiles: pd.Series) -> dict:
    """Prepare feature arrays per-model depending on model input type.

    Returns a mapping model_idx -> feature ndarray to be passed to
    ``model.predict``. All returned arrays are numeric numpy ndarrays
    (for fingerprints) or object arrays (for SMILES) depending on model.
    """
    features = {}
    for i, m in enumerate(models):
        if getattr(m, "input_type", "fingerprint") == "fingerprint":
            # Use the recorded feature dimension if available, else default 2048
            fp_size = getattr(m, "n_features_", None) or 2048
            fg = MorganFingerprintGenerator(fp_size=fp_size)
            fp_df = fg.calculate_fingerprints(smiles)
            features[i] = fp_df.to_numpy(dtype=float)
        elif getattr(m, "input_type", "fingerprint") == "smiles":
            # Predictors that operate on raw SMILES may accept a 1-D ndarray
            # of strings (object dtype) or a 2D array; we convert to 2D
            features[i] = smiles.to_numpy(dtype=object).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported model input_type {getattr(m, 'input_type', None)}")
    return features


def predict_per_model(
    models: List[BaseModel],
    smiles: pd.Series,
    *,
    n_jobs: int = 1,
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

    features_map = _generate_features_for_models(models, smiles)

    # Parallelize prediction across models when requested.
    pred_list: List[np.ndarray] = []
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            def _predict_single(idx: int, mdl: BaseModel) -> np.ndarray:
                return mdl.predict(features_map[idx])  # type: ignore[no-any-return]

            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_predict_single)(i, m) for i, m in enumerate(models)
            )
            pred_list = [np.asarray(r) for r in parallel_results]
        except ImportError:
            # Fallback to serial execution if joblib is unavailable.
            for i, m in enumerate(models):
                X = features_map[i]
                pred_list.append(m.predict(X))
    else:
        for i, m in enumerate(models):
            X = features_map[i]
            pred_list.append(m.predict(X))

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
    df: pd.DataFrame,
    per_model_preds: np.ndarray,
    ensemble_preds: np.ndarray,
    endpoints: Sequence[str],
    prefix: str = "pred",
) -> pd.DataFrame:
    """Build a standardized predictions DataFrame containing per-model and
    ensemble predictions for all endpoints in log space.

    The returned DataFrame contains at least: Molecule Name, SMILES and
    for each endpoint: `pred_{endpoint}_{model_name}_log` and
    `pred_{endpoint}_ensemble_log` columns.
    """
    out = df[["Molecule Name", "SMILES"]].copy()
    n_models = per_model_preds.shape[0]
    n_rows = per_model_preds.shape[1]
    assert n_rows == len(out)
    for j, ep in enumerate(endpoints):
        for i in range(n_models):
            col = f"{prefix}_{ep}_model{i}_log"
            out[col] = per_model_preds[i, :, j]
        out[f"{prefix}_{ep}_ensemble_log"] = ensemble_preds[:, j]
    return out


def to_linear_space_array(log_array: np.ndarray, endpoints: Sequence[str]) -> np.ndarray:
    """Convert a 2D array of log10 predictions to linear space according to
    the repository convention (exponentiate all endpoints except `LogD`).
    """
    return _apply_transform_space(log_array, endpoints, space="linear")


def evaluate_labeled_dataset(
    models: List[BaseModel],
    df_eval: pd.DataFrame,
    endpoints: Sequence[str],
    agg_fn: str = "mean",
    *,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run ensemble evaluation for a labeled dataset.

    Returns a tuple of prediction and metric DataFrames for evaluation.
    """
    # Validate endpoints
    if not models:
        raise ValueError("No models supplied")
    for m in models:
        if tuple(m.endpoints) != tuple(endpoints):
            raise ValueError("Mismatch between provided endpoint schema and model endpoints")

    smiles = df_eval["SMILES"].astype(str)
    per_model_preds = predict_per_model(models, smiles, n_jobs=n_jobs)  # (n_models, N, D)
    ens_preds_log = aggregate_predictions(per_model_preds, agg_fn)
    ens_preds_lin = to_linear_space_array(ens_preds_log, endpoints)

    preds_log_df = predictions_to_dataframe(df_eval, per_model_preds, ens_preds_log, endpoints)
    # For linear predictions, convert both per-model and ensemble
    n_models = per_model_preds.shape[0]
    per_model_preds_lin = per_model_preds.copy()
    per_model_preds_lin = per_model_preds_lin.transpose(0, 2, 1)  # (n_models, D, N)
    # Apply linear transform per endpoint and transpose back
    for i in range(n_models):
        per_model_preds_lin[i] = to_linear_space_array(per_model_preds[i], endpoints)
    preds_lin_df = predictions_to_dataframe(
        df_eval, per_model_preds_lin, ens_preds_lin, endpoints, prefix="pred_linear"
    )

    # Construct masks and ground truth
    Y_true_log = df_eval[list(endpoints)].to_numpy(dtype=float)
    mask = (~np.isnan(Y_true_log)).astype(int)

    log_metrics = eval_metrics.compute_metrics(Y_true_log, ens_preds_log, mask, endpoints)
    lin_metrics = eval_metrics.compute_metrics(
        to_linear_space_array(Y_true_log, endpoints),
        to_linear_space_array(ens_preds_log, endpoints),
        mask,
        endpoints,
    )

    # Create metric dataframes for easy saving
    def metrics_dict_to_df(mdict, space: str):
        rows = []
        for ep in list(endpoints) + ["macro"]:
            d = mdict[ep]
            for k, v in d.items():
                rows.append({"endpoint": ep, "space": space, "metric": k, "value": v})
        return pd.DataFrame(rows)

    metrics_log_df = metrics_dict_to_df(log_metrics, "log")
    metrics_lin_df = metrics_dict_to_df(lin_metrics, "linear")

    # Per-model metrics
    per_model_rows = []
    for i, m in enumerate(models):
        preds = per_model_preds[i]  # (N, D)
        m_metrics_log = eval_metrics.compute_metrics(Y_true_log, preds, mask, endpoints)
        m_metrics_lin = eval_metrics.compute_metrics(
            to_linear_space_array(Y_true_log, endpoints),
            to_linear_space_array(preds, endpoints),
            mask,
            endpoints,
        )
        for ep in list(endpoints) + ["macro"]:
            for k, v in m_metrics_log[ep].items():
                per_model_rows.append(
                    {"endpoint": ep, "space": "log", "model": f"model_{i}", "metric": k, "value": v}
                )
            for k, v in m_metrics_lin[ep].items():
                per_model_rows.append(
                    {"endpoint": ep, "space": "linear", "model": f"model_{i}", "metric": k, "value": v}
                )
    # Ensemble metrics included too
    for ep in list(endpoints) + ["macro"]:
        for k, v in log_metrics[ep].items():
            per_model_rows.append(
                {"endpoint": ep, "space": "log", "model": "ensemble", "metric": k, "value": v}
            )
        for k, v in lin_metrics[ep].items():
            per_model_rows.append(
                {"endpoint": ep, "space": "linear", "model": "ensemble", "metric": k, "value": v}
            )

    per_model_vs_ensemble_df = pd.DataFrame(per_model_rows)

    return preds_log_df, preds_lin_df, metrics_log_df, metrics_lin_df, per_model_vs_ensemble_df


def evaluate_blind_dataset(
    models: List[BaseModel],
    df_blind: pd.DataFrame,
    endpoints: Sequence[str],
    agg_fn: str = "mean",
    *,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run ensemble inference on a blind (unlabeled) dataset.

    Returns: (preds_log_df, preds_lin_df)
    """
    if not models:
        raise ValueError("No models supplied")
    for m in models:
        if tuple(m.endpoints) != tuple(endpoints):
            raise ValueError("Mismatch between provided endpoint schema and model endpoints")

    smiles = df_blind["SMILES"].astype(str)
    per_model_preds = predict_per_model(models, smiles, n_jobs=n_jobs)
    ens_preds_log = aggregate_predictions(per_model_preds, agg_fn)
    ens_preds_lin = to_linear_space_array(ens_preds_log, endpoints)

    preds_log_df = predictions_to_dataframe(df_blind, per_model_preds, ens_preds_log, endpoints)
    n_models = per_model_preds.shape[0]
    per_model_preds_lin = per_model_preds.copy()
    for i in range(n_models):
        per_model_preds_lin[i] = to_linear_space_array(per_model_preds[i], endpoints)
    preds_lin_df = predictions_to_dataframe(
        df_blind, per_model_preds_lin, ens_preds_lin, endpoints, prefix="pred_linear"
    )

    return preds_log_df, preds_lin_df


__all__ = [
    "load_model_from_dir",
    "load_models",
    "discover_model_runs",
    "EnsemblePredictConfig",
    "EnsemblePredictSummary",
    "run_ensemble_predictions_from_root",
    "predict_per_model",
    "aggregate_predictions",
    "predictions_to_dataframe",
    "evaluate_labeled_dataset",
    "evaluate_blind_dataset",
]
