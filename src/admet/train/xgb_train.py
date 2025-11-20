"""Training orchestration for XGBoost multi-endpoint model (initial version)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple
import logging
import multiprocessing

import numpy as np
import pandas as pd
import ray

from admet.data.load import LoadedDataset
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.utils import set_global_seeds
from admet.evaluate.metrics import compute_metrics_log_and_linear
from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars

logger = logging.getLogger(__name__)


def _extract_features(df, fingerprint_cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, fingerprint_cols].to_numpy(dtype=float)


def _extract_targets(df, endpoints: Sequence[str]) -> np.ndarray:
    return df[endpoints].to_numpy(dtype=float)


def _target_mask(y: np.ndarray) -> np.ndarray:
    return (~np.isnan(y)).astype(int)


def _infer_split_metadata(hf_path: Path, root: Path) -> Dict[str, str]:
    """Infer split/fold/cluster metadata from an ``hf_dataset`` path.

    This inspects the relative path from ``root`` to the provided
    ``hf_dataset`` directory and extracts best-effort metadata for
    logging and output directory naming.
    """

    # Prefer a relative path provided the `root` is a parent of the HF dataset
    try:
        rel = hf_path.relative_to(root)
        rel_parts: List[str] = [p for p in rel.parts if p]
        meta: Dict[str, str] = {"relative_path": str(rel)}
    except Exception:
        rel = None
        rel_parts = []
        meta = {"relative_path": str(hf_path)}

    # Add the full/unqualified path for cases where `root` is too deep
    # to contain cluster metadata. This enables the caller to determine
    # cluster methods and quality tags from the full filesystem layout.
    try:
        # Normalize the full path, but avoid failing if path doesn't exist
        full_path = str(hf_path.resolve())
    except Exception:
        full_path = str(hf_path)
    full_parts: List[str] = [p for p in Path(full_path).parts if p and p != "/"]
    meta["full_path"] = full_path

    # Heuristic extraction for common layout:
    #   /path/to/assets/<quality>/<cluster_method>/split_<i>/fold_<j>/hf_dataset
    for part in rel_parts:
        if part.startswith("split_"):
            meta["split"] = part.replace("split_", "")
        elif part.startswith("fold_"):
            meta["fold"] = part.replace("fold_", "")

    if len(rel_parts) > 3:
        meta["cluster"] = f"{rel_parts[-4]}-{rel_parts[-3]}"
    elif full_parts:
        # Similar fallback for cluster: take two segments prior to split if
        # available (e.g., 'high_quality/random_cluster').
        for i, part in enumerate(full_parts):
            if part.startswith("split_") and i > 1:
                meta["cluster"] = f"{full_parts[i - 2]}/{full_parts[i - 1]}"
                break

    print(meta)

    logger.info("Inferred metadata for %s: %s", rel, meta)

    return meta


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
        "validation": compute_metrics_log_and_linear(Y_val, pred_val, mask_val, endpoints),
        "test": compute_metrics_log_and_linear(Y_test, pred_test, mask_test, endpoints),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir / "model"))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # get number of CPUs for parallel plotting
        n_cpus = multiprocessing.cpu_count()
        logger.info(f"Using {n_cpus} CPU cores for plotting.")

        # Compose dicts for plotting utilities
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
    return metrics


@ray.remote
def _train_single_xgb_remote(
    hf_path: str,
    root_dir: str,
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_root: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, object]]:
    """Ray-remote wrapper to train a single XGBoost model on one HF dataset.

    Parameters are serialized-friendly wrappers around :func:`train_xgb_models`.
    Returns the relative hf-dataset path (key) and the metrics dictionary.
    """

    hf_dir = Path(hf_path)
    root = Path(root_dir)
    from admet.data.load import load_dataset  # local import for Ray workers

    meta = _infer_split_metadata(hf_dir, root)
    rel_key = meta.get("relative_path", hf_dir.name)

    dataset = load_dataset(hf_dir)

    out_dir: Optional[Path] = None
    if output_root is not None:
        base = Path(output_root)
        # Compose hierarchical output path encoding cluster, split, and fold
        cluster = meta.get("cluster", "unknown_method")
        split = meta.get("split", "unknown_split")
        fold = meta.get("fold", "unknown_fold")
        out_dir = base / cluster / f"split_{split}" / f"fold_{fold}"

    metrics = train_xgb_models(
        dataset=dataset,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        sample_weight_mapping=sample_weight_mapping,
        output_dir=out_dir,
        seed=seed,
    )

    return rel_key, {"metrics": metrics, "meta": meta}


def train_xgb_models_ray(
    root: Path,
    *,
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_root: Optional[Path] = None,
    seed: Optional[int] = None,
    num_cpus: Optional[int] = None,
    ray_address: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Train XGBoost models for all discovered HF datasets using Ray.

    The function searches ``root`` recursively for directories named
    ``hf_dataset`` and launches a Ray task for each directory. Each task
    trains a model, writes artifacts/figures mirroring the split/fold/
    cluster method layout, and returns its metrics.
    """

    if output_root is None:
        output_root = Path("xgb_artifacts")

    hf_paths: List[Path] = [p for p in root.rglob("hf_dataset") if p.is_dir()]
    if not hf_paths:
        raise ValueError(f"No 'hf_dataset' directories found under {root}.")

    logger.info("Discovered %d hf_dataset directories under %s", len(hf_paths), root)

    if not ray.is_initialized():
        if ray_address and ray_address.lower() != "local":
            logger.info("Connecting to existing Ray cluster at %s", ray_address)
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            # If num_cpus is not provided, let Ray use all available cores.
            cpu_count = num_cpus or multiprocessing.cpu_count()
            logger.info("Starting local Ray runtime with %d CPUs", cpu_count)
            ray.init(num_cpus=cpu_count, ignore_reinit_error=True)

    # Deterministic per-model seeds derived from base seed and sorted paths.
    sorted_hf_paths = sorted(hf_paths, key=lambda p: str(p.relative_to(root)))
    base_seed = seed if seed is not None else 0

    remote_tasks = []
    for idx, hf_path in enumerate(sorted_hf_paths):
        model_seed = base_seed + idx
        rel = hf_path.relative_to(root)
        logger.info("Scheduling training for %s with seed %d", rel, model_seed)
        remote_tasks.append(
            _train_single_xgb_remote.remote(
                str(hf_path),
                str(root),
                model_params=model_params,
                early_stopping_rounds=early_stopping_rounds,
                sample_weight_mapping=sample_weight_mapping,
                output_root=str(output_root),
                seed=model_seed,
            )
        )

    # Track progress as Ray tasks complete.
    remaining = set(remote_tasks)
    results: List[Tuple[str, Dict[str, object]]] = []
    total = len(remote_tasks)
    completed = 0
    logger.info("Starting Ray training for %d models", total)
    while remaining:
        # Number of tasks currently executing or queued but not yet finished
        in_progress = len(remaining)
        logger.info(
            "Progress: %d/%d completed, %d in progress",
            completed,
            total,
            in_progress,
        )

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

    aggregated: Dict[str, Dict[str, object]] = {rel_key: payload for rel_key, payload in results}

    # Build and persist a summary of macro metrics across all datasets.
    summary_rows: List[Dict[str, object]] = []
    for rel_key, payload in aggregated.items():
        metrics = payload.get("metrics", {})  # type: ignore[assignment]
        meta = payload.get("meta", {})  # type: ignore[assignment]
        for split_name in ["train", "validation", "test"]:
            split_metrics = metrics.get(split_name, {})  # type: ignore[assignment]
            macro = split_metrics.get("macro", {})
            row: Dict[str, object] = {
                "dataset": rel_key,
                "split": split_name,
            }
            # include metadata such as cluster_method, split, fold if present
            if isinstance(meta, dict):
                for k, v in meta.items():
                    row[f"meta_{k}"] = v
            # flatten macro metrics one level
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

    return aggregated


__all__ = ["train_xgb_models", "train_xgb_models_ray"]
