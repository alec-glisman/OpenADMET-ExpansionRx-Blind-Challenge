"""Ray multi‑dataset training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import datetime
import logging
import multiprocessing
import os

import pandas as pd
import ray
import mlflow

from .model_trainer import BaseModelTrainer, FeaturizationMethod
from .utils import infer_split_metadata
from ..mlflow_utils import flatten_metrics, flatten_params, set_mlflow_tracking

logger = logging.getLogger(__name__)


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
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    base_params: Optional[Dict[str, object]] = None,
    cli_params: Optional[Dict[str, object]] = None,
    worker_thread_limit: Optional[int] = None,
) -> Tuple[str, Dict[str, object]]:
    hf_dir = Path(hf_path)
    root = Path(root_dir)
    from admet.data.load import load_dataset as _load_dataset

    meta = infer_split_metadata(hf_dir, root)
    rel_key = str(meta.get("relative_path", hf_dir.name))
    start_ts = datetime.datetime.now()
    start_time = start_ts.isoformat()
    try:
        from admet.logging import configure_logging

        if log_level or log_file or log_json:
            configure_logging(level=log_level or "INFO", file=log_file, structured=log_json)
    except Exception:  # noqa: BLE001
        pass
    run_metrics: Optional[Dict[str, object]] = None
    summary: Optional[object] = None
    status = "error"
    error: Optional[str] = None
    end_time = start_time
    duration_seconds = 0.0
    out_dir: Optional[Path] = None
    local_trainer_kwargs = dict(trainer_kwargs)

    set_mlflow_tracking(tracking_uri, experiment_name)
    tags = {"dataset.relative_path": rel_key, "featurization": featurization.value}
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id

    if worker_thread_limit:
        thread_str = str(max(1, int(worker_thread_limit)))
        for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            os.environ[env_var] = thread_str
        model_params = dict(local_trainer_kwargs.get("model_params") or {})
        model_params.setdefault("n_jobs", int(worker_thread_limit))
        local_trainer_kwargs["model_params"] = model_params

    try:
        with mlflow.start_run(run_name=rel_key, tags=tags):
            if base_params:
                mlflow.log_params(base_params)
            if cli_params:
                mlflow.log_params(cli_params)

            dataset_params = flatten_params(meta, prefix="dataset")
            dataset_params["featurization"] = featurization.value
            if seed is not None:
                dataset_params["seed"] = seed
            mlflow.log_params(dataset_params)

            logger.info("Loading dataset '%s' with featurization=%s", hf_dir, featurization)
            if featurization == FeaturizationMethod.MORGAN_FP:
                if n_fingerprint_bits:
                    dataset = _load_dataset(hf_dir, n_fingerprint_bits=n_fingerprint_bits)
                else:
                    dataset = _load_dataset(hf_dir)
            else:
                dataset = _load_dataset(hf_dir)
            if output_root is not None:
                base = Path(output_root)
                cluster = str(meta.get("cluster", "unknown_method"))
                split = str(meta.get("split", "unknown_split"))
                fold = str(meta.get("fold", "unknown_fold"))
                out_dir = base / cluster / f"split_{split}" / f"fold_{fold}"
            if seed is not None:
                local_trainer_kwargs.setdefault("seed", seed)
            trainer = trainer_cls(**local_trainer_kwargs)
            try:
                run_metrics, summary = trainer.fit(
                    dataset,
                    sample_weight_mapping=sample_weight_mapping,
                    early_stopping_rounds=early_stopping_rounds,
                    output_dir=out_dir,
                    dry_run=dry_run,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                status = "error"
                logger.exception("Dataset training failed for %s: %s", rel_key, exc)
            else:
                status = "ok"
                if dry_run:
                    status = "skipped"
                    run_metrics = None
                    summary = None
                if run_metrics is None and not dry_run:
                    status = "error"
                elif isinstance(run_metrics, dict):
                    expected_splits = {"train", "validation", "test"}
                    present = set(run_metrics.keys())
                    if not expected_splits.issubset(present):
                        status = "partial"
                    else:
                        for split in expected_splits:
                            if not isinstance(run_metrics.get(split, {}), dict) or "macro" not in run_metrics[split]:
                                status = "partial"
                                break
            end_ts = datetime.datetime.now()
            end_time = end_ts.isoformat()
            duration_seconds = (end_ts - start_ts).total_seconds()
            if max_duration_seconds is not None and duration_seconds > max_duration_seconds:
                status = "timeout"
            mlflow.set_tag("status", status)
            if error:
                mlflow.set_tag("error", error)
            mlflow.log_metric("duration_seconds", duration_seconds)
            if run_metrics:
                mlflow.log_metrics(flatten_metrics(run_metrics))
            if out_dir and out_dir.exists():
                mlflow.log_artifacts(str(out_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dataset training failed for %s: %s", rel_key, exc)
        error = str(exc)
        end_ts = datetime.datetime.now()
        end_time = end_ts.isoformat()
        duration_seconds = (end_ts - start_ts).total_seconds()
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                set_mlflow_tracking(tracking_uri, experiment_name)
                with mlflow.start_run(run_name=rel_key, tags=tags):
                    mlflow.set_tag("status", "error")
                    mlflow.set_tag("error", error)
                    mlflow.log_metric("duration_seconds", duration_seconds)
            else:
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error", error)
                mlflow.log_metric("duration_seconds", duration_seconds)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to log MLflow error state", exc_info=True)

    return rel_key, {
        "run_metrics": run_metrics,
        "summary": summary,
        "meta": meta,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "error": error,
    }


class BaseEnsembleTrainer:
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
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        base_params: Optional[Dict[str, object]] = None,
        cli_params: Optional[Dict[str, object]] = None,
        worker_thread_limit: Optional[int] = None,
    ) -> Dict[str, Dict[str, object]]:
        if output_root is None:
            output_root = Path("models")
        hf_paths = list(self.discover_datasets(root))
        if not hf_paths:
            raise ValueError(f"No 'hf_dataset' directories found under {root}.")
        logger.info("Discovered %d hf_dataset directories under %s", len(hf_paths), root)
        cpu_count = num_cpus or multiprocessing.cpu_count()
        worker_thread_limit = worker_thread_limit or max(1, cpu_count // max(1, min(len(hf_paths), cpu_count)))
        if ray.is_initialized():
            logger.info("Reusing existing Ray runtime (num_cpus=%s)", ray.cluster_resources().get("CPU"))
        else:
            if ray_address and ray_address.lower() != "local":
                logger.info("Connecting to existing Ray cluster at %s", ray_address)
                ray.init(address=ray_address, ignore_reinit_error=True)
            else:
                logger.info("Starting local Ray runtime with %d CPUs", cpu_count)
                ray.init(num_cpus=cpu_count, ignore_reinit_error=True)
        sorted_hf_paths = sorted(hf_paths, key=lambda p: str(p.relative_to(root)))
        base_seed = seed if seed is not None else 0
        try:
            from admet.logging import get_logging_config

            log_cfg = get_logging_config()
        except Exception:  # noqa: BLE001
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
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    parent_run_id=parent_run_id,
                    base_params=base_params,
                    cli_params=cli_params,
                    worker_thread_limit=worker_thread_limit,
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
        summary_rows: List[Dict[str, object]] = []
        success_count = 0
        failure_count = 0
        for rel_key, payload in aggregated.items():
            status = str(payload.get("status", "ok"))
            success_count += status == "ok"
            failure_count += status != "ok"
            run_metrics = payload.get("run_metrics") or {}  # type: ignore[assignment]
            meta = payload.get("meta") or {}
            start_time = payload.get("start_time")
            end_time = payload.get("end_time")
            duration_seconds = payload.get("duration_seconds")
            for split_name in ["train", "validation", "test"]:
                split_metrics = run_metrics.get(split_name, {})  # type: ignore[assignment]
                macro = split_metrics.get("macro", {})
                row: Dict[str, object] = {"dataset": rel_key, "split": split_name, "status": status}
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
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "metrics_summary.csv").write_text(summary_df.to_csv(index=False))
            (output_root / "metrics_summary.json").write_text(summary_df.to_json(orient="records", indent=2))
            logger.info("Ray training summary: %d succeeded, %d failed", success_count, failure_count)

        # TODO: Run predictions, compute metrics, and visualize performance of the entire ensemble (report mean and stderr)

        # TODO: Run predictions (log and linear) on blind test set and save results

        return aggregated


__all__ = ["BaseEnsembleTrainer"]
