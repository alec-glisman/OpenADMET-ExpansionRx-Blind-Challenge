"""Robust MLflow callback for Ray Tune HPO.

This module provides a wrapper around Ray's MLflowLoggerCallback that handles
common errors gracefully, particularly when ASHA scheduler terminates trials
and MLflow runs become inactive or deleted.

It also provides utilities for backfilling MLflow with metrics from Ray Tune's
result storage after HPO completes.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from typing import Any

import mlflow
from mlflow.entities import Metric
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.tune.result import TRAINING_ITERATION
from ray.tune.result_grid import ResultGrid

logger = logging.getLogger(__name__)


class AsyncBatchedMLflowCallback(Callback):
    """High-performance async MLflow callback with batched metric logging.

    This callback eliminates Ray Tune performance bottlenecks by:
    1. Batching metrics instead of logging one-by-one (single HTTP request)
    2. Using a background thread for async logging (non-blocking)
    3. Gracefully handling MLflow errors for terminated trials

    Attributes:
        _tracking_uri: MLflow tracking server URI
        _experiment_name: MLflow experiment name
        _tags: Tags to apply to MLflow runs
        _trial_runs: Mapping of trial -> run_id
        _failed_trials: Set of trial IDs that have failed logging
        _log_queue: Queue for async metric batching
        _worker_thread: Background thread for async logging
    """

    # Batch size and timing for async logging
    BATCH_SIZE = 50  # Max metrics per batch
    BATCH_TIMEOUT_S = 2.0  # Max seconds to wait before flushing batch

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        save_artifact: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize the async batched MLflow callback.

        Args:
            tracking_uri: MLflow tracking URI (e.g., http://127.0.0.1:8084)
            experiment_name: Name of the MLflow experiment
            save_artifact: Whether to save artifacts (ignored for performance)
            tags: Tags to apply to MLflow runs
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._tags = tags or {}
        self._save_artifact = save_artifact

        self._trial_runs: dict[Trial, str] = {}
        self._failed_trials: set[str] = set()
        self._mlflow_client: mlflow.MlflowClient | None = None

        # Async logging infrastructure
        self._log_queue: queue.Queue[tuple[str, list[Metric]] | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Initialize MLflow client and start background worker."""
        try:
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)

            if self._experiment_name:
                mlflow.set_experiment(self._experiment_name)

            self._mlflow_client = mlflow.MlflowClient()

            # Start background worker thread for async logging
            self._shutdown_event.clear()
            self._worker_thread = threading.Thread(
                target=self._async_log_worker,
                daemon=True,
                name="mlflow-async-logger",
            )
            self._worker_thread.start()
            logger.debug("Started async MLflow logging worker")

            # Register cleanup on exit
            atexit.register(self._flush_and_shutdown)

        except Exception as e:
            logger.warning("MLflow callback setup failed (non-fatal): %s", e)

    def _async_log_worker(self) -> None:
        """Background worker that batches and logs metrics asynchronously."""
        pending_metrics: dict[str, list[Metric]] = {}  # run_id -> metrics
        last_flush_time = time.time()

        while not self._shutdown_event.is_set():
            try:
                # Wait for items with timeout to allow periodic flushing
                try:
                    item = self._log_queue.get(timeout=0.5)
                except queue.Empty:
                    item = None

                if item is None and self._shutdown_event.is_set():
                    break

                if item is not None:
                    run_id, metrics = item
                    if run_id not in pending_metrics:
                        pending_metrics[run_id] = []
                    pending_metrics[run_id].extend(metrics)

                # Flush if batch is full or timeout reached
                now = time.time()
                should_flush = any(len(m) >= self.BATCH_SIZE for m in pending_metrics.values()) or (
                    now - last_flush_time >= self.BATCH_TIMEOUT_S and pending_metrics
                )

                if should_flush:
                    self._flush_metrics(pending_metrics)
                    pending_metrics.clear()
                    last_flush_time = now

            except Exception as e:
                logger.debug("Async MLflow worker error (continuing): %s", e)

        # Final flush on shutdown
        if pending_metrics:
            self._flush_metrics(pending_metrics)

    def _flush_metrics(self, pending_metrics: dict[str, list[Metric]]) -> None:
        """Flush pending metrics to MLflow using batch API."""
        if not self._mlflow_client or not pending_metrics:
            return

        for run_id, metrics in pending_metrics.items():
            if not metrics:
                continue
            try:
                # Use batch logging API (single HTTP request for all metrics)
                self._mlflow_client.log_batch(run_id=run_id, metrics=metrics)
            except Exception as e:
                error_str = str(e)
                if any(
                    msg in error_str
                    for msg in [
                        "must be in the 'active' state",
                        "Current state is deleted",
                        "Current state is finished",
                        "INVALID_PARAMETER_VALUE",
                        "RESOURCE_DOES_NOT_EXIST",
                    ]
                ):
                    logger.debug("MLflow batch log skipped (run inactive): %s", run_id[:8])
                else:
                    logger.debug("MLflow batch log failed for %s: %s", run_id[:8], e)

    def _flush_and_shutdown(self) -> None:
        """Flush remaining metrics and shutdown worker thread."""
        self._shutdown_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def on_trial_start(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Create MLflow run for new trial."""
        if trial in self._trial_runs:
            return

        try:
            tags = self._tags.copy()
            tags["trial_name"] = str(trial)
            tags["trial_id"] = trial.trial_id

            run = mlflow.start_run(run_name=str(trial), tags=tags, nested=False)
            self._trial_runs[trial] = run.info.run_id

            # Log config parameters (do this synchronously since it's once per trial)
            if trial.config:
                params = _flatten_dict(trial.config)
                # Filter to MLflow-compatible params
                params_to_log = {
                    k[:250]: str(v)[:500] for k, v in params.items() if not k.startswith("_") and v is not None
                }
                if params_to_log and self._mlflow_client:
                    try:
                        # Log params one by one (MlflowClient doesn't have log_params)
                        for key, value in params_to_log.items():
                            self._mlflow_client.log_param(run_id=run.info.run_id, key=key, value=value)
                    except Exception as e:
                        logger.debug("Failed to log params for %s: %s", trial.trial_id, e)

            # End the run context but keep it active for logging
            mlflow.end_run(status="RUNNING")

        except Exception as e:
            logger.debug("MLflow on_trial_start failed for %s: %s", trial.trial_id, e)

    def on_trial_result(
        self,
        iteration: int,
        trials: list[Trial],
        trial: Trial,
        result: dict[str, Any],
        **info: Any,
    ) -> None:
        """Queue metrics for async batch logging (non-blocking)."""
        if trial.trial_id in self._failed_trials:
            return

        if trial not in self._trial_runs:
            # Trial wasn't started properly, skip
            return

        run_id = self._trial_runs[trial]
        step = result.get(TRAINING_ITERATION, 0)
        timestamp_ms = int(time.time() * 1000)

        # Convert result dict to MLflow Metric objects
        metrics: list[Metric] = []
        for key, value in result.items():
            if key.startswith("_") or key in ("config", "done", "trial_id"):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    metrics.append(Metric(key=key[:250], value=float(value), timestamp=timestamp_ms, step=step))
                except (ValueError, TypeError):
                    pass

        if metrics:
            # Non-blocking queue put - worker will batch and log
            try:
                self._log_queue.put_nowait((run_id, metrics))
            except queue.Full:
                logger.debug("MLflow log queue full, dropping metrics for %s", trial.trial_id)

    def on_trial_complete(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Mark trial run as finished."""
        if trial not in self._trial_runs:
            return

        run_id = self._trial_runs[trial]
        try:
            if self._mlflow_client:
                self._mlflow_client.set_terminated(run_id=run_id, status="FINISHED")
        except Exception as e:
            logger.debug("MLflow on_trial_complete failed for %s: %s", trial.trial_id, e)

    def on_trial_error(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Mark trial run as failed."""
        if trial not in self._trial_runs:
            return

        run_id = self._trial_runs[trial]
        try:
            if self._mlflow_client:
                self._mlflow_client.set_terminated(run_id=run_id, status="FAILED")
        except Exception as e:
            logger.debug("MLflow on_trial_error failed for %s: %s", trial.trial_id, e)

    def on_experiment_end(self, trials: list[Trial], **info: Any) -> None:
        """Flush remaining metrics and cleanup."""
        # Flush any remaining metrics
        self._flush_and_shutdown()

        if self._failed_trials:
            logger.info(
                "MLflow logging was skipped for %d trial(s) due to run state issues",
                len(self._failed_trials),
            )


# Backward compatibility alias
RobustMLflowLoggerCallback = AsyncBatchedMLflowCallback


def backfill_mlflow_from_ray_results(
    results: ResultGrid,
    experiment_name: str,
    parent_run_id: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Backfill MLflow with metrics from Ray Tune trials.

    This function extracts all metrics from Ray Tune's result storage and logs
    them to MLflow as separate child runs. This is useful when the MLflow callback
    failed to log metrics during HPO due to run state issues with ASHA early stopping.

    Logged data includes:
        - All hyperparameters from trial config (search space + fixed params)
        - All metrics reported by the trial:
            * Validation metrics: val_loss, val_mae, val_rmse, val_R2
            * Correlation metrics: val_pearson_r, val_spearman_rho, val_kendall_tau
            * Training metrics: train_loss, train_mae
            * Learning rate and epoch information
        - Checkpoint paths and trial metadata

    Args:
        results: Ray Tune ResultGrid containing trial results
        experiment_name: MLflow experiment name
        parent_run_id: Optional parent run ID to link trials as children
        tracking_uri: Optional MLflow tracking URI

    Example:
        >>> from admet.model.chemprop.hpo import ChempropHPO
        >>> hpo = ChempropHPO(config)
        >>> results = hpo.run()
        >>> backfill_mlflow_from_ray_results(
        ...     results,
        ...     experiment_name="chemprop_hpo",
        ...     parent_run_id=hpo._mlflow_run_id
        ... )
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    backfilled_count = 0
    failed_count = 0

    for trial_result in results:
        trial_id = "unknown"
        try:
            # Extract trial metadata
            trial_id = (trial_result.metrics or {}).get("trial_id", "unknown")
            config = trial_result.config or {}

            # Create tags
            tags = {
                "trial_id": str(trial_id),
                "source": "ray_tune_backfill",
            }
            if parent_run_id:
                tags["mlflow.parentRunId"] = parent_run_id

            # Start a new run for this trial
            with mlflow.start_run(
                run_name=f"trial_{trial_id}_backfill",
                tags=tags,
                nested=bool(parent_run_id),
            ):
                # Log ALL config parameters (search space + fixed params)
                # Ray's trial.config already contains the complete parameter set,
                # including data paths, target columns, and all hyperparameters
                flat_config = _flatten_dict(config)

                # Filter to MLflow-compatible params:
                # - Remove internal Ray params (starting with "_")
                # - Truncate key names to 250 chars (MLflow limit)
                # - Skip values longer than 500 chars (very long lists/paths)
                params_to_log = {
                    k[:250]: v for k, v in flat_config.items() if not k.startswith("_") and len(str(v)) < 500
                }
                if params_to_log:
                    mlflow.log_params(params_to_log)

                # Log ALL metrics from the trial
                # This includes validation, training, and correlation metrics:
                # - val_loss, val_mae, val_rmse, val_R2
                # - val_pearson_r, val_spearman_rho, val_kendall_tau
                # - train_loss, train_mae
                # - lr (learning rate), epoch
                if trial_result.metrics:
                    metrics_to_log = {
                        k: float(v)
                        for k, v in trial_result.metrics.items()
                        if isinstance(v, (int, float)) and not k.startswith("_")
                    }
                    if metrics_to_log:
                        mlflow.log_metrics(metrics_to_log)

                # If we have a checkpoint path, log it as a tag
                if trial_result.checkpoint:
                    try:
                        checkpoint_path = trial_result.checkpoint.path
                        if checkpoint_path:
                            mlflow.set_tag("checkpoint_path", str(checkpoint_path))
                    except Exception as e:
                        logger.debug("Could not log checkpoint path: %s", e)

                backfilled_count += 1

        except Exception as e:
            logger.warning("Failed to backfill trial %s: %s", trial_id, e)
            failed_count += 1

    logger.info(
        "Backfilled %d trial(s) to MLflow, %d failed",
        backfilled_count,
        failed_count,
    )


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dictionary for MLflow param logging.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix for recursive calls
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            # Convert complex types to strings
            if not isinstance(v, (str, int, float, bool, type(None))):
                v = str(v)
            items.append((new_key, v))
    return dict(items)
