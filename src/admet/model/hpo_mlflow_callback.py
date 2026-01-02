"""Robust MLflow callback for Ray Tune HPO.

This module provides a wrapper around Ray's MLflowLoggerCallback that handles
common errors gracefully, particularly when ASHA scheduler terminates trials
and MLflow runs become inactive or deleted.

It also provides utilities for backfilling MLflow with metrics from Ray Tune's
result storage after HPO completes.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.tune.result_grid import ResultGrid

logger = logging.getLogger(__name__)


class RobustMLflowLoggerCallback(Callback):
    """MLflow callback wrapper that gracefully handles logging errors.

    This wrapper catches MLflow exceptions (e.g., when trying to log to deleted
    or inactive runs) that commonly occur when ASHA scheduler terminates trials.
    Instead of failing the entire HPO run, it logs a warning and continues.

    Attributes:
        _inner: The wrapped MLflowLoggerCallback instance
        _failed_trials: Set of trial IDs that have experienced logging failures
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        save_artifact: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize the robust MLflow callback.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the MLflow experiment
            save_artifact: Whether to save artifacts (disabled by default for HPO)
            tags: Tags to apply to MLflow runs
        """
        self._inner = MLflowLoggerCallback(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            save_artifact=save_artifact,
            tags=tags or {},
        )
        self._failed_trials: set[str] = set()

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Delegate setup to inner callback."""
        try:
            self._inner.setup(*args, **kwargs)
        except Exception as e:
            logger.warning("MLflow callback setup failed (non-fatal): %s", e)

    def on_trial_start(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Handle trial start event."""
        try:
            self._inner.on_trial_start(iteration, trials, trial, **info)
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
        """Handle trial result event, catching MLflow errors gracefully."""
        if trial.trial_id in self._failed_trials:
            return

        try:
            self._inner.on_trial_result(iteration, trials, trial, result, **info)
        except Exception as e:
            error_str = str(e)
            # Check for common MLflow errors related to run state
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
                logger.debug(
                    "MLflow logging skipped for trial %s (run inactive/deleted)",
                    trial.trial_id,
                )
                self._failed_trials.add(trial.trial_id)
            else:
                logger.warning(
                    "MLflow on_trial_result failed for %s: %s",
                    trial.trial_id,
                    e,
                )
                self._failed_trials.add(trial.trial_id)

    def on_trial_complete(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Handle trial completion event."""
        try:
            self._inner.on_trial_complete(iteration, trials, trial, **info)
        except Exception as e:
            logger.debug("MLflow on_trial_complete failed for %s: %s", trial.trial_id, e)

    def on_trial_error(self, iteration: int, trials: list[Trial], trial: Trial, **info: Any) -> None:
        """Handle trial error event."""
        try:
            self._inner.on_trial_error(iteration, trials, trial, **info)
        except Exception as e:
            logger.debug("MLflow on_trial_error failed for %s: %s", trial.trial_id, e)

    def on_experiment_end(self, trials: list[Trial], **info: Any) -> None:
        """Handle experiment end event."""
        try:
            self._inner.on_experiment_end(trials, **info)
        except Exception as e:
            logger.warning("MLflow on_experiment_end failed (non-fatal): %s", e)

        if self._failed_trials:
            logger.info(
                "MLflow logging was skipped for %d trial(s) due to run state issues",
                len(self._failed_trials),
            )


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
