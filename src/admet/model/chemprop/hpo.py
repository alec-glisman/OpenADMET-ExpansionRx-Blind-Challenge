"""Chemprop Hyperparameter Optimization with Ray Tune.

This module provides the main orchestrator class for running hyperparameter
optimization of Chemprop models using Ray Tune with ASHA scheduler.

Example usage
-------------
CLI:
    python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml

Python:
    from admet.model.chemprop.hpo import ChempropHPO
    from admet.model.chemprop.hpo_config import HPOConfig

    config = HPOConfig(
        experiment_name="my_hpo",
        data_path="data/train.csv",
        target_columns=["logD", "solubility"],
    )
    hpo = ChempropHPO(config)
    results = hpo.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import mlflow
from omegaconf import OmegaConf
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from admet.model.chemprop.hpo_config import HPOConfig
from admet.model.chemprop.hpo_search_space import build_search_space
from admet.model.chemprop.hpo_trainable import train_chemprop_trial
from admet.model.hpo_mlflow_callback import AsyncBatchedMLflowCallback
from admet.util.logging import configure_logging
from admet.util.profiling import TrainingPhase, TrainingProfiler
from admet.util.ray_logging import QuietProgressReporter, RayLogManager

# Set Ray Tune environment variables at module load time (BEFORE ray.init)
# These must be set before Ray workers spawn to take effect
os.environ.setdefault("TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S", "300")
os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", "600")
os.environ.setdefault("TUNE_WARN_THRESHOLD_S", "30")  # Warn only if >30s (async callback is fast)
# OPTIMIZATION: Reduce buffering for more responsive metric reporting (5-10% overhead reduction)
os.environ.setdefault("TUNE_RESULT_BUFFER_LENGTH", "1")  # Reduced from 10
os.environ.setdefault("TUNE_RESULT_BUFFER_MIN_TIME_S", "1")  # Reduced from 10
# Disable tqdm progress bars globally for HPO
os.environ["TQDM_DISABLE"] = "1"


def _trial_dirname_creator(trial) -> str:
    """Create a short directory name for the trial to avoid filesystem limits."""
    return f"trial_{trial.trial_id}"


logger = logging.getLogger("admet.model.chemprop.hpo")


class ChempropHPO:
    """Orchestrator for Chemprop hyperparameter optimization.

    This class manages the full HPO workflow:
    1. Builds Ray Tune search space from configuration
    2. Runs HPO trials with ASHA early stopping
    3. Logs results to MLflow
    4. Saves top-k configurations for downstream ensemble training

    Attributes:
        config: HPO configuration
        results: Ray Tune results after running HPO
    """

    def __init__(self, config: HPOConfig) -> None:
        """Initialize the HPO orchestrator.

        Args:
            config: HPO configuration dataclass
        """
        self.config = config
        self.results: tune.ResultGrid | None = None
        self._mlflow_run_id: str | None = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._profiler = TrainingProfiler(name="hpo")

    def run(self) -> tune.ResultGrid:
        """Run hyperparameter optimization.

        Executes Ray Tune HPO with ASHA scheduler, logs results to MLflow,
        and returns the result grid.

        Returns:
            Ray Tune ResultGrid containing all trial results
        """
        # Start profiler for overall HPO timing
        self._profiler.start()

        # Setup MLflow tracking
        with self._profiler.phase(TrainingPhase.MLFLOW_INIT):
            self._setup_mlflow()

        # Setup Ray logging (if enabled in config)
        ray_log_manager = None
        if self.config.logging.enabled and self._mlflow_run_id:
            ray_log_manager = RayLogManager(
                mlflow_run_id=self._mlflow_run_id,
                output_dir=Path(self.config.output_dir),
                verbose=self.config.logging.verbose,
                max_total_logs_gb=self.config.logging.max_total_logs_gb,
                fail_on_upload_error=self.config.logging.fail_on_upload_error,
            )

        # Use context manager for logging if enabled
        ctx = ray_log_manager if ray_log_manager else self._null_context()

        with ctx:
            # Build search space
            with self._profiler.phase(TrainingPhase.HPO_SEARCH_SPACE_BUILD):
                search_space = self._build_search_space()

                # Build ASHA scheduler
                scheduler = self._build_scheduler()

                # Build search algorithm (Optuna, BayesOpt, etc.)
                search_alg = self._build_search_algorithm()

            # Configure Ray Tune
            # Note: metric/mode are specified in scheduler, not TuneConfig, to avoid conflict
            tune_config = tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,  # Add search algorithm
                num_samples=self.config.resources.num_samples,
                max_concurrent_trials=self.config.resources.max_concurrent_trials,
                trial_dirname_creator=_trial_dirname_creator,
            )

            # Configure resources per trial
            trainable = tune.with_resources(
                train_chemprop_trial,
                resources={
                    "cpu": self.config.resources.cpus_per_trial,
                    "gpu": self.config.resources.gpus_per_trial,
                },
            )

            # Setup storage path (must be absolute for Ray Tune)
            storage_path = self.config.ray_storage_path
            if storage_path is None:
                storage_path = str(Path(self.config.output_dir) / "ray_results")
            # Convert to absolute path if relative
            storage_path = str(Path(storage_path).resolve())

            # Create a dedicated temp directory inside the storage path to avoid issues
            # when system /tmp is cleaned during long-running HPO jobs
            ray_temp_dir = str(Path(storage_path) / "_ray_tmp")
            Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)

            # Initialize Ray with custom temp dir if storage path is provided
            # This helps avoid FileNotFoundError during sync when /tmp is cleaned
            # Disable dashboard to avoid MetricsHead startup failures on some systems

            # Suppress Ray future warning about GPU environment variables
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            import ray

            with self._profiler.phase(TrainingPhase.HPO_RAY_INIT):
                if not ray.is_initialized():
                    ray.init(
                        _temp_dir=ray_temp_dir,
                        include_dashboard=False,  # Disable dashboard to avoid startup errors
                    )
                    logger.info("Ray initialized with temp dir: %s", ray_temp_dir)

            # Run HPO
            logger.info(
                "Starting HPO: %d trials, metric=%s, mode=%s",
                self.config.resources.num_samples,
                self.config.asha.metric,
                self.config.asha.mode,
            )

            # Setup async batched MLflow callback for per-trial logging (non-blocking)
            tags: dict[str, str] = {"parent_run_id": self._mlflow_run_id or ""}
            if self._mlflow_run_id:
                # Attach Ray Tune trial runs as children of the parent HPO run
                tags["mlflow.parentRunId"] = self._mlflow_run_id

            mlflow_callback = AsyncBatchedMLflowCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=self.config.experiment_name,
                save_artifact=False,  # Disable artifact saving during HPO to avoid performance bottleneck
                tags=tags,
            )

            # Use quiet progress reporter if logging is enabled
            progress_reporter = QuietProgressReporter() if self.config.logging.enabled else None

            tuner = tune.Tuner(
                trainable,
                param_space=search_space,
                tune_config=tune_config,
                run_config=tune.RunConfig(
                    name=self.config.experiment_name,
                    storage_path=storage_path,
                    verbose=0 if self.config.logging.verbose == 0 else 1,
                    progress_reporter=progress_reporter,
                    callbacks=[mlflow_callback],
                    sync_config=tune.SyncConfig(
                        sync_period=300,  # Sync every 5 minutes instead of every result
                    ),
                    checkpoint_config=tune.CheckpointConfig(
                        num_to_keep=2,  # Keep only 2 checkpoints per trial (reduces state size)
                        checkpoint_score_attribute="val_loss",
                        checkpoint_score_order="min",
                    ),
                ),
            )

            try:
                with self._profiler.phase(TrainingPhase.HPO_TOTAL):
                    # Suppress Ray Tune's stdout messages during HPO
                    import contextlib
                    from io import StringIO

                    if self.config.logging.verbose == 0:
                        # Redirect stdout to suppress Ray messages
                        with contextlib.redirect_stdout(StringIO()):
                            self.results = tuner.fit()
                    else:
                        self.results = tuner.fit()
            except Exception as e:
                logger.error("HPO failed or interrupted: %s", e)
                # Try to restore results if possible, or just log what we have
                # Note: tuner.fit() might raise, but we might still have partial results on disk
                # However, getting the ResultGrid object from a failed run is tricky without restoring.
                # For now, we'll just log the error and try to proceed if self.results was set (unlikely)
                # or if we can recover something.
                # Actually, Ray Tune usually returns the ResultGrid even on failure if configured,
                # but here it raises.
                # We can try to restore the tuner to get results.
                try:
                    logger.info("Attempting to restore Tuner to retrieve partial results...")
                    tuner = tune.Tuner.restore(
                        path=str(Path(storage_path) / self.config.experiment_name),
                        trainable=trainable,
                    )
                    self.results = tuner.get_results()
                except Exception as restore_error:
                    logger.warning("Could not restore Tuner results: %s", restore_error)
            finally:
                # Log results to MLflow (best so far)
                with self._profiler.phase(TrainingPhase.HPO_RESULTS_AGGREGATION):
                    if self.results:
                        self._log_results()
                        # Backfill disabled: RobustMLflowLoggerCallback already logs during trials
                        # Uncomment only if you need to recover metrics after HPO failure:
                        # logger.info("Backfilling MLflow with metrics from all trials...")
                        # backfill_mlflow_from_ray_results(
                        #     self.results,
                        #     experiment_name=self.config.experiment_name,
                        #     parent_run_id=self._mlflow_run_id,
                        #     tracking_uri=self.config.mlflow_tracking_uri,
                        # )
                    else:
                        logger.warning("No results to log to MLflow.")
                        if self._mlflow_run_id:
                            mlflow.end_run()

                # Stop profiler and print summary
                self._profiler.stop()
                # Disable profiling output for cleaner logs
                # self._profiler.print_summary()

                # Log profiling metrics to MLflow if tracking enabled
                # Disabled: profiling has minimal overhead but adds clutter
                # if self._mlflow_run_id:
                #     self._log_profiling_to_mlflow()

        if self.results is None:
            raise RuntimeError("HPO failed to produce any results.")

        return self.results

    def _build_search_space(self) -> dict[str, Any]:
        """Build the Ray Tune search space.

        Combines the configurable search space with fixed parameters
        needed by the trainable function.

        Returns:
            Complete parameter space dictionary for Ray Tune
        """
        # Get configurable search space (pass target_columns for per-target weights)
        space = build_search_space(
            self.config.search_space,
            target_columns=list(self.config.target_columns),
        )

        # Add fixed parameters needed by trainable
        # Convert paths to absolute to ensure Ray workers can find them
        space["data_path"] = str(Path(self.config.data_path).resolve())
        space["val_data_path"] = str(Path(self.config.val_data_path).resolve()) if self.config.val_data_path else None
        space["test_data_path"] = (
            str(Path(self.config.test_data_path).resolve()) if self.config.test_data_path else None
        )
        space["smiles_column"] = self.config.smiles_column
        space["target_columns"] = self.config.target_columns
        space["max_epochs"] = self.config.asha.max_t
        space["metric"] = self.config.asha.metric
        space["seed"] = self.config.seed

        # Training parameters (pass through from HPO config)
        space["patience"] = self.config.patience
        space["warmup_epochs"] = self.config.warmup_epochs
        space["report_every_n_epochs"] = self.config.report_every_n_epochs

        # Pass fixed target weights if provided
        if self.config.target_weights is not None:
            space["target_weights"] = self.config.target_weights

        return space

    def _build_scheduler(self) -> ASHAScheduler:
        """Build the ASHA scheduler.

        Returns:
            Configured ASHAScheduler instance
        """
        return ASHAScheduler(
            time_attr="epoch",
            metric=self.config.asha.metric,
            mode=self.config.asha.mode,
            max_t=self.config.asha.max_t,
            grace_period=self.config.asha.grace_period,
            reduction_factor=self.config.asha.reduction_factor,
            brackets=self.config.asha.brackets,
        )

    def _build_search_algorithm(self):
        """Build the search algorithm (Optuna, BayesOpt, HyperOpt, or random).

        Returns:
            Configured search algorithm or None for random search
        """
        if self.config.search_algorithm is None:
            logger.info("No search algorithm configured, using random search")
            return None

        algo_type = self.config.search_algorithm.type.lower()

        if algo_type == "random" or algo_type == "none":
            logger.info("Random search algorithm selected")
            return None

        elif algo_type == "optuna":
            import optuna
            from ray.tune.search.optuna import OptunaSearch

            # Determine storage location and study persistence
            storage: optuna.storages.BaseStorage | None = None
            if self.config.search_algorithm.persist_study:
                storage_dir = (
                    Path(self.config.search_algorithm.storage_dir or self.config.output_dir) / "optuna_studies"
                )
                storage_dir.mkdir(parents=True, exist_ok=True)
                storage_url = f"sqlite:///{storage_dir / 'studies.db'}"
                storage = optuna.storages.RDBStorage(url=storage_url)

                # Generate or use provided study name
                study_name = self.config.search_algorithm.study_name
                if study_name is None:
                    study_name = f"hpo_{datetime.now():%Y%m%d_%H%M%S}"

                logger.info(
                    "Creating persistent Optuna study: %s (storage: %s)",
                    study_name,
                    storage_url,
                )
            else:
                storage = None
                study_name = None
                logger.info("Using ephemeral Optuna study (no persistence)")

            # Create sampler
            sampler = optuna.samplers.TPESampler(
                seed=self.config.search_algorithm.seed,
                n_startup_trials=self.config.search_algorithm.n_initial_points,
            )

            # Determine direction
            direction = "minimize" if self.config.asha.mode == "min" else "maximize"

            # Create or load study and handle warmstart
            if storage and study_name:
                # Check for warmstart first
                warmstart_from = self.config.search_algorithm.warmstart_from
                if warmstart_from:
                    logger.info("Warmstarting from study: %s", warmstart_from)
                    try:
                        old_study = optuna.load_study(
                            study_name=warmstart_from,
                            storage=storage,
                        )

                        # Get top N trials from previous study
                        n_warmstart = self.config.search_algorithm.warmstart_n_trials
                        top_trials = old_study.best_trials[:n_warmstart]

                        logger.info(
                            "Enqueuing %d top trials from %s (best value: %.4f)",
                            len(top_trials),
                            warmstart_from,
                            old_study.best_value,
                        )

                        # Create new study and enqueue warmstart trials
                        study = optuna.create_study(
                            study_name=study_name,
                            storage=storage,
                            sampler=sampler,
                            direction=direction,
                            load_if_exists=False,
                        )

                        # Enqueue trials as seeds
                        for trial in top_trials:
                            study.enqueue_trial(trial.params)

                    except Exception as e:
                        logger.warning(
                            "Failed to load warmstart study %s: %s. Continuing without warmstart.",
                            warmstart_from,
                            e,
                        )
                        # Create study without warmstart
                        study = optuna.create_study(
                            study_name=study_name,
                            storage=storage,
                            sampler=sampler,
                            direction=direction,
                            load_if_exists=False,
                        )
                else:
                    # Create study without warmstart
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=storage,
                        sampler=sampler,
                        direction=direction,
                        load_if_exists=False,
                    )

                # Create OptunaSearch with storage object (must be BaseStorage instance)
                search_alg = OptunaSearch(
                    space=None,  # Let Ray Tune define the space
                    sampler=sampler,
                    metric=self.config.asha.metric,
                    mode=self.config.asha.mode,
                    storage=storage,
                    study_name=study_name,
                )
            else:
                # Ephemeral study (original behavior)
                search_alg = OptunaSearch(
                    sampler=sampler,
                    metric=self.config.asha.metric,
                    mode=self.config.asha.mode,
                )

            logger.info(
                "Using Optuna search algorithm (TPESampler) with %d initial random points",
                self.config.search_algorithm.n_initial_points,
            )
            return search_alg

        elif algo_type == "bayesopt":
            from ray.tune.search.bayesopt import BayesOptSearch

            logger.info("Using BayesOptSearch algorithm")
            return BayesOptSearch(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
                random_state=self.config.search_algorithm.seed,
            )

        elif algo_type == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            logger.info("Using HyperOpt search algorithm")
            return HyperOptSearch(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
                random_state_seed=self.config.search_algorithm.seed,
            )

        else:
            logger.warning("Unknown search algorithm type: %s, falling back to random search", algo_type)
            return None

    @staticmethod
    def _null_context():
        """Return a no-op context manager for when logging is disabled."""
        from contextlib import nullcontext

        return nullcontext()

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        mlflow.set_experiment(self.config.experiment_name)

        # Log run parameters and capture run_id
        with mlflow.start_run(run_name=f"hpo_master_{self.timestamp}") as run:
            self._mlflow_run_id = run.info.run_id

            # Log comprehensive HPO configuration
            params_to_log = {
                # Experiment metadata
                "experiment_name": self.config.experiment_name,
                "timestamp": self.timestamp,
                # Data configuration
                "data_path": str(self.config.data_path),
                "val_data_path": str(self.config.val_data_path) if self.config.val_data_path else None,
                "smiles_column": self.config.smiles_column,
                "target_columns": str(self.config.target_columns),
                "seed": self.config.seed,
                # Search algorithm configuration
                "search_algorithm.type": (
                    self.config.search_algorithm.type if self.config.search_algorithm else "random"
                ),
                "search_algorithm.seed": self.config.search_algorithm.seed if self.config.search_algorithm else None,
                "search_algorithm.n_initial_points": (
                    self.config.search_algorithm.n_initial_points if self.config.search_algorithm else None
                ),
                "search_algorithm.persist_study": (
                    self.config.search_algorithm.persist_study if self.config.search_algorithm else False
                ),
                "search_algorithm.study_name": (
                    self.config.search_algorithm.study_name if self.config.search_algorithm else None
                ),
                "search_algorithm.storage_dir": (
                    str(self.config.search_algorithm.storage_dir)
                    if self.config.search_algorithm and self.config.search_algorithm.storage_dir
                    else None
                ),
                "search_algorithm.warmstart_from": (
                    self.config.search_algorithm.warmstart_from if self.config.search_algorithm else None
                ),
                "search_algorithm.warmstart_n_trials": (
                    self.config.search_algorithm.warmstart_n_trials if self.config.search_algorithm else None
                ),
                # ASHA scheduler configuration
                "asha.metric": self.config.asha.metric,
                "asha.mode": self.config.asha.mode,
                "asha.max_t": self.config.asha.max_t,
                "asha.grace_period": self.config.asha.grace_period,
                "asha.reduction_factor": self.config.asha.reduction_factor,
                # Resource allocation
                "resources.num_samples": self.config.resources.num_samples,
                "resources.cpus_per_trial": self.config.resources.cpus_per_trial,
                "resources.gpus_per_trial": self.config.resources.gpus_per_trial,
                "resources.max_concurrent_trials": self.config.resources.max_concurrent_trials,
                # Transfer learning settings
                "transfer_learning.top_k": self.config.transfer_learning.top_k,
                "transfer_learning.full_epochs": self.config.transfer_learning.full_epochs,
                "transfer_learning.ensemble_size": self.config.transfer_learning.ensemble_size,
            }
            # Filter out None values and truncate long strings for MLflow param limits (500 chars)
            filtered_params = {}
            for k, v in params_to_log.items():
                if v is not None:
                    str_v = str(v)
                    filtered_params[k] = str_v[:500] if len(str_v) > 500 else str_v
            mlflow.log_params(filtered_params)

    def _log_profiling_to_mlflow(self) -> None:
        """Log profiling metrics to MLflow using fluent API."""
        try:
            prefix = "profiling"

            # Log total duration
            mlflow.log_metric(f"{prefix}.total_seconds", self._profiler.total_duration)

            # Log per-phase metrics
            for phase, stats in self._profiler.get_all_stats().items():
                safe_phase = phase.replace(".", "_").replace("-", "_")
                mlflow.log_metric(f"{prefix}.{safe_phase}.total_seconds", stats.total_seconds)
                mlflow.log_metric(f"{prefix}.{safe_phase}.mean_seconds", stats.mean_seconds)
                mlflow.log_metric(f"{prefix}.{safe_phase}.count", stats.count)

                if self._profiler.total_duration > 0:
                    pct = stats.total_seconds / self._profiler.total_duration * 100
                    mlflow.log_metric(f"{prefix}.{safe_phase}.percentage", float(pct))

            logger.debug("Logged profiling metrics to MLflow")

        except Exception as e:
            logger.warning("Failed to log profiling metrics to MLflow: %s", e)

    def _log_results(self) -> None:
        """Log HPO results to MLflow including detailed trial metrics.

        Logs:
        - Best trial configuration and metrics
        - Best model checkpoint
        - All trial metrics to MLflow as nested artifacts/parameters
        - HPO results dataframe and top-k configurations

        Important: This method reactivates the master HPO run to ensure all
        artifacts are logged to the parent run, not child trial runs.
        """
        if self.results is None:
            return

        # Reactivate the master HPO run to log results to parent run
        if not self._mlflow_run_id:
            logger.warning("No MLflow run ID available, skipping MLflow logging")
            return

        with mlflow.start_run(run_id=self._mlflow_run_id):
            # Get best result
            best_result = self.results.get_best_result(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )

            if best_result is not None:
                # Log best config
                best_config = best_result.config
                if best_config is not None:
                    mlflow.log_params({f"best.{k}": v for k, v in best_config.items() if not k.startswith("_")})

                # Log best metrics
                if best_result.metrics:
                    best_metrics: dict[str, float] = {
                        f"best.{k}": float(v) for k, v in best_result.metrics.items() if isinstance(v, (int, float))
                    }
                    mlflow.log_metrics(best_metrics)

                # Log best model artifact
                if best_result.checkpoint:
                    try:
                        # Ray Tune Checkpoint is a directory or file
                        # We want to log the best-*.ckpt file inside it
                        with best_result.checkpoint.as_directory() as checkpoint_dir:
                            checkpoint_path = Path(checkpoint_dir)
                            best_checkpoints = list(checkpoint_path.glob("best-*.ckpt"))
                            # Sort by modification time
                            best_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                            if best_checkpoints:
                                ckpt_file = best_checkpoints[0]
                                logger.info("Logging best HPO model artifact: %s", ckpt_file.name)
                                mlflow.log_artifact(str(ckpt_file), artifact_path="best_model")
                            else:
                                logger.warning("No best-*.ckpt found in best result checkpoint: %s", checkpoint_path)
                    except Exception as e:
                        logger.warning("Failed to log best model artifact: %s", e)

            # Log all trial metrics as detailed information
            logger.info("Logging detailed metrics from all %d trials to MLflow", len(self.results))
            results_df = self.results.get_dataframe()

            # Extract and log key metrics for each trial
            metric_cols = [col for col in results_df.columns if col.startswith(("val_", "train_"))]

            # Log summary statistics for each metric across all trials
            for metric_col in metric_cols:
                if metric_col in results_df.columns and results_df[metric_col].notna().any():
                    try:
                        values = results_df[metric_col].dropna()
                        if len(values) > 0:
                            mlflow.log_metrics(
                                {
                                    f"trials.{metric_col}.mean": float(values.mean()),
                                    f"trials.{metric_col}.std": float(values.std()),
                                    f"trials.{metric_col}.min": float(values.min()),
                                    f"trials.{metric_col}.max": float(values.max()),
                                }
                            )
                    except Exception as e:
                        logger.debug("Could not compute stats for metric %s: %s", metric_col, e)

            # Save all results as artifact
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results_path = output_dir / "hpo_results.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            logger.info("Logged HPO results dataframe with %d trials to MLflow", len(results_df))

            # Save top-k configs as JSON
            top_k = self.get_top_k_configs()
            top_k_path = output_dir / "top_k_configs.json"
            with open(top_k_path, "w", encoding="utf-8") as f:
                json.dump(top_k, f, indent=2)
            mlflow.log_artifact(str(top_k_path))
            logger.info("Logged top-%d configurations to MLflow", len(top_k))

            # Save study metadata if using persistent studies
            if self.config.search_algorithm.persist_study:
                study_metadata = {
                    "study_name": self.config.search_algorithm.study_name,
                    "storage_dir": (
                        str(self.config.search_algorithm.storage_dir)
                        if self.config.search_algorithm.storage_dir
                        else None
                    ),
                    "warmstart_from": self.config.search_algorithm.warmstart_from,
                    "warmstart_n_trials": self.config.search_algorithm.warmstart_n_trials,
                    "timestamp": self.timestamp,
                    "experiment_name": self.config.experiment_name,
                    "n_trials": len(results_df),
                    "best_metric": (
                        float(best_result.metrics.get(self.config.asha.metric, 0))
                        if best_result and best_result.metrics
                        else None
                    ),
                }
                metadata_path = output_dir / "study_metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(study_metadata, f, indent=2)
                mlflow.log_artifact(str(metadata_path))
                logger.info("Logged study metadata for study: %s", study_metadata["study_name"])

            # Export Optuna study information if using persistent study
            if self.config.search_algorithm.persist_study:
                self._export_optuna_study_artifacts(output_dir)

            # Log all files from storage_dir (excluding model checkpoints)
            if self.config.search_algorithm.storage_dir:
                self._log_storage_dir_artifacts()

            logger.info("HPO results logged to MLflow")

    def _export_optuna_study_artifacts(self, output_dir: Path) -> None:
        """Export Optuna study information to artifacts for MLflow logging.

        Exports:
        - Study trials to CSV and JSON
        - Study statistics and best trials
        - SQLite database copy (if available)
        - Study parameters and importance
        """
        try:
            import optuna

            if not self.config.search_algorithm.study_name:
                logger.warning("No study name available for Optuna export")
                return

            # Determine storage location
            storage_dir = Path(self.config.search_algorithm.storage_dir or self.config.output_dir) / "optuna_studies"
            storage_url = f"sqlite:///{storage_dir / 'studies.db'}"

            # Load the study
            study = optuna.load_study(
                study_name=self.config.search_algorithm.study_name,
                storage=storage_url,
            )

            # Export trials to DataFrame
            trials_df = study.trials_dataframe()
            trials_csv_path = output_dir / "optuna_trials.csv"
            trials_df.to_csv(trials_csv_path, index=False)
            mlflow.log_artifact(str(trials_csv_path), artifact_path="optuna")
            logger.info("Logged Optuna trials CSV: %d trials", len(trials_df))

            # Export study summary
            study_summary = {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "best_trial": {
                    "number": study.best_trial.number,
                    "value": study.best_value,
                    "params": study.best_params,
                    "datetime_start": str(study.best_trial.datetime_start),
                    "datetime_complete": str(study.best_trial.datetime_complete),
                    "duration": str(study.best_trial.duration) if study.best_trial.duration else None,
                },
                "best_trials": [
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                    }
                    for t in study.best_trials[:10]
                ],
                "direction": study.direction.name,
                "user_attrs": study.user_attrs,
                "system_attrs": study.system_attrs,
            }

            study_summary_path = output_dir / "optuna_study_summary.json"
            with open(study_summary_path, "w", encoding="utf-8") as f:
                json.dump(study_summary, f, indent=2, default=str)
            mlflow.log_artifact(str(study_summary_path), artifact_path="optuna")
            logger.info("Logged Optuna study summary")

            # Try to compute parameter importance if enough trials
            if len(study.trials) >= 10:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    importance_path = output_dir / "optuna_param_importance.json"
                    with open(importance_path, "w", encoding="utf-8") as f:
                        json.dump(importance, f, indent=2)
                    mlflow.log_artifact(str(importance_path), artifact_path="optuna")
                    logger.info("Logged parameter importance")
                except Exception as e:
                    logger.debug("Could not compute parameter importance: %s", e)

            # Copy SQLite database to MLflow (for complete reproducibility)
            db_source = storage_dir / "studies.db"
            if db_source.exists():
                import shutil

                db_dest = output_dir / "optuna_studies.db"
                shutil.copy2(db_source, db_dest)
                mlflow.log_artifact(str(db_dest), artifact_path="optuna")
                logger.info("Logged Optuna SQLite database: %s", db_dest.name)

        except Exception as e:
            logger.warning("Failed to export Optuna study artifacts: %s", e)

    def _log_storage_dir_artifacts(self) -> None:
        """Log all relevant files from storage_dir to MLflow, excluding model checkpoints."""
        try:
            import shutil

            if not self.config.search_algorithm.storage_dir:
                logger.debug("No storage_dir configured")
                return

            storage_dir = Path(self.config.search_algorithm.storage_dir)
            if not storage_dir.exists():
                logger.debug("Storage directory does not exist: %s", storage_dir)
                return

            # Patterns to include (JSON, CSV, YAML, text files, SQLite DBs)
            include_patterns = ["*.json", "*.csv", "*.yaml", "*.yml", "*.txt", "*.md", "*.db"]
            # Patterns to exclude (model checkpoints and large binary files)
            exclude_patterns = ["*.ckpt", "*.pth", "*.pt", "*.h5", "*.pkl", "*.pickle"]

            output_dir = Path(self.config.output_dir)
            storage_artifacts_dir = output_dir / "storage_dir_artifacts"
            storage_artifacts_dir.mkdir(parents=True, exist_ok=True)

            files_logged = 0
            for pattern in include_patterns:
                for file_path in storage_dir.rglob(pattern):
                    # Check if file matches any exclude pattern
                    if any(file_path.match(exclude_pat) for exclude_pat in exclude_patterns):
                        continue

                    # Preserve directory structure relative to storage_dir
                    rel_path = file_path.relative_to(storage_dir)
                    dest_path = storage_artifacts_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(file_path, dest_path)
                    files_logged += 1

            # Log the entire directory structure to MLflow
            if files_logged > 0:
                mlflow.log_artifacts(str(storage_artifacts_dir), artifact_path="storage_dir")
                logger.info("Logged %d files from storage_dir to MLflow", files_logged)
            else:
                logger.debug("No matching files found in storage_dir")

        except Exception as e:
            logger.warning("Failed to log storage_dir artifacts: %s", e)

    def get_top_k_configs(self) -> list[dict[str, Any]]:
        """Get the top-k configurations from HPO results.

        Returns:
            List of top-k configuration dictionaries with hyperparameters.
            Each config includes '_rank' and '_metric_value' metadata.
        """
        if self.results is None:
            return []

        k = self.config.transfer_learning.top_k
        results_df = self.results.get_dataframe()

        # Sort by metric
        metric = self.config.asha.metric
        ascending = self.config.asha.mode == "min"
        results_df = results_df.sort_values(metric, ascending=ascending)

        # Extract top-k configs
        top_k: list[dict[str, Any]] = []
        config_cols = [c for c in results_df.columns if c.startswith("config/")]

        # Fixed parameters to exclude from config
        fixed_params = {
            "data_path",
            "val_data_path",
            "smiles_column",
            "target_columns",
            "max_epochs",
            "metric",
            "seed",
        }

        for _, row in results_df.head(k).iterrows():
            config: dict[str, Any] = {}
            for col in config_cols:
                param_name = col.replace("config/", "")
                value = row[col]
                # Skip internal and fixed parameters
                if not param_name.startswith("_") and param_name not in fixed_params:
                    config[param_name] = value
            config["_rank"] = len(top_k) + 1
            config["_metric_value"] = row.get(metric)
            top_k.append(config)

        return top_k


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            # Convert to string for non-primitive types
            if not isinstance(v, (str, int, float, bool, type(None))):
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def main() -> None:
    """CLI entry point for Chemprop HPO."""
    parser = argparse.ArgumentParser(
        description="Run Chemprop hyperparameter optimization with Ray Tune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to HPO configuration YAML file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of HPO trials (overrides config value)",
    )
    parser.add_argument(
        "--logging-verbose",
        type=int,
        default=None,
        help="Logging verbosity level (0=quiet, 1=standard, 2=debug)",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable Ray logging to artifacts",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = OmegaConf.load(config_path)
    merged_config = OmegaConf.merge(OmegaConf.structured(HPOConfig), raw_config)
    config = cast(HPOConfig, OmegaConf.to_object(merged_config))

    # Override with CLI arguments if provided
    if args.num_samples is not None:
        config.resources.num_samples = args.num_samples

    # Run HPO
    hpo = ChempropHPO(config)
    results = hpo.run()

    # Print summary
    best = results.get_best_result(metric=config.asha.metric, mode=config.asha.mode)
    if best is not None:
        print(f"\nBest trial config: {best.config}")
        print(f"Best trial metrics: {best.metrics}")

    # Print top-k configs for downstream use
    top_k = hpo.get_top_k_configs()
    print(f"\nTop {len(top_k)} configurations saved to {config.output_dir}/top_k_configs.json")


if __name__ == "__main__":
    configure_logging(level="INFO")
    main()
