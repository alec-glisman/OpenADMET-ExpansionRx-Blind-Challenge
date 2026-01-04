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
from admet.util.ray_logging import QuietProgressReporter, RayLogManager

# Set Ray Tune environment variables at module load time (BEFORE ray.init)
# These must be set before Ray workers spawn to take effect
os.environ.setdefault("TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S", "300")
os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", "600")
os.environ.setdefault("TUNE_WARN_THRESHOLD_S", "30")  # Warn only if >30s (async callback is fast)
os.environ.setdefault("TUNE_RESULT_BUFFER_LENGTH", "10")
os.environ.setdefault("TUNE_RESULT_BUFFER_MIN_TIME_S", "10")


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

    def run(self) -> tune.ResultGrid:
        """Run hyperparameter optimization.

        Executes Ray Tune HPO with ASHA scheduler, logs results to MLflow,
        and returns the result grid.

        Returns:
            Ray Tune ResultGrid containing all trial results
        """
        # Setup MLflow tracking
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
            import ray

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

            logger.info(
                "Using Optuna search algorithm (TPESampler) with %d initial random points",
                self.config.search_algorithm.n_initial_points,
            )
            return OptunaSearch(
                sampler=optuna.samplers.TPESampler(
                    seed=self.config.search_algorithm.seed,
                    n_startup_trials=self.config.search_algorithm.n_initial_points,
                ),
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )

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

    def _log_results(self) -> None:
        """Log HPO results to MLflow."""
        if self.results is None:
            return

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

        # Save all results as artifact

        # Save all results as artifact
        results_df = self.results.get_dataframe()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "hpo_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        # Save top-k configs as JSON
        top_k = self.get_top_k_configs()
        top_k_path = output_dir / "top_k_configs.json"
        with open(top_k_path, "w") as f:
            json.dump(top_k, f, indent=2)
        mlflow.log_artifact(str(top_k_path))

        # End MLflow run
        mlflow.end_run()
        logger.info("HPO results logged to MLflow")

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

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = OmegaConf.load(config_path)
    merged_config = OmegaConf.merge(OmegaConf.structured(HPOConfig), raw_config)
    config = cast(HPOConfig, OmegaConf.to_object(merged_config))

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
