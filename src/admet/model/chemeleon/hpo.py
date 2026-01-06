"""CheMeleon Hyperparameter Optimization with Ray Tune.

This module provides the main orchestrator class for running hyperparameter
optimization of CheMeleon models using Ray Tune with ASHA scheduler.

CheMeleon HPO differs from Chemprop HPO:
- The message passing encoder is frozen (pre-trained)
- Only FFN architecture and training dynamics are tuned
- Supports all FFN types: regression, mixture_of_experts, branched

Example usage
-------------
CLI:
    python -m admet.model.chemeleon.hpo --config configs/1-hpo-single/hpo_chemeleon.yaml

Python:
    from admet.model.chemeleon.hpo import ChemeleonHPO
    from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig

    config = ChemeleonHPOConfig(
        experiment_name="chemeleon_hpo",
        data_path="data/train.csv",
        target_columns=["logD", "solubility"],
    )
    hpo = ChemeleonHPO(config)
    results = hpo.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import mlflow
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig
from admet.model.chemeleon.hpo_search_space import build_chemeleon_search_space
from admet.model.hpo_mlflow_callback import AsyncBatchedMLflowCallback
from admet.util.logging import configure_logging
from admet.util.ray_logging import QuietProgressReporter, RayLogManager

# Set Ray Tune environment variables at module load time (BEFORE ray.init)
os.environ.setdefault("TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S", "300")
os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", "600")
os.environ.setdefault("TUNE_WARN_THRESHOLD_S", "30")
os.environ.setdefault("TUNE_RESULT_BUFFER_LENGTH", "10")
os.environ.setdefault("TUNE_RESULT_BUFFER_MIN_TIME_S", "10")
# Disable tqdm progress bars globally for HPO
os.environ["TQDM_DISABLE"] = "1"

logger = logging.getLogger("admet.model.chemeleon.hpo")


class ChemeleonRayTuneCallback(Callback):
    """PyTorch Lightning callback to report metrics to Ray Tune for CheMeleon HPO.

    This callback integrates with Ray Tune's reporting mechanism to enable
    early stopping via the ASHA scheduler. It reports validation metrics
    after each epoch and saves checkpoints for trial recovery.

    Attributes
    ----------
    metric : str
        Name of the primary metric for ASHA scheduling (default: "val_mae")
    checkpoint_dir : Path | None
        Directory to save checkpoints for recovery
    report_every_n_epochs : int
        Epoch cadence for Ray reporting (default: 5)
    """

    METRICS_TO_REPORT = (
        "val_mae",
        "val_loss",
        "val_rmse",
        "val_R2",
        "val_pearson_r",
        "val_spearman_rho",
        "val_kendall_tau",
        "train_loss",
        "train_mae",
        "lr",
    )

    def __init__(
        self,
        metric: str = "val_mae",
        checkpoint_dir: Path | None = None,
        report_every_n_epochs: int = 5,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        metric : str
            Name of the primary validation metric for ASHA scheduling.
        checkpoint_dir : Path | None
            Directory for saving checkpoints. If None, no checkpoints saved.
        report_every_n_epochs : int
            Number of epochs between Ray reports. Must be >= 1.
        """
        super().__init__()
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        self.report_every_n_epochs = max(1, report_every_n_epochs)
        self._params_total: int | None = None
        self._params_trainable: int | None = None
        self._params_frozen: int | None = None
        self._last_reported_epoch: int | None = None
        self._final_checkpoint_reported = False
        self._start_time = time.time()

    def _compute_param_counts(self, pl_module: pl.LightningModule) -> None:
        """Compute and cache parameter counts from the model.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The Lightning module (MPNN) to count parameters from.
        """
        if self._params_total is not None:
            return

        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for param in pl_module.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params

        self._params_total = total_params
        self._params_trainable = trainable_params
        self._params_frozen = frozen_params

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics to Ray Tune after validation."""
        if not trainer.callback_metrics:
            return

        epoch_index = int(trainer.current_epoch) + 1
        if self._last_reported_epoch == epoch_index:
            return

        max_epochs = getattr(trainer, "max_epochs", None)
        should_stop = bool(getattr(trainer, "should_stop", False))
        is_last_epoch = max_epochs is not None and epoch_index >= int(max_epochs)
        is_final_event = should_stop or is_last_epoch

        should_report = epoch_index == 1 or (epoch_index % self.report_every_n_epochs == 0) or is_final_event

        if not should_report:
            return

        metrics: dict[str, float] = {}
        metrics["epoch"] = float(trainer.current_epoch)
        metrics["training_time_seconds"] = time.time() - self._start_time

        # Report all tracked metrics that are available
        for metric_name in self.METRICS_TO_REPORT:
            if metric_name in trainer.callback_metrics:
                value = trainer.callback_metrics[metric_name]
                metrics[metric_name] = float(value.item() if hasattr(value, "item") else value)

        # Get current learning rate from optimizer
        if "lr" not in metrics and trainer.optimizers:
            try:
                opt = trainer.optimizers[0]
                if hasattr(opt, "param_groups") and opt.param_groups:
                    metrics["lr"] = float(opt.param_groups[0]["lr"])
            except (IndexError, KeyError):
                pass

        # Compute parameter counts dynamically from the model
        self._compute_param_counts(pl_module)
        metrics["params_total"] = float(self._params_total or 0)
        metrics["params_trainable"] = float(self._params_trainable or 0)
        metrics["params_frozen"] = float(self._params_frozen or 0)

        # Check if early stopping was triggered
        early_stopped = False
        for callback in getattr(trainer, "callbacks", []):
            if hasattr(callback, "stopped_epoch") and callback.stopped_epoch > 0:
                early_stopped = True
                break
        metrics["early_stopped"] = float(early_stopped)

        # Ensure primary metric is present
        if self.metric not in metrics:
            if "val_loss" in trainer.callback_metrics:
                value = trainer.callback_metrics["val_loss"]
                metrics[self.metric] = float(value.item() if hasattr(value, "item") else value)

        # Report to Ray Tune
        if self.metric in metrics:
            checkpoint: Checkpoint | None = None
            if is_final_event and not self._final_checkpoint_reported:
                checkpoint = self._build_final_checkpoint(trainer)
                if checkpoint is not None:
                    self._final_checkpoint_reported = True

            self._submit_report(metrics, checkpoint)
            self._last_reported_epoch = epoch_index

    def _build_final_checkpoint(self, trainer: pl.Trainer) -> Checkpoint | None:
        """Package the latest checkpoint directory for Ray Tune reporting."""
        _ = trainer  # Future: could use trainer.callback_metrics for additional data
        if self.checkpoint_dir is None:
            return None

        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.exists():
            return None

        # Find preferred checkpoint file
        best_checkpoints = sorted(ckpt_dir.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoint_file: Path | None = best_checkpoints[0] if best_checkpoints else None

        if checkpoint_file is None:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                checkpoint_file = last_ckpt

        if checkpoint_file is None:
            return None

        export_dir = ckpt_dir / "ray_checkpoint"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(checkpoint_file, export_dir / checkpoint_file.name)

        try:
            return Checkpoint.from_directory(str(export_dir))
        except Exception:
            return None

    def _submit_report(self, metrics: dict[str, float], checkpoint: Checkpoint | None) -> None:
        """Send metrics to Ray Tune using tune.report."""
        if checkpoint is not None:
            tune.report(metrics, checkpoint=checkpoint)
        else:
            tune.report(metrics)


def _trial_dirname_creator(trial) -> str:
    """Create a short directory name for the trial."""
    return f"trial_{trial.trial_id}"


def train_chemeleon_trial(config: dict[str, Any]) -> None:
    """Ray Tune trainable function for CheMeleon HPO trials.

    This function is called by Ray Tune for each trial. It creates a
    CheMeleon model with the sampled hyperparameters and trains it.

    Parameters
    ----------
    config : dict[str, Any]
        Hyperparameter configuration sampled by Ray Tune.
    """
    import pandas as pd

    from admet.model.chemeleon import ChemeleonModel

    # Suppress tqdm progress bars FIRST (before any data loading)
    os.environ["TQDM_DISABLE"] = "1"

    # Stagger trial starts to avoid race conditions
    time.sleep(1)

    # Suppress noisy loggers to reduce terminal spam
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
    logging.getLogger("chemprop").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    # Extract fixed parameters
    data_path = config.pop("data_path")
    val_data_path = config.pop("val_data_path", None)
    smiles_column = config.pop("smiles_column")
    target_columns = config.pop("target_columns")
    max_epochs = config.pop("max_epochs")
    metric = config.pop("metric")
    seed = config.pop("seed", 42)
    checkpoint_path = config.pop("checkpoint_path", "auto")
    freeze_encoder = config.pop("freeze_encoder", True)
    report_every_n_epochs = config.pop("report_every_n_epochs", 5)

    # Extract profiling configuration from HPO config
    profiling_config_dict = config.pop("profiling", {})
    from admet.model.chemprop.config import ProfilingConfig

    profiling_config = ProfilingConfig(
        enabled=profiling_config_dict.get("enabled", False),
        print_summary=profiling_config_dict.get("print_summary", False),
        log_to_mlflow=profiling_config_dict.get("log_to_mlflow", False),
    )

    # Extract target weights if provided
    target_weights = config.pop("target_weights", None)

    # Extract unfreezing schedule parameters (may be overridden by search space)
    unfreeze_encoder_epoch = config.pop("unfreeze_encoder_epoch", None)
    unfreeze_encoder_lr_multiplier = config.pop("unfreeze_encoder_lr_multiplier", 0.1)

    # Seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # Load data
    df_train = pd.read_csv(data_path)
    df_val = pd.read_csv(val_data_path) if val_data_path else None

    # Build unfreeze schedule config
    unfreeze_schedule = {
        "freeze_encoder": freeze_encoder,
        "unfreeze_encoder_epoch": unfreeze_encoder_epoch,
        "unfreeze_encoder_lr_multiplier": unfreeze_encoder_lr_multiplier,
    }

    # Get trial directory for checkpoint saving
    trial_dir = None
    try:
        trial_dir_str = tune.get_context().get_trial_dir()
        if trial_dir_str:
            trial_dir = Path(trial_dir_str)
    except Exception:
        pass

    if trial_dir is None:
        trial_dir = Path.cwd() / "ray_trial"

    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build model config
    # All hyperparameters should be sampled from the search space
    # Use config.get() without defaults to ensure Ray Tune provides all values
    model_config = OmegaConf.create(
        {
            "model": {
                "type": "chemeleon",
                "chemeleon": {
                    "checkpoint_path": checkpoint_path,
                    "freeze_encoder": freeze_encoder,
                    "unfreeze_schedule": unfreeze_schedule,
                    "ffn_type": config["ffn_type"],  # Required from search space
                    "ffn_hidden_dim": config["ffn_hidden_dim"],  # Required from search space
                    "ffn_num_layers": config["ffn_num_layers"],  # Required from search space
                    "dropout": config["dropout"],  # Required from search space
                    "batch_norm": config["batch_norm"],  # Required from search space
                    "n_experts": config.get("n_experts"),  # Conditional: None if not MoE
                    "trunk_n_layers": config.get("trunk_n_layers"),  # Conditional: None if not branched
                    "trunk_hidden_dim": config.get("trunk_hidden_dim"),  # Conditional: None if not branched
                },
            },
            "data": {
                "smiles_col": smiles_column,
                "target_cols": list(target_columns),
                "target_weights": target_weights,
            },
            "optimization": {
                "max_epochs": max_epochs,
                "batch_size": config["batch_size"],  # Required from search space
                "patience": config.get("patience", 15),  # Optional with reasonable default
                "learning_rate": config["learning_rate"],  # Required from search space
                "weight_decay": config["weight_decay"],  # Required from search space
                "lr_warmup_ratio": config["lr_warmup_ratio"],  # Required from search space
                "lr_final_ratio": config["lr_final_ratio"],  # Required from search space
                "warmup_epochs": config.get("warmup_epochs", 2),  # Optional with default
                "checkpoint_dir": str(checkpoint_dir),
            },
            "mlflow": {"enabled": False},
        }
    )

    # Create model with MLflow tracking disabled (Ray Tune manages MLflow)
    # Metrics are still computed by PyTorch Lightning and reported via ChemeleonRayTuneCallback
    model = ChemeleonModel(
        model_config,
        profiling_config=profiling_config,  # Use profiling config from HPO config
    )

    # Extract data
    train_smiles = df_train[smiles_column].tolist()
    train_y = df_train[target_columns].values
    val_smiles = df_val[smiles_column].tolist() if df_val is not None else None
    val_y = df_val[target_columns].values if df_val is not None else None

    # Add Ray Tune callback for intermediate reporting
    # Parameter counts are computed dynamically from pl_module during on_validation_end
    ray_callback = ChemeleonRayTuneCallback(
        metric=metric,
        checkpoint_dir=checkpoint_dir,
        report_every_n_epochs=report_every_n_epochs,
    )
    model.add_callback(ray_callback)

    # Train the model (this handles all setup internally)
    model.fit(train_smiles, train_y, val_smiles, val_y)


class ChemeleonHPO:
    """Orchestrator for CheMeleon hyperparameter optimization.

    This class manages the full HPO workflow:
    1. Builds Ray Tune search space from configuration
    2. Runs HPO trials with ASHA early stopping
    3. Logs results to MLflow
    4. Saves top-k configurations for downstream use

    Attributes
    ----------
    config : ChemeleonHPOConfig
        HPO configuration
    results : tune.ResultGrid | None
        Ray Tune results after running HPO
    """

    def __init__(self, config: ChemeleonHPOConfig) -> None:
        """Initialize the HPO orchestrator.

        Parameters
        ----------
        config : ChemeleonHPOConfig
            HPO configuration dataclass
        """
        self.config = config
        self.results: tune.ResultGrid | None = None
        self._mlflow_run_id: str | None = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self) -> tune.ResultGrid:
        """Run hyperparameter optimization.

        Returns
        -------
        tune.ResultGrid
            Ray Tune ResultGrid containing all trial results
        """
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
            search_space = self._build_search_space()
            scheduler = self._build_scheduler()
            search_alg = self._build_search_algorithm()  # Add search algorithm

            tune_config = tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,  # Add search algorithm
                num_samples=self.config.resources.num_samples,
                max_concurrent_trials=self.config.resources.max_concurrent_trials,
                trial_dirname_creator=_trial_dirname_creator,
            )

            trainable = tune.with_resources(
                train_chemeleon_trial,
                resources={
                    "cpu": self.config.resources.cpus_per_trial,
                    "gpu": self.config.resources.gpus_per_trial,
                },
            )

            storage_path = self.config.ray_storage_path
            if storage_path is None:
                storage_path = str(Path(self.config.output_dir) / "ray_results")
            storage_path = str(Path(storage_path).resolve())

            ray_temp_dir = str(Path(storage_path) / "_ray_tmp")
            Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)

            import os

            import ray

            # Suppress Ray future warning about GPU environment variables
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

            if not ray.is_initialized():
                ray.init(_temp_dir=ray_temp_dir, include_dashboard=False)
                logger.info("Ray initialized with temp dir: %s", ray_temp_dir)

            logger.info(
                "Starting CheMeleon HPO: %d trials, metric=%s, mode=%s",
                self.config.resources.num_samples,
                self.config.asha.metric,
                self.config.asha.mode,
            )

            tags: dict[str, str] = {"parent_run_id": self._mlflow_run_id or ""}
            if self._mlflow_run_id:
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
                        checkpoint_score_attribute="val_mae",
                        checkpoint_score_order="min",
                    ),
                ),
            )

            try:
                self.results = tuner.fit()
            except Exception as e:
                logger.error("HPO failed or interrupted: %s", e)
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
        """Build the Ray Tune search space."""
        space = build_chemeleon_search_space(
            self.config.search_space,
            target_columns=list(self.config.target_columns),
        )

        # Add fixed parameters
        space["data_path"] = str(Path(self.config.data_path).resolve())
        space["val_data_path"] = str(Path(self.config.val_data_path).resolve()) if self.config.val_data_path else None
        space["smiles_column"] = self.config.smiles_column
        space["target_columns"] = self.config.target_columns
        space["max_epochs"] = self.config.asha.max_t
        space["metric"] = self.config.asha.metric
        space["seed"] = self.config.seed
        space["checkpoint_path"] = self.config.checkpoint_path

        # Training parameters (use config values)
        space["patience"] = self.config.patience
        space["warmup_epochs"] = self.config.warmup_epochs
        space["report_every_n_epochs"] = self.config.report_every_n_epochs

        # Pass fixed target weights if provided (not in search space)
        if self.config.target_weights is not None:
            space["target_weights"] = self.config.target_weights

        # Pass unfreezing schedule defaults (only used if freeze_encoder=true in search space)
        space["unfreeze_encoder_epoch"] = self.config.unfreeze_schedule.unfreeze_encoder_epoch
        space["unfreeze_encoder_lr_multiplier"] = self.config.unfreeze_schedule.unfreeze_encoder_lr_multiplier

        return space

    def _build_scheduler(self) -> ASHAScheduler:
        """Build the ASHA scheduler."""
        return ASHAScheduler(
            time_attr="epoch",
            metric=self.config.asha.metric,
            mode=self.config.asha.mode,
            max_t=self.config.asha.max_t,
            grace_period=self.config.asha.grace_period,
            reduction_factor=self.config.asha.reduction_factor,
        )

    def _build_search_algorithm(self):
        """Build search algorithm from configuration.

        Returns
        -------
        Search algorithm or None for random search
        """
        search_type = self.config.search_algorithm.type.lower()

        if search_type == "random" or search_type is None:
            logger.info("Using random search (no search algorithm)")
            return None

        elif search_type == "optuna":
            import optuna
            from ray.tune.search.optuna import OptunaSearch

            sampler = optuna.samplers.TPESampler(
                seed=self.config.search_algorithm.seed,
                n_startup_trials=self.config.search_algorithm.n_initial_points,
            )
            search_alg = OptunaSearch(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
                sampler=sampler,
            )
            logger.info(
                "Using Optuna search (Bayesian optimization) with %d initial random trials",
                self.config.search_algorithm.n_initial_points,
            )
            return search_alg

        elif search_type == "bayesopt":
            from ray.tune.search.bayesopt import BayesOptSearch

            search_alg = BayesOptSearch(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
                random_state=self.config.search_algorithm.seed,
            )
            logger.info("Using BayesOpt search")
            return search_alg

        elif search_type == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            search_alg = HyperOptSearch(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
                random_state_seed=self.config.search_algorithm.seed,
            )
            logger.info("Using HyperOpt search")
            return search_alg

        else:
            logger.warning(
                "Unknown search algorithm: %s. Valid options: random, optuna, bayesopt, hyperopt\n"
                "Falling back to random search.",
                search_type,
            )
            return None

    @staticmethod
    def _null_context():
        """Return a no-op context manager for when logging is disabled."""
        from contextlib import nullcontext

        return nullcontext()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking.

        Uses a context manager to start the parent run and immediately exit
        the active run context. This is critical: the run stays open (RUNNING)
        but is no longer the "active" run, allowing AsyncBatchedMLflowCallback
        to create child runs with nested=False without conflicts.
        """
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        mlflow.set_experiment(self.config.experiment_name)

        # Use context manager to capture run_id and clear active run context
        # The run remains RUNNING but is not the "active" run, preventing
        # conflicts when AsyncBatchedMLflowCallback starts trial runs
        with mlflow.start_run(run_name=f"hpo_{self.timestamp}") as run:
            self._mlflow_run_id = run.info.run_id

            # Log all HPO configuration parameters
            params_to_log = {
                # Experiment metadata
                "experiment_name": self.config.experiment_name,
                "timestamp": self.timestamp,
                # Data configuration
                "data_path": str(self.config.data_path),
                "val_data_path": str(self.config.val_data_path) if self.config.val_data_path else None,
                "smiles_column": self.config.smiles_column,
                "target_columns": str(self.config.target_columns),
                "target_weights": str(self.config.target_weights) if self.config.target_weights else None,
                "seed": self.config.seed,
                # CheMeleon-specific settings
                "checkpoint_path": self.config.checkpoint_path,
                "freeze_encoder": self.config.freeze_encoder,
                # Unfreeze schedule
                "unfreeze_schedule.freeze_encoder": self.config.unfreeze_schedule.freeze_encoder,
                "unfreeze_schedule.unfreeze_encoder_epoch": self.config.unfreeze_schedule.unfreeze_encoder_epoch,
                "unfreeze_schedule.lr_multiplier": self.config.unfreeze_schedule.unfreeze_encoder_lr_multiplier,
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

            # Log the full config as a YAML artifact for complete reproducibility
            self._log_config_artifact()

    def _log_config_artifact(self) -> None:
        """Log the full HPO configuration as a YAML artifact."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to dict for YAML serialization
        config_dict = OmegaConf.to_container(OmegaConf.structured(self.config), resolve=True)

        config_path = output_dir / f"hpo_config_{self.timestamp}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(OmegaConf.create(config_dict), f)

        mlflow.log_artifact(str(config_path), artifact_path="config")
        logger.info("Logged HPO config artifact: %s", config_path)

    def _log_results(self) -> None:
        """Log HPO results to MLflow including detailed trial metrics.

        Logs:
        - Best trial configuration and metrics
        - Best model checkpoint
        - Summary statistics across all trials
        - HPO results dataframe and top-k configurations
        """
        if self.results is None:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            best_result = self.results.get_best_result(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )

            if best_result and best_result.config:
                # Filter out internal params and truncate values for MLflow limits
                best_params = {}
                for k, v in best_result.config.items():
                    if not k.startswith("_"):
                        str_v = str(v)
                        best_params[f"best.{k}"] = str_v[:500] if len(str_v) > 500 else str_v
                mlflow.log_params(best_params)

            if best_result and best_result.metrics:
                best_metrics = {
                    f"best.{k}": float(v) for k, v in best_result.metrics.items() if isinstance(v, (int, float))
                }
                mlflow.log_metrics(best_metrics)

            # Log best model checkpoint if available
            if best_result and best_result.checkpoint:
                try:
                    with best_result.checkpoint.as_directory() as checkpoint_dir:
                        checkpoint_path = Path(checkpoint_dir)
                        best_checkpoints = list(checkpoint_path.glob("best-*.ckpt"))
                        best_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                        if best_checkpoints:
                            ckpt_file = best_checkpoints[0]
                            logger.info("Logging best HPO model artifact: %s", ckpt_file.name)
                            mlflow.log_artifact(str(ckpt_file), artifact_path="best_model")
                        else:
                            logger.warning("No best-*.ckpt found in checkpoint: %s", checkpoint_path)
                except Exception as e:
                    logger.warning("Failed to log best model artifact: %s", e)

        except Exception as e:
            logger.warning("Could not log best result: %s", e)

        # Log detailed metrics from all trials
        try:
            results_df = self.results.get_dataframe()

            # Extract and log summary statistics for each metric across all trials
            metric_cols = [col for col in results_df.columns if col.startswith(("val_", "train_"))]

            for metric_col in metric_cols:
                if metric_col in results_df.columns:
                    values = results_df[metric_col].dropna()
                    if len(values) > 0:
                        try:
                            mlflow.log_metrics(
                                {
                                    f"trials.{metric_col}.mean": float(values.mean()),
                                    f"trials.{metric_col}.std": float(values.std()),
                                    f"trials.{metric_col}.min": float(values.min()),
                                    f"trials.{metric_col}.max": float(values.max()),
                                }
                            )
                        except Exception as e:
                            logger.debug("Could not compute stats for %s: %s", metric_col, e)

            logger.info("Logged detailed metrics from %d trials to MLflow", len(results_df))
        except Exception as e:
            logger.warning("Could not log detailed trial metrics: %s", e)

        # Save all results as CSV artifact
        try:
            results_df = self.results.get_dataframe()
            results_path = output_dir / f"hpo_results_{self.timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            logger.info("Logged HPO results artifact: %s", results_path)
        except Exception as e:
            logger.warning("Could not save results dataframe: %s", e)

        # Save top-k configs as JSON artifact
        try:
            top_k = self._get_top_k_configs()
            top_k_path = output_dir / f"top_k_configs_{self.timestamp}.json"
            with open(top_k_path, "w", encoding="utf-8") as f:
                json.dump(top_k, f, indent=2, default=str)
            mlflow.log_artifact(str(top_k_path))
            logger.info("Logged top-k configs artifact: %s", top_k_path)
        except Exception as e:
            logger.warning("Could not save top-k configs: %s", e)

        mlflow.end_run()
        logger.info("HPO results logged to MLflow")

    def _get_top_k_configs(self) -> list[dict[str, Any]]:
        """Get the top-k configurations from HPO results.

        Returns
        -------
        list[dict[str, Any]]
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
            "checkpoint_path",
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

    def save_top_configs(self, k: int | None = None) -> list[dict[str, Any]]:
        """Save top-k configurations to file.

        Parameters
        ----------
        k : int | None
            Number of top configs to save. Defaults to transfer_learning.top_k.

        Returns
        -------
        list[dict[str, Any]]
            List of top configurations.
        """
        if self.results is None:
            raise RuntimeError("No HPO results available.")

        k = k or self.config.transfer_learning.top_k

        top_configs = []
        for result in self.results:  # type: ignore[attr-defined]
            if result.config:
                top_configs.append(
                    {
                        "config": result.config,
                        "metrics": result.metrics,
                    }
                )

        # Sort by metric
        metric = self.config.asha.metric
        reverse = self.config.asha.mode == "max"
        top_configs.sort(key=lambda x: x["metrics"].get(metric, float("inf")), reverse=reverse)
        top_configs = top_configs[:k]

        # Save to file
        output_path = Path(self.config.output_dir) / f"top_{k}_configs_{self.timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_configs, f, indent=2, default=str)

        logger.info("Saved top %d configs to %s", k, output_path)
        return top_configs


def main() -> None:
    """CLI entry point for CheMeleon HPO."""
    configure_logging()

    parser = argparse.ArgumentParser(description="Run CheMeleon HPO with Ray Tune")
    parser.add_argument("--config", type=str, required=True, help="Path to HPO config YAML")
    args = parser.parse_args()

    raw_config = OmegaConf.load(args.config)
    merged_config = OmegaConf.merge(OmegaConf.structured(ChemeleonHPOConfig), raw_config)
    config = cast(ChemeleonHPOConfig, OmegaConf.to_object(merged_config))

    hpo = ChemeleonHPO(config)
    results = hpo.run()

    logger.info("HPO complete. %d trials finished.", len(results))
    hpo.save_top_configs()


if __name__ == "__main__":
    main()
