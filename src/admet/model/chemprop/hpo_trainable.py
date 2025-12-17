"""Ray Tune trainable function for Chemprop hyperparameter optimization.

This module provides the trainable function and callback for integrating
Chemprop training with Ray Tune's ASHA scheduler.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import ray.tune
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from ray.air import session
from ray.train import Checkpoint

from admet.model.chemprop.model import ChempropHyperparams, ChempropModel

logger = logging.getLogger("admet.model.chemprop.hpo_trainable")


class RayTuneReportCallback(Callback):
    """PyTorch Lightning callback to report metrics to Ray Tune.

    This callback integrates with Ray Tune's reporting mechanism to enable
    early stopping via the ASHA scheduler. It reports validation metrics
    after each epoch including comprehensive correlation metrics.

    It also saves checkpoints for trial recovery if a trial crashes.

    Attributes:
        metric: Name of the primary metric for ASHA scheduling (default: "val_mae")
        checkpoint_dir: Directory to save checkpoints for recovery
        report_every_n_epochs: Epoch cadence for Ray reporting (default: 5)
    """

    # All metrics to report to Ray Tune for tracking
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
        "lr",  # Current learning rate
    )

    def __init__(
        self,
        metric: str = "val_mae",
        checkpoint_dir: Path | None = None,
        report_every_n_epochs: int = 5,
    ) -> None:
        """Initialize the callback.

        Args:
            metric: Name of the primary validation metric for ASHA scheduling.
            checkpoint_dir: Directory for saving checkpoints. If None, no checkpoints saved.
            report_every_n_epochs: Number of epochs between Ray reports. Must be >= 1.
        """
        super().__init__()
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        if report_every_n_epochs < 1:
            raise ValueError("report_every_n_epochs must be >= 1")
        self.report_every_n_epochs = report_every_n_epochs
        self._last_reported_epoch: int | None = None
        self._final_checkpoint_reported = False

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics to Ray Tune after validation.

        Reports all available metrics from METRICS_TO_REPORT plus the primary
        metric used for ASHA scheduling. Also saves checkpoint for recovery.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: The Lightning module being trained.
        """
        # Skip if no logged metrics
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

        # Extract all available metrics
        metrics: dict[str, float] = {}

        # Always report epoch
        metrics["epoch"] = float(trainer.current_epoch)

        # Report all tracked metrics that are available
        for metric_name in self.METRICS_TO_REPORT:
            if metric_name in trainer.callback_metrics:
                value = trainer.callback_metrics[metric_name]
                metrics[metric_name] = float(value.item() if hasattr(value, "item") else value)

        # Try to get current learning rate from optimizer
        if "lr" not in metrics and trainer.optimizers:
            try:
                opt = trainer.optimizers[0]
                if hasattr(opt, "param_groups") and opt.param_groups:
                    metrics["lr"] = float(opt.param_groups[0]["lr"])
            except (IndexError, KeyError):
                pass

        # Ensure primary metric is present (map val_loss to val_mae if needed)
        if self.metric not in metrics:
            # Fallback mapping for primary metric
            fallback_mapping = {
                "val_mae": ["val_loss"],
                "val_loss": ["val_mae"],
            }
            for fallback_key in fallback_mapping.get(self.metric, []):
                if fallback_key in trainer.callback_metrics:
                    value = trainer.callback_metrics[fallback_key]
                    metrics[self.metric] = float(value.item() if hasattr(value, "item") else value)
                    break

        # Report to Ray Tune only if primary metric is available
        if self.metric in metrics:
            checkpoint: Checkpoint | None = None
            if is_final_event and not self._final_checkpoint_reported:
                checkpoint = self._build_final_checkpoint(trainer)
                if checkpoint is not None:
                    self._final_checkpoint_reported = True

            self._submit_report(metrics, checkpoint)
            self._last_reported_epoch = epoch_index

    def _build_final_checkpoint(self, trainer: pl.Trainer) -> Checkpoint | None:
        """Package the latest checkpoint directory for Ray Tune reporting.

        Prefers the best-*.ckpt file if available, then falls back to
        last.ckpt, and finally triggers a manual checkpoint save if
        neither exists. The selected checkpoint is copied into a
        dedicated ray_checkpoint directory to keep uploads minimal.

        Args:
            trainer: Active Lightning trainer (used for manual checkpointing).

        Returns:
            Ray AIR Checkpoint instance or None if no checkpoint could be created.
        """

        if self.checkpoint_dir is None:
            return None

        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.exists():
            return None

        # Find preferred checkpoint file
        best_checkpoints = sorted(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoint_file: Path | None = best_checkpoints[0] if best_checkpoints else None

        if checkpoint_file is None:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                checkpoint_file = last_ckpt

        if checkpoint_file is None:
            # Create a final checkpoint manually as a last resort
            try:
                final_ckpt = ckpt_dir / "ray-final.ckpt"
                trainer.save_checkpoint(str(final_ckpt))
                checkpoint_file = final_ckpt
            except Exception:
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
        """Send metrics to Ray using the most compatible API available."""

        try:
            if checkpoint is not None:
                session.report(metrics, checkpoint=checkpoint)
            else:
                session.report(metrics)
            return
        except Exception as exc:
            logger.debug("session.report failed, falling back to tune.report: %s", exc)

        if checkpoint is not None:
            ray.tune.report(checkpoint=checkpoint, **metrics)
        else:
            ray.tune.report(**metrics)


def train_chemprop_trial(config: dict[str, Any]) -> None:
    """Ray Tune trainable function for a single Chemprop HPO trial.

    This function is called by Ray Tune for each trial. It constructs
    a ChempropModel with the sampled hyperparameters and trains it,
    reporting validation metrics for early stopping.

    Args:
        config: Dictionary containing:
            - Sampled hyperparameters (learning_rate, dropout, etc.)
            - Per-target weight parameters (target_weight_<target_name>)
            - Fixed parameters passed via param_space:
                - data_path: Path to training data CSV
                - val_data_path: Path to validation data CSV (optional)
                - smiles_column: SMILES column name
                - target_columns: List of target columns
                - max_epochs: Maximum training epochs
                - metric: Metric to report (default: val_mae)
                - seed: Random seed
                    - report_every_n_epochs: Epoch cadence for Ray reports (default: 5)
    """
    # Extract fixed parameters
    data_path = config.get("data_path")
    val_data_path = config.get("val_data_path")
    smiles_column = config.get("smiles_column", "smiles")
    target_columns = config.get("target_columns", [])
    max_epochs = config.get("max_epochs", 150)
    metric = config.get("metric", "val_mae")
    report_every_n_epochs_raw = config.get("report_every_n_epochs", 5)
    try:
        report_every_n_epochs = max(1, int(report_every_n_epochs_raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid report_every_n_epochs value %s, defaulting to 5",
            report_every_n_epochs_raw,
        )
        report_every_n_epochs = 5
    seed = config.get("seed", 42)

    # Load data
    df_train = pd.read_csv(data_path)
    df_val = pd.read_csv(val_data_path) if val_data_path else None

    # Build hyperparameters from sampled config
    hyperparams = _build_hyperparams(config, max_epochs, seed)

    # Extract target weights from config
    # Check for fixed weights first, then fall back to per-task weights
    if "target_weights" in config and config["target_weights"] is not None:
        target_weights = config["target_weights"]
    else:
        target_weights = _extract_target_weights(config, target_columns)

    # Create output directory in Ray's trial directory (Ray Tune / Ray Train session)
    trial_dir = None

    # Preferred: Ray Train context API (Ray >=2.x)
    try:
        trial_dir_str = ray.tune.get_context().get_trial_dir()
        if trial_dir_str:
            trial_dir = Path(trial_dir_str)
    except Exception:
        # Not running within a Ray Train context
        trial_dir = None

    # Fallback: Ray AIR (session) API if available (older/newer AIR APIs)
    if trial_dir is None:
        try:
            # session is the ray.air.session module which exposes helper functions
            trial_dir_str = session.get_trial_dir()
            if trial_dir_str:
                trial_dir = Path(trial_dir_str)
        except Exception:
            trial_dir = None

    # Final fallback: use current working dir (useful for unit tests)
    if trial_dir is None:
        trial_dir = Path.cwd() / "ray_trial"

    output_dir = trial_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model (disable MLflow tracking - HPO orchestrator handles logging)
    model = ChempropModel(
        df_train=df_train,
        df_validation=df_val,
        smiles_col=smiles_column,
        target_cols=target_columns,
        target_weights=target_weights,
        output_dir=output_dir,
        progress_bar=False,
        hyperparams=hyperparams,
        mlflow_tracking=False,  # Disable MLflow in individual trials
    )

    # Add Ray Tune callback with checkpoint directory for trial recovery
    ray_callback = RayTuneReportCallback(
        metric=metric,
        checkpoint_dir=output_dir,
        report_every_n_epochs=report_every_n_epochs,
    )
    model.trainer.callbacks.append(ray_callback)  # type: ignore[attr-defined]

    # Train the model
    model.fit()


def _extract_target_weights(
    config: dict[str, Any],
    target_columns: list[str],
) -> list[float]:
    """Extract target weights from HPO config.

    Looks for parameters named target_weight_<safe_target_name> and
    assembles them into a list matching the target_columns order.

    Args:
        config: Sampled hyperparameter configuration from Ray Tune.
        target_columns: List of target column names.

    Returns:
        List of target weights in same order as target_columns.
        Defaults to 1.0 for any missing weights.
    """
    weights = []
    for target in target_columns:
        # Create safe parameter name matching build_search_space
        safe_name = target.replace(" ", "_").replace(">", "gt").replace("<", "lt")
        param_name = f"target_weight_{safe_name}"
        weight = config.get(param_name, 1.0)
        weights.append(float(weight) if weight is not None else 1.0)
    return weights


def _build_hyperparams(
    config: dict[str, Any],
    max_epochs: int,
    seed: int,
) -> ChempropHyperparams:
    """Build ChempropHyperparams from sampled HPO config.

    Maps the flat HPO config dictionary to the structured
    ChempropHyperparams dataclass.

    Args:
        config: Sampled hyperparameter configuration from Ray Tune.
        max_epochs: Maximum training epochs.
        seed: Random seed for reproducibility.

    Returns:
        ChempropHyperparams instance with sampled values.
    """
    # Start with defaults
    params: dict[str, Any] = {
        "max_epochs": max_epochs,
        "seed": seed,
    }

    # Map sampled hyperparameters
    # Learning rate schedule (max_lr with configurable warmup/final ratios)
    if "learning_rate" in config and config["learning_rate"] is not None:
        lr = config["learning_rate"]
        params["max_lr"] = lr

        # Get warmup ratio (init_lr = max_lr * warmup_ratio, typically 0.01-0.1)
        warmup_ratio = config.get("lr_warmup_ratio", 0.1)
        params["init_lr"] = lr * warmup_ratio

        # Get final ratio (final_lr = max_lr * final_ratio, typically 0.01-0.1)
        final_ratio = config.get("lr_final_ratio", 0.1)
        params["final_lr"] = lr * final_ratio

    if "dropout" in config and config["dropout"] is not None:
        params["dropout"] = config["dropout"]

    # Message passing architecture
    if "depth" in config and config["depth"] is not None:
        params["depth"] = int(config["depth"])

    # Message hidden dim (MPNN layer width)
    if "message_hidden_dim" in config and config["message_hidden_dim"] is not None:
        params["message_hidden_dim"] = int(config["message_hidden_dim"])
    elif "hidden_dim" in config and config["hidden_dim"] is not None:
        # Fallback: use hidden_dim for message_hidden_dim if not specified separately
        params["message_hidden_dim"] = int(config["hidden_dim"])

    # FFN architecture
    if "ffn_num_layers" in config and config["ffn_num_layers"] is not None:
        params["num_layers"] = int(config["ffn_num_layers"])

    # FFN hidden dim (can be different from message_hidden_dim)
    if "ffn_hidden_dim" in config and config["ffn_hidden_dim"] is not None:
        params["hidden_dim"] = int(config["ffn_hidden_dim"])
    elif "hidden_dim" in config and config["hidden_dim"] is not None:
        params["hidden_dim"] = int(config["hidden_dim"])

    if "batch_size" in config and config["batch_size"] is not None:
        params["batch_size"] = int(config["batch_size"])

    # FFN type (map HPO names to Chemprop names)
    ffn_type_mapping = {
        "mlp": "regression",
        "moe": "mixture_of_experts",
        "branched": "branched",
        # Also accept original names
        "regression": "regression",
        "mixture_of_experts": "mixture_of_experts",
    }
    if "ffn_type" in config and config["ffn_type"] is not None:
        ffn_type = config["ffn_type"]
        params["ffn_type"] = ffn_type_mapping.get(ffn_type, ffn_type)

    # MoE-specific parameters
    if "n_experts" in config and config["n_experts"] is not None:
        params["n_experts"] = int(config["n_experts"])

    # Branched FFN parameters
    if "trunk_depth" in config and config["trunk_depth"] is not None:
        params["trunk_n_layers"] = int(config["trunk_depth"])

    if "trunk_hidden_dim" in config and config["trunk_hidden_dim"] is not None:
        params["trunk_hidden_dim"] = int(config["trunk_hidden_dim"])

    return ChempropHyperparams(**params)
