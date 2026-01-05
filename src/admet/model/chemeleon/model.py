"""Chemeleon foundation model implementation.

This module provides the ChemeleonModel class which wraps the pre-trained
Chemeleon molecular encoder for property prediction tasks.

Chemeleon uses a message-passing neural network encoder pre-trained on large
molecular datasets. By default, the encoder is frozen and only the prediction
head is trained, enabling effective transfer learning.

Key features:
- Auto-download of pre-trained weights from Zenodo
- Frozen encoder by default for efficient transfer learning
- Optional gradual unfreezing schedule
- Consistent BaseModel interface

References:
- Chemeleon paper and documentation at chemprop.readthedocs.io
- Zenodo weights: https://zenodo.org/records/15460715
"""

from __future__ import annotations

import fcntl
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from ray import tune
from torch.utils.data import DataLoader

from admet.data.stats import correlation
from admet.model.base import BaseModel
from admet.model.chemeleon.callbacks import GradualUnfreezeCallback
from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.joint_sampler import JointSampler
from admet.model.config import UnfreezeScheduleConfig
from admet.model.ffn_factory import create_ffn_predictor
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry
from admet.util.profiling import TrainingPhase, TrainingProfiler, create_lightning_profiling_callback

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Zenodo URL for pre-trained Chemeleon weights
ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "admet" / "chemeleon"


def _get_dataloader_kwargs(num_workers: int, is_train: bool = True) -> dict[str, Any]:
    """
    Get optimized DataLoader kwargs for performance.

    Returns kwargs that enable GPU training optimizations when appropriate:
    - pin_memory: Pre-loads data to GPU pinned memory for faster transfers
    - persistent_workers: Keeps workers alive between epochs (reduces startup overhead)
    - prefetch_factor: Number of batches to prefetch per worker

    Parameters
    ----------
    num_workers : int
        Number of data loading workers.
    is_train : bool, default=True
        Whether this is for training (enables more aggressive prefetching).

    Returns
    -------
    dict[str, Any]
        Kwargs to pass to DataLoader constructor.
    """
    kwargs: dict[str, Any] = {}

    # Enable pin_memory for GPU training (faster CPU->GPU transfers)
    if torch.cuda.is_available():
        kwargs["pin_memory"] = True

    # Enable persistent_workers and prefetch_factor when using multiprocessing
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        # Prefetch 2 batches per worker for training, 1 for validation
        kwargs["prefetch_factor"] = 2 if is_train else 1

    return kwargs


class CorrelationMetricsCallback(pl.Callback):
    """Callback to compute and log correlation metrics during validation.

    Computes RAE, Pearson r, Spearman ρ, and Kendall τ metrics on validation
    outputs and logs them to MLflow and Ray Tune for HPO tracking.

    Attributes
    ----------
    scaler : Any
        Data scaler to inverse-transform predictions
    target_cols : list[str]
        Target column names
    val_loader : DataLoader | None
        Validation dataloader for computing metrics
    report_every_n_epochs : int
        Epoch cadence for reporting (default: 5)
    """

    def __init__(
        self,
        scaler: Any = None,
        target_cols: list[str] | None = None,
        val_loader: DataLoader | None = None,
        report_every_n_epochs: int = 5,
        compute_rank_correlations: bool = True,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        scaler : Any
            Data scaler for inverse-transforming predictions
        target_cols : list[str] | None
            Target column names
        val_loader : DataLoader | None
            Validation dataloader for computing metrics
        report_every_n_epochs : int
            Epoch cadence for Ray reporting (default: 5)
        compute_rank_correlations : bool, default=True
            Whether to compute expensive rank correlations (Spearman, Kendall).
            Set to False to speed up metrics computation by 30-50%.
        """
        super().__init__()
        self.scaler = scaler
        self.target_cols = target_cols or []
        self.val_loader = val_loader
        self.report_every_n_epochs = max(1, report_every_n_epochs)
        self.compute_rank_correlations = compute_rank_correlations
        self._last_reported_epoch: int | None = None

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute correlation metrics after validation epoch."""
        if self.val_loader is None:
            return

        epoch_index = int(trainer.current_epoch) + 1

        # Skip reporting based on epoch cadence
        if (
            self._last_reported_epoch is not None
            and epoch_index - self._last_reported_epoch < self.report_every_n_epochs
        ):
            return

        try:
            # Collect predictions and targets from validation dataloader
            y_true_list = []
            y_pred_list = []

            pl_module.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    if isinstance(batch, (list, tuple)):
                        # Chemprop batch format: (bmg, V_d, X_d, targets, ...)
                        if len(batch) >= 4:
                            bmg, V_d, X_d, targets = batch[0], batch[1], batch[2], batch[3]

                            # Move to device
                            device = next(pl_module.parameters()).device
                            if hasattr(bmg, "to"):
                                bmg.to(device)
                            if V_d is not None and hasattr(V_d, "to"):
                                V_d = V_d.to(device)
                            if X_d is not None and hasattr(X_d, "to"):
                                X_d = X_d.to(device)
                            targets = targets.to(device) if hasattr(targets, "to") else targets

                            # Get predictions
                            preds = pl_module(bmg, V_d, X_d)

                            # Handle shape: squeeze last dim if it's 1 (single-task case)
                            if preds.ndim == 3 and preds.shape[-1] == 1:
                                preds = preds[..., 0]

                            y_true_list.append(targets.cpu().numpy())
                            y_pred_list.append(preds.cpu().numpy())
                        else:
                            logger.warning("Unexpected batch format, skipping metrics computation")
                            return
                    elif isinstance(batch, dict):
                        # Dictionary format
                        if "y_true" in batch and "y_pred" in batch:
                            y_t = batch["y_true"]
                            y_p = batch["y_pred"]
                            # Keep as tensors for GPU-native metric computation
                            y_true_list.append(y_t)
                            y_pred_list.append(y_p)
                    else:
                        logger.warning("Unsupported batch format, skipping metrics computation")
                        return

            if not y_true_list or not y_pred_list:
                return

            # Concatenate predictions and targets (keep as tensors)
            import torch

            y_true = torch.cat(y_true_list, dim=0)
            y_pred = torch.cat(y_pred_list, dim=0)

            # Unscale if scaler is available (convert to numpy only if needed)
            if self.scaler is not None:
                y_true_np = y_true.cpu().numpy()
                y_pred_np = y_pred.cpu().numpy()
                y_true_np = self.scaler.inverse_transform(y_true_np)
                y_pred_np = self.scaler.inverse_transform(y_pred_np)
                y_true = torch.from_numpy(y_true_np).to(y_true.device)
                y_pred = torch.from_numpy(y_pred_np).to(y_pred.device)

            # Ensure 2D tensors
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(1)
            if y_pred.ndim == 1:
                y_pred = y_pred.unsqueeze(1)

            # Compute metrics for each target
            metrics_dict: dict[str, float] = {}

            for task_idx in range(y_true.shape[1]):
                task_name = self.target_cols[task_idx] if task_idx < len(self.target_cols) else f"target_{task_idx}"
                y_t = y_true[:, task_idx]
                y_p = y_pred[:, task_idx]

                # Compute correlation metrics using torch tensors (GPU-native)
                metrics = correlation(y_t, y_p, compute_rank_correlations=self.compute_rank_correlations)

                # Log metrics with task prefix
                for metric_name, metric_value in metrics.items():
                    if not np.isnan(metric_value):
                        mlflow_key = f"val_{metric_name}_{task_name}"
                        metrics_dict[mlflow_key] = float(metric_value)

            # Also compute overall metrics (pooled across tasks) - keep as tensors
            y_true_pool = y_true.flatten()
            y_pred_pool = y_pred.flatten()
            metrics_pool = correlation(
                y_true_pool, y_pred_pool, compute_rank_correlations=self.compute_rank_correlations
            )

            for metric_name, metric_value in metrics_pool.items():
                if not np.isnan(metric_value):
                    mlflow_key = f"val_{metric_name}_overall"
                    metrics_dict[mlflow_key] = float(metric_value)

            # Log to MLflow
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics_dict, step=epoch_index)

            # Log to Ray Tune for HPO
            try:
                tune.report(**metrics_dict)
            except Exception:
                pass

            self._last_reported_epoch = epoch_index
            logger.info("Logged correlation metrics at epoch %d", epoch_index)

        except Exception as e:
            logger.warning("Failed to compute correlation metrics: %s", e)


@ModelRegistry.register("chemeleon")
class ChemeleonModel(BaseModel, MLflowMixin):
    """Chemeleon foundation model for molecular property prediction.

    This model wraps the pre-trained Chemeleon message-passing encoder with
    a trainable regression head. By default, the encoder is frozen to enable
    efficient transfer learning.

    Parameters
    ----------
    config : DictConfig
        Configuration containing model and training parameters.
        Expected structure:
        - model.type: "chemeleon"
        - model.chemeleon: ChemeleonModelParams
        - data: DataConfig
        - optimization: OptimizationConfig (optional)
        - mlflow: MLflowConfig (optional)

    Attributes
    ----------
    model_type : str
        Model type identifier ("chemeleon").
    mp : nn.BondMessagePassing
        Pre-trained message passing encoder.
    ffn : nn.RegressionFFN
        Trainable regression head.
    mpnn : models.MPNN
        Combined MPNN model.

    Examples
    --------
    Create and train a Chemeleon model:

    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.create({
    ...     "model": {
    ...         "type": "chemeleon",
    ...         "chemeleon": {
    ...             "freeze_encoder": True,
    ...             "ffn_hidden_dim": 300,
    ...         },
    ...     },
    ...     "data": {"target_cols": ["LogD"]},
    ...     "mlflow": {"enabled": False},
    ... })
    >>> model = ChemeleonModel(config)
    >>> model.fit(train_smiles, train_y)
    >>> predictions = model.predict(test_smiles)
    """

    model_type = "chemeleon"

    def __init__(self, config: DictConfig) -> None:
        """Initialize Chemeleon model.

        Parameters
        ----------
        config : DictConfig
            Configuration object.
        """
        super().__init__(config)

        # Get model params - support both new and legacy config structures
        model_section = config.get("model", OmegaConf.create({}))
        if "chemeleon" in model_section:
            self._model_params = model_section.chemeleon
        else:
            # Legacy: params directly in model section
            self._model_params = model_section

        # Initialize components (deferred until fit to know target count)
        self.mp: nn.BondMessagePassing | None = None
        self.ffn: nn.RegressionFFN | None = None
        self.mpnn: models.MPNN | None = None
        self.trainer: pl.Trainer | None = None
        self.scaler: Any = None
        self.featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer | None = None
        self.agg: nn.MeanAggregation | None = None

        self._smiles_col = config.get("data", {}).get("smiles_col", "smiles")
        self._target_cols: list[str] = list(config.get("data", {}).get("target_cols", []))
        self._target_weights: list[float] = list(config.get("data", {}).get("target_weights", []) or [])
        self._quality_col: str = config.get("data", {}).get("quality_col", "quality")

        # Checkpoint directory (created during training)
        self._checkpoint_dir: tempfile.TemporaryDirectory | None = None

        # JointSampler for curriculum/task-aware sampling
        self._joint_sampler: Optional[JointSampler] = None
        self._curriculum_state: Optional[CurriculumState] = None

        # Unfreeze callback
        unfreeze_config = self._get_unfreeze_config()
        self._unfreeze_callback = GradualUnfreezeCallback(unfreeze_config)

        # Correlation metrics callback (for RAE, Pearson, Spearman, Kendall)
        # Check if compute_rank_correlations is specified in model params
        compute_rank = self._model_params.get("compute_rank_correlations", True)
        self._correlation_metrics_callback = CorrelationMetricsCallback(
            scaler=None,  # Will be set after scaler is created
            target_cols=self._target_cols,
            compute_rank_correlations=compute_rank,
        )

        # External callbacks (e.g., for HPO integration)
        self._external_callbacks: list[pl.Callback] = []

        # Initialize profiler for tracking phase-level performance
        self._profiler = TrainingProfiler(
            name="chemeleon",
            enabled=True,
        )

    def add_callback(self, callback: pl.Callback) -> None:
        """Add an external callback to be used during training.

        This allows external integrations (e.g., Ray Tune HPO) to inject
        callbacks for metric reporting without modifying the training loop.

        Parameters
        ----------
        callback : pl.Callback
            PyTorch Lightning callback to add.
        """
        self._external_callbacks.append(callback)

    def _get_model_param(self, key: str, default: Any) -> Any:
        """Get model parameter with default."""
        return self._model_params.get(key, default)

    def _get_unfreeze_config(self) -> UnfreezeScheduleConfig:
        """Get unfreeze schedule configuration.

        Returns
        -------
        UnfreezeScheduleConfig
            Unfreeze schedule configuration.
        """
        unfreeze_section = self._model_params.get("unfreeze_schedule", {})
        return UnfreezeScheduleConfig(
            freeze_encoder=unfreeze_section.get("freeze_encoder", True),
            freeze_decoder_initially=unfreeze_section.get("freeze_decoder_initially", False),
            unfreeze_encoder_epoch=unfreeze_section.get("unfreeze_encoder_epoch"),
            unfreeze_decoder_epoch=unfreeze_section.get("unfreeze_decoder_epoch"),
            unfreeze_encoder_lr_multiplier=unfreeze_section.get("unfreeze_encoder_lr_multiplier", 0.1),
        )

    def _init_model(self, n_tasks: int) -> None:
        """Initialize model components.

        Parameters
        ----------
        n_tasks : int
            Number of prediction tasks.
        """
        # Load pre-trained message passing
        checkpoint_path = self._get_model_param("checkpoint_path", "auto")
        self.mp = self._load_pretrained_mp(checkpoint_path)

        # Freeze encoder if configured
        if self._get_model_param("freeze_encoder", True):
            self._freeze_encoder()

        # Initialize featurizer and aggregation
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.agg = nn.MeanAggregation()

        # Initialize FFN using shared factory
        # Cast to int since Ray Tune may sample floats for integer hyperparameters
        ffn_type = self._get_model_param("ffn_type", "regression")
        ffn_hidden_dim = self._get_model_param("ffn_hidden_dim", 300)
        ffn_num_layers = self._get_model_param("ffn_num_layers", 2)
        n_experts = self._get_model_param("n_experts", None)
        trunk_n_layers = self._get_model_param("trunk_n_layers", None)
        trunk_hidden_dim = self._get_model_param("trunk_hidden_dim", None)

        self.ffn = create_ffn_predictor(
            ffn_type=ffn_type,
            input_dim=self.mp.output_dim,
            n_tasks=n_tasks,
            hidden_dim=int(ffn_hidden_dim) if ffn_hidden_dim is not None else 300,
            n_layers=int(ffn_num_layers) if ffn_num_layers is not None else 2,
            dropout=self._get_model_param("dropout", 0.0),
            n_experts=int(n_experts) if n_experts is not None else None,
            trunk_n_layers=int(trunk_n_layers) if trunk_n_layers is not None else None,
            trunk_hidden_dim=int(trunk_hidden_dim) if trunk_hidden_dim is not None else None,
        )

        # Set default target weights if not provided
        if not self._target_weights:
            self._target_weights = [1.0] * n_tasks

        # Create metrics with task weights for proper multi-task loss weighting
        metrics = [
            nn.metrics.MAE(self._target_weights),
            nn.metrics.MSE(self._target_weights),
            nn.metrics.RMSE(self._target_weights),
            nn.metrics.R2Score(self._target_weights),
        ]

        # Get learning rate parameters from optimization config
        opt_config = self.config.get("optimization", {})
        max_lr = opt_config.get("learning_rate", 1e-4)
        warmup_ratio = opt_config.get("lr_warmup_ratio", 1.0)
        final_ratio = opt_config.get("lr_final_ratio", 1.0)
        init_lr = max_lr * warmup_ratio
        final_lr = max_lr * final_ratio
        warmup_epochs = opt_config.get("warmup_epochs", 2)

        # Create MPNN with metrics and learning rate schedule
        self.mpnn = models.MPNN(
            message_passing=self.mp,
            agg=self.agg,
            predictor=self.ffn,
            batch_norm=self._get_model_param("batch_norm", False),
            metrics=metrics,
            warmup_epochs=warmup_epochs,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
        )

    def _load_pretrained_mp(self, path: str) -> nn.BondMessagePassing:
        """Load pre-trained message passing from checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint or "auto" to download from Zenodo.

        Returns
        -------
        nn.BondMessagePassing
            Loaded message passing module.
        """
        if path == "auto":
            path = self._download_from_zenodo()

        logger.info("Loading Chemeleon checkpoint from %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Extract hyperparameters and state dict
        if "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]
            state_dict = checkpoint["state_dict"]
        else:
            # Assume raw state dict
            hyper_params = {"d_h": 300, "depth": 3}
            state_dict = checkpoint

        mp = nn.BondMessagePassing(**hyper_params)
        mp.load_state_dict(state_dict)

        return mp

    def _download_from_zenodo(self) -> str:
        """Download checkpoint from Zenodo with file locking for parallel safety.

        Uses file locking to prevent race conditions when multiple Ray workers
        attempt to download the checkpoint simultaneously.

        Returns
        -------
        str
            Path to downloaded checkpoint.
        """
        cache_dir = DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / "chemeleon_mp.pt"
        lock_path = cache_dir / "chemeleon_mp.pt.lock"

        # Use file locking to prevent concurrent downloads
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if not checkpoint_path.exists() or not self._validate_checkpoint(checkpoint_path):
                    logger.info("Downloading Chemeleon checkpoint from %s", ZENODO_URL)
                    # Download to temp file first, then move atomically
                    temp_path = checkpoint_path.with_suffix(".pt.tmp")
                    try:
                        urllib.request.urlretrieve(ZENODO_URL, temp_path)
                        # Validate downloaded file before moving
                        if self._validate_checkpoint(temp_path):
                            temp_path.rename(checkpoint_path)
                            logger.info("Downloaded to %s", checkpoint_path)
                        else:
                            temp_path.unlink(missing_ok=True)
                            raise RuntimeError("Downloaded checkpoint is corrupted")
                    except Exception:
                        temp_path.unlink(missing_ok=True)
                        raise
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return str(checkpoint_path)

    def _validate_checkpoint(self, path: Path) -> bool:
        """Validate that a checkpoint file is a valid PyTorch archive.

        Parameters
        ----------
        path : Path
            Path to checkpoint file.

        Returns
        -------
        bool
            True if the checkpoint is valid, False otherwise.
        """
        try:
            # Try to load just the file headers to validate structure
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception as e:
            logger.warning("Checkpoint validation failed for %s: %s", path, e)
            return False

    def _freeze_encoder(self) -> None:
        """Freeze message passing encoder."""
        if self.mp is not None:
            self.mp.eval()
            for param in self.mp.parameters():
                param.requires_grad = False
            logger.info("Froze Chemeleon encoder")

    def _unfreeze_encoder(self) -> None:
        """Unfreeze message passing encoder."""
        if self.mp is not None:
            self.mp.train()
            for param in self.mp.parameters():
                param.requires_grad = True
            logger.info("Unfroze Chemeleon encoder")

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
        quality_labels: list[str] | None = None,
    ) -> "ChemeleonModel":
        """Train the model.

        Parameters
        ----------
        smiles : list[str]
            Training SMILES strings.
        y : np.ndarray
            Training target values. Shape: (n_samples,) or (n_samples, n_tasks).
        val_smiles : list[str] | None, optional
            Validation SMILES strings.
        val_y : np.ndarray | None, optional
            Validation target values.
        quality_labels : list[str] | None, optional
            Quality labels for curriculum learning (e.g., ["high", "medium", "low"]).
            Required if joint_sampling.curriculum.enabled is True.

        Returns
        -------
        ChemeleonModel
            Self, for method chaining.
        """
        # Start profiler for overall training timing
        self._profiler.start()

        # Initialize MLflow if enabled
        with self._profiler.phase(TrainingPhase.MLFLOW_INIT):
            self.init_mlflow(run_name=self.config.get("mlflow", {}).get("run_name"))

        # Determine number of tasks
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_tasks = y.shape[1]

        # Initialize model (includes loading pretrained checkpoint)
        with self._profiler.phase(TrainingPhase.MODEL_INIT):
            self._init_model(n_tasks)

        # Set target columns if not specified
        if not self._target_cols:
            self._target_cols = [f"target_{i}" for i in range(n_tasks)]

        # Create datasets
        with self._profiler.phase(TrainingPhase.DATASET_CREATION):
            train_dataset = self._create_dataset(smiles, y)
            val_dataset = None
            if val_smiles is not None and val_y is not None:
                if val_y.ndim == 1:
                    val_y = val_y.reshape(-1, 1)
                val_dataset = self._create_dataset(val_smiles, val_y)

        # Scale targets
        with self._profiler.phase(TrainingPhase.TARGET_SCALING):
            self.scaler = train_dataset.normalize_targets()
            if val_dataset is not None:
                val_dataset.normalize_targets(self.scaler)

        # Update correlation metrics callback with scaler
        self._correlation_metrics_callback.scaler = self.scaler

        # Create dataloaders with optional JointSampler
        opt_config = self.config.get("optimization", {})
        batch_size = opt_config.get("batch_size", 32)
        num_workers = opt_config.get("num_workers", 0)

        with self._profiler.phase(TrainingPhase.DATALOADER_CREATION):
            train_loader = self._create_train_dataloader(
                train_dataset,
                y,
                quality_labels,
                batch_size,
                num_workers,
            )

            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=data.collate_batch,
                    **_get_dataloader_kwargs(num_workers, is_train=False),
                )

        # Update correlation metrics callback with validation loader
        self._correlation_metrics_callback.val_loader = val_loader

        # Setup trainer
        with self._profiler.phase(TrainingPhase.TRAINER_SETUP):
            self._setup_trainer()

        # Add profiling callback
        profiling_callback = create_lightning_profiling_callback(self._profiler)
        if self.trainer is not None and profiling_callback not in self.trainer.callbacks:
            self.trainer.callbacks.append(profiling_callback)

        # Log params
        self.log_params_from_config()

        # Train
        if self.trainer is None:
            msg = "Trainer not initialized"
            raise RuntimeError(msg)

        try:
            with self._profiler.phase(TrainingPhase.TRAINING_TOTAL):
                self.trainer.fit(
                    self.mpnn,  # type: ignore[arg-type]
                    train_loader,
                    val_loader,
                )

            # Load best checkpoint weights (PyTorch Lightning does NOT auto-restore)
            with self._profiler.phase(TrainingPhase.BEST_CHECKPOINT_LOAD):
                self._load_best_checkpoint()

            self._fitted = True

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving profiling info...")
        except Exception as e:
            logger.error("Training failed with error: %s", e)
            raise
        finally:
            # Clean up checkpoint directory
            with self._profiler.phase(TrainingPhase.CLEANUP):
                if self._checkpoint_dir is not None:
                    self._checkpoint_dir.cleanup()
                    self._checkpoint_dir = None

            # Stop profiler and print summary (always, even on interrupt)
            self._profiler.stop()
            self._profiler.print_summary()

            # Log profiling metrics to MLflow
            if self._mlflow_client is not None and self._mlflow_run_id is not None:
                self._profiler.log_to_mlflow(
                    prefix="profiling",
                    client=self._mlflow_client,
                    run_id=self._mlflow_run_id,
                )

            # End MLflow run
            self.end_mlflow()

        return self

    def _create_dataset(self, smiles: list[str], y: np.ndarray) -> data.MoleculeDataset:
        """Create MoleculeDataset from SMILES and targets.

        Parameters
        ----------
        smiles : list[str]
            SMILES strings.
        y : np.ndarray
            Target values.

        Returns
        -------
        data.MoleculeDataset
            Chemprop dataset.
        """
        datapoints = []
        for smi, targets in zip(smiles, y):
            target_list = targets.tolist() if isinstance(targets, np.ndarray) else [targets]
            mol_data = data.MoleculeDatapoint.from_smi(smi, target_list)
            datapoints.append(mol_data)

        return data.MoleculeDataset(datapoints, featurizer=self.featurizer)

    def _create_train_dataloader(
        self,
        dataset: data.MoleculeDataset,
        targets: np.ndarray,
        quality_labels: list[str] | None,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        """Create training DataLoader with optional JointSampler.

        Parameters
        ----------
        dataset : data.MoleculeDataset
            Training dataset.
        targets : np.ndarray
            Target values for JointSampler.
        quality_labels : list[str] | None
            Quality labels for curriculum learning.
        batch_size : int
            Batch size.
        num_workers : int
            Number of workers.

        Returns
        -------
        DataLoader
            Training DataLoader.
        """
        js_config = self.config.get("joint_sampling", {})
        js_enabled = js_config.get("enabled", False)

        if not js_enabled:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=data.collate_batch,
                **_get_dataloader_kwargs(num_workers, is_train=True),
            )

        # Get joint sampling configuration
        task_oversampling = js_config.get("task_oversampling", {})
        curriculum_config = js_config.get("curriculum", {})

        task_alpha = task_oversampling.get("alpha", 0.0)
        curriculum_enabled = curriculum_config.get("enabled", False)

        # Setup curriculum state if needed
        if curriculum_enabled and quality_labels:
            patience = curriculum_config.get("patience", 3)
            qualities = curriculum_config.get("qualities", ["high", "medium", "low"])

            self._curriculum_state = CurriculumState(
                qualities=qualities,
                patience=patience,
            )

        # Create JointSampler
        num_samples = js_config.get("num_samples", len(dataset))
        seed = js_config.get("seed", 42)
        increment_seed = js_config.get("increment_seed_per_epoch", True)

        self._joint_sampler = JointSampler(
            targets=targets,
            quality_labels=quality_labels,
            curriculum_state=self._curriculum_state,
            task_alpha=task_alpha,
            num_samples=num_samples if num_samples else len(dataset),
            seed=seed,
            increment_seed_per_epoch=increment_seed,
        )

        logger.info(
            "JointSampler enabled for Chemeleon: task_alpha=%.2f, curriculum=%s",
            task_alpha,
            curriculum_enabled,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._joint_sampler,
            num_workers=num_workers,
            collate_fn=data.collate_batch,
            **_get_dataloader_kwargs(num_workers, is_train=True),
        )

    def _setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer."""
        from lightning.pytorch.callbacks import Callback

        opt_config = self.config.get("optimization", {})

        callbacks: list[Callback] = [self._unfreeze_callback]

        # Early stopping
        if opt_config.get("patience", 0) > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=opt_config.get("patience", 15),
                    mode="min",
                )
            )

        # Model checkpoint - create persistent temp directory
        self._checkpoint_dir = tempfile.TemporaryDirectory()
        callbacks.append(
            ModelCheckpoint(
                dirpath=self._checkpoint_dir.name,
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        )

        # Add external callbacks
        callbacks.extend(self._external_callbacks)

        # Set optimal float32 matrix multiplication precision for better GPU performance
        torch.set_float32_matmul_precision("medium")

        self.trainer = pl.Trainer(
            max_epochs=opt_config.get("max_epochs", 100),
            enable_progress_bar=opt_config.get("progress_bar", False),
            callbacks=callbacks,
            logger=False,  # We use MLflow directly
            accelerator="auto",
            gradient_clip_val=1.0,  # Clip gradients for training stability
        )

    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint weights into the model.

        PyTorch Lightning's ModelCheckpoint saves checkpoints to disk but does NOT
        automatically restore the best weights to the model after training.
        This method loads the best checkpoint weights.
        """
        if self.trainer is None or self.mpnn is None:
            return

        best_checkpoint_path = self._get_best_checkpoint_path()
        if best_checkpoint_path is None:
            logger.warning("No best checkpoint found, using final model weights")
            return

        try:
            checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            self.mpnn.load_state_dict(checkpoint["state_dict"])
            logger.info("Loaded best checkpoint from %s", best_checkpoint_path)
        except Exception as e:
            logger.warning("Failed to load best checkpoint: %s", e)

    def _get_best_checkpoint_path(self) -> str | None:
        """Get path to the best checkpoint file.

        Returns
        -------
        str | None
            Path to best checkpoint, or None if not found.
        """
        if self.trainer is None:
            return None

        # Try to get from ModelCheckpoint callback
        callbacks = getattr(self.trainer, "callbacks", []) or []
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_path = callback.best_model_path
                if best_path and Path(best_path).exists():
                    return best_path

        # Fallback: search in checkpoint directory for best*.ckpt
        if self._checkpoint_dir is not None:
            checkpoint_dir = Path(self._checkpoint_dir.name)
            if checkpoint_dir.exists():
                # Look for best*.ckpt first, then any .ckpt as last resort
                ckpt_files = list(checkpoint_dir.glob("best*.ckpt"))
                if not ckpt_files:
                    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
                if ckpt_files:
                    # Sort by modification time to get the most recent
                    ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return str(ckpt_files[0])

        return None

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Generate predictions for SMILES.

        Parameters
        ----------
        smiles : list[str]
            SMILES strings to predict.

        Returns
        -------
        np.ndarray
            Predictions. Shape: (n_samples,) or (n_samples, n_tasks).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if not self._fitted or self.mpnn is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Create dataset
        datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
        dataset = data.MoleculeDataset(datapoints, featurizer=self.featurizer)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=data.collate_batch,
        )

        # Predict
        self.mpnn.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                preds = self.mpnn(batch)
                predictions.append(preds.cpu().numpy())

        preds_array = np.vstack(predictions)

        # Unscale predictions
        if self.scaler is not None:
            preds_array = self.scaler.inverse_transform(preds_array)

        # Squeeze if single task
        if preds_array.shape[1] == 1:
            preds_array = preds_array.squeeze(1)

        return preds_array

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChemeleonModel":
        """Create model from configuration.

        Parameters
        ----------
        config : DictConfig
            Configuration object.

        Returns
        -------
        ChemeleonModel
            Initialized model.
        """
        return cls(config)

    def get_trainer_callbacks(self) -> list[pl.Callback]:
        """Get PyTorch Lightning callbacks.

        Returns
        -------
        list[pl.Callback]
            List of callbacks including GradualUnfreezeCallback and CorrelationMetricsCallback.
        """
        return [self._unfreeze_callback, self._correlation_metrics_callback]
