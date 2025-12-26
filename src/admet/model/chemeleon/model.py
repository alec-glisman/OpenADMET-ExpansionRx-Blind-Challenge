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

import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from admet.model.base import BaseModel
from admet.model.chemeleon.callbacks import GradualUnfreezeCallback
from admet.model.config import ChemeleonModelParams, UnfreezeScheduleConfig
from admet.model.ffn_factory import create_ffn_predictor
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Zenodo URL for pre-trained Chemeleon weights
ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "admet" / "chemeleon"


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

        self._smiles_col = config.get("data", {}).get("smiles_col", "smiles")
        self._target_cols: list[str] = list(config.get("data", {}).get("target_cols", []))
        self._target_weights: list[float] = []

        # Checkpoint directory (created during training)
        self._checkpoint_dir: tempfile.TemporaryDirectory | None = None

        # Unfreeze callback
        unfreeze_config = self._get_unfreeze_config()
        self._unfreeze_callback = GradualUnfreezeCallback(unfreeze_config)

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
        ffn_type = self._get_model_param("ffn_type", "regression")
        self.ffn = create_ffn_predictor(
            ffn_type=ffn_type,
            input_dim=self.mp.output_dim,
            n_tasks=n_tasks,
            hidden_dim=self._get_model_param("ffn_hidden_dim", 300),
            n_layers=self._get_model_param("ffn_num_layers", 2),
            dropout=self._get_model_param("dropout", 0.0),
            n_experts=self._get_model_param("n_experts", None),
            trunk_n_layers=self._get_model_param("trunk_n_layers", None),
            trunk_hidden_dim=self._get_model_param("trunk_hidden_dim", None),
        )

        # Create MPNN
        self.mpnn = models.MPNN(
            message_passing=self.mp,
            agg=self.agg,
            predictor=self.ffn,
            batch_norm=self._get_model_param("batch_norm", False),
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

        logger.info(f"Loading Chemeleon checkpoint from {path}")
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
        """Download checkpoint from Zenodo.

        Returns
        -------
        str
            Path to downloaded checkpoint.
        """
        cache_dir = DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / "chemeleon_mp.pt"

        if not checkpoint_path.exists():
            logger.info(f"Downloading Chemeleon checkpoint from {ZENODO_URL}")
            urllib.request.urlretrieve(ZENODO_URL, checkpoint_path)
            logger.info(f"Downloaded to {checkpoint_path}")

        return str(checkpoint_path)

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

        Returns
        -------
        ChemeleonModel
            Self, for method chaining.
        """
        # Initialize MLflow if enabled
        self.init_mlflow(run_name=self.config.get("mlflow", {}).get("run_name"))

        # Determine number of tasks
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_tasks = y.shape[1]

        # Initialize model
        self._init_model(n_tasks)

        # Set target columns if not specified
        if not self._target_cols:
            self._target_cols = [f"target_{i}" for i in range(n_tasks)]

        # Create datasets
        train_dataset = self._create_dataset(smiles, y)
        val_dataset = None
        if val_smiles is not None and val_y is not None:
            if val_y.ndim == 1:
                val_y = val_y.reshape(-1, 1)
            val_dataset = self._create_dataset(val_smiles, val_y)

        # Scale targets
        self.scaler = train_dataset.normalize_targets()
        if val_dataset is not None:
            val_dataset.normalize_targets(self.scaler)

        # Create dataloaders
        opt_config = self.config.get("optimization", {})
        batch_size = opt_config.get("batch_size", 32)
        num_workers = opt_config.get("num_workers", 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data.collate_batch,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=data.collate_batch,
            )

        # Setup trainer
        self._setup_trainer()

        # Log params
        self.log_params_from_config()

        # Train
        self.trainer.fit(
            self.mpnn,
            train_loader,
            val_loader,
        )

        # Load best checkpoint weights (PyTorch Lightning does NOT auto-restore)
        self._load_best_checkpoint()

        self._fitted = True

        # Clean up checkpoint directory
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.cleanup()
            self._checkpoint_dir = None

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
            mol_data = data.MoleculeDatapoint(
                mol=data.Molecule(smi),
                y=targets.tolist() if isinstance(targets, np.ndarray) else [targets],
            )
            datapoints.append(mol_data)

        return data.MoleculeDataset(datapoints, featurizer=self.featurizer)

    def _setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer."""
        opt_config = self.config.get("optimization", {})

        callbacks = [self._unfreeze_callback]

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

        self.trainer = pl.Trainer(
            max_epochs=opt_config.get("max_epochs", 100),
            enable_progress_bar=opt_config.get("progress_bar", False),
            callbacks=callbacks,
            logger=False,  # We use MLflow directly
            accelerator="auto",
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
            logger.info(f"Loaded best checkpoint from {best_checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load best checkpoint: {e}")

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
        for callback in self.trainer.callbacks:
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
        datapoints = [data.MoleculeDatapoint(mol=data.Molecule(smi)) for smi in smiles]
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
            List of callbacks including GradualUnfreezeCallback.
        """
        return [self._unfreeze_callback]
