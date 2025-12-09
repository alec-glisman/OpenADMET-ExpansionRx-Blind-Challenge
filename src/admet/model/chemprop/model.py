"""
Chemprop model wrapper for ADMET property prediction
=====================================================

This module provides a high-level wrapper around the Chemprop library for
training and inference of molecular property prediction models using
message-passing neural networks (MPNNs).

Key components
--------------
- ``ChempropHyperparams``: Dataclass holding all model hyperparameters including
  optimization settings, message passing configuration, and FFN architecture choices.
- ``ChempropModel``: Main class that handles data preparation, model construction,
  training via PyTorch Lightning, and prediction.

Supported FFN architectures
---------------------------
- ``regression``: Standard multi-task regression FFN (default)
- ``mixture_of_experts``: Mixture of experts regression FFN
- ``branched``: Branched FFN with shared trunk and task-specific heads

Example usage
-------------
>>> model = ChempropModel(
...     df_train=train_df,
...     df_validation=val_df,
...     smiles_col="smiles",
...     target_cols=["logD", "solubility"],
... )
>>> model.fit()
>>> predictions = model.predict(test_df)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.data
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn import RegressionFFN
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from matplotlib import pyplot as plt
from mlflow import MlflowClient
from mlflow.tracking.fluent import ActiveRun
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from admet.data.smiles import parallel_canonicalize_smiles
from admet.data.stats import correlation, distribution
from admet.model.chemprop.config import (
    ChempropConfig,
    CurriculumConfig,
    DataConfig,
    MlflowConfig,
    ModelConfig,
    OptimizationConfig,
)
from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState
from admet.model.chemprop.curriculum_sampler import (
    DynamicCurriculumSampler,
    get_quality_indices,
)
from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN
from admet.model.chemprop.task_sampler import TaskAwareSampler
from admet.plot.metrics import METRIC_NAMES, compute_metrics_df, plot_metric_bar
from admet.plot.parity import plot_parity

# Module logger
logger = logging.getLogger("admet.model.chemprop.model")


# TODO: think about task weights: _tasks = [1.018, 1.000, 1.364, 1.134, 2.377, 2.373]


class CriterionName(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    RMSE = "RMSE"
    SID = "SID"
    BCE = "BCE"
    R2 = "R2Score"
    CROSS_ENTROPY = "CrossEntropy"
    DIRICHLET = "Dirichlet"
    EVIDENTIAL = "Evidential"
    MVE = "MVE"
    QUANTILE = "Quantile"

    @classmethod
    def resolve(cls, name: str, **kwargs: Any) -> Any:
        try:
            key = cls(name)
        except ValueError as exc:
            raise ValueError(f"Unsupported criterion: {name}") from exc
        return _criterion_from_enum(key, **kwargs)


def _criterion_from_enum(criterion: CriterionName, **kwargs: Any) -> Any:
    attr = getattr(nn.metrics, criterion.value, None)
    if attr is None:
        raise ValueError(f"Criterion class not found for '{criterion.value}'")
    if callable(attr):
        try:
            return attr(**kwargs)
        except TypeError as exc:
            raise TypeError(f"Unable to instantiate criterion '{criterion.value}'") from exc
    return attr


def _sanitize_mlflow_metric_name(name: str) -> str:
    """
    Sanitize a metric name for MLflow compatibility.

    MLflow only allows '_', '/', '.' and ' ' special characters in metric names.
    This function replaces disallowed characters with safe alternatives.

    Parameters
    ----------
    name : str
        The metric name to sanitize.

    Returns
    -------
    str
        The sanitized metric name.
    """
    # Replace common problematic characters
    replacements = {
        ">": "",
        "<": "",
        ":": "_",
        ";": "_",
        "|": "_",
        "\\": "_",
        "?": "",
        "*": "",
        '"': "",
        "'": "",
        "[": "_",
        "]": "_",
        "(": "_",
        ")": "_",
        "{": "_",
        "}": "_",
        "#": "_",
        "%": "pct",
        "&": "and",
        "@": "at",
        "!": "",
        "+": "plus",
        "=": "eq",
        "^": "",
        "~": "",
        "`": "",
    }
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)
    return name


class MLflowModelCheckpoint(ModelCheckpoint):
    """
    Model checkpoint callback that registers best models with MLflow.

    Extends PyTorch Lightning's ModelCheckpoint to automatically log
    the best model checkpoint to MLflow when a new best model is saved.

    Parameters
    ----------
    mlflow_client : MlflowClient
        The MLflow client instance for logging artifacts.
    run_id : str
        The MLflow run ID.
    **kwargs
        Additional arguments passed to ModelCheckpoint.

    Attributes
    ----------
    mlflow_client : MlflowClient
        The MLflow client instance.
    run_id : str
        The MLflow run ID.
    """

    def __init__(
        self,
        mlflow_client: MlflowClient,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mlflow_client = mlflow_client
        self._run_id = run_id

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Called when a checkpoint is saved.

        Logs the checkpoint as an MLflow artifact when a new best model
        is saved.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        checkpoint : Dict[str, Any]
            The checkpoint dictionary.
        """
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Log best model checkpoint to MLflow when it's updated
        if self.best_model_path and Path(self.best_model_path).exists():
            try:
                # Log the best checkpoint as an artifact
                self._mlflow_client.log_artifact(
                    self._run_id,
                    self.best_model_path,
                    artifact_path="checkpoints/best",
                )
                # Log best model score
                if self.best_model_score is not None:
                    self._mlflow_client.log_metric(
                        self._run_id,
                        "best_val_loss",
                        float(self.best_model_score),
                    )
            except Exception as e:
                logger.warning("Failed to log checkpoint to MLflow: %s", e)


@dataclass
class ChempropHyperparams:
    """
    Hyperparameters for Chemprop MPNN model.

    This dataclass encapsulates all configurable hyperparameters for training
    a Chemprop message-passing neural network, including optimization settings,
    message passing architecture, and feed-forward network configuration.

    Attributes
    ----------
    init_lr : float, default=0.0001
        Initial learning rate for the optimizer.
    max_lr : float, default=0.001
        Maximum learning rate during warmup.
    final_lr : float, default=0.0001
        Final learning rate after decay.
    warmup_epochs : int, default=3
        Number of epochs for learning rate warmup.
    patience : int, default=15
        Number of epochs without improvement before early stopping.
    max_epochs : int, default=80
        Maximum number of training epochs.
    batch_size : int, default=16
        Batch size for training.
    num_workers : int, default=0
        Number of data loading workers.
    seed : int, default=12345
        Random seed for reproducibility.
    depth : int, default=3
        Number of message passing iterations.
    message_hidden_dim : int, default=300
        Hidden dimension for message passing layers.
    num_layers : int, default=2
        Number of layers in the feed-forward network.
    hidden_dim : int, default=300
        Hidden dimension for FFN layers.
    dropout : float, default=0.0
        Dropout probability.
    criterion : str, default='MSE'
        Loss criterion. Options: 'MSE', 'SID', 'BCE', 'CrossEntropy',
        'Dirichlet', 'Evidential', 'MVE', 'Quantile'.
    ffn_type : str, default='regression'
        Type of feed-forward network. Options: 'regression',
        'mixture_of_experts', 'branched'.
    trunk_n_layers : int, default=1
        Number of trunk layers for branched FFN.
    trunk_hidden_dim : int, default=300
        Hidden dimension for trunk layers in branched FFN.
    n_experts : int, default=3
        Number of experts for mixture of experts FFN.
    batch_norm : bool, default=True
        Whether to use batch normalization in MPNN.
    """

    # Optimization
    init_lr: float = 1.0e-4
    max_lr: float = 1.0e-3
    final_lr: float = 1.0e-4
    warmup_epochs: int = 5
    patience: int = 15
    max_epochs: int = 150
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 12345

    # Message passing
    depth: int = 5
    message_hidden_dim: int = 600

    # Feed forward
    dropout: float = 0.1
    num_layers: int = 2
    hidden_dim: int = 600
    # options: "MAE", "MSE", "RMSE", "SID", "BCE", "CrossEntropy", "Dirichlet", "Evidential", "MVE", "Quantile"
    criterion: str = "MAE"
    ffn_type: str = "regression"  # options: 'regression', 'mixture_of_experts', 'branched'

    # Branched FFN
    trunk_n_layers: int = 2
    trunk_hidden_dim: int = 600

    # Mixture of Experts FFN
    n_experts: int = 4

    # MPNN
    batch_norm: bool = True
    task_sampling_alpha: Optional[float] = None


class ChempropModel:
    """
    High-level wrapper for Chemprop MPNN training and inference.

    This class handles the full workflow of preparing data, building the model,
    training with PyTorch Lightning, and generating predictions. It supports
    multiple FFN architectures and automatic target normalization.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataframe containing SMILES and target columns.
    df_validation : pandas.DataFrame or None, optional
        Validation dataframe for early stopping and model selection.
    df_test : pandas.DataFrame or None, optional
        Test dataframe (not used during training).
    smiles_col : str, default='smiles'
        Column name containing SMILES strings.
    target_cols : list[str], default=[]
        List of target column names for multi-task prediction.
    target_weights : list[float], default=[]
        Per-task weights for the loss function. If empty, all tasks
        are weighted equally.
    output_dir : pathlib.Path or None, optional
        Directory to save model checkpoints.
    progress_bar : bool, default=False
        Whether to show training progress bar.
    hyperparams : ChempropHyperparams or None, optional
        Model hyperparameters. If None, defaults are used.
    mlflow_tracking : bool, default=True
        Whether to enable MLflow tracking for params, metrics, and artifacts.
    mlflow_experiment_name : str, default='chemprop'
        MLflow experiment name to log runs under.
    mlflow_run_name : str or None, optional
        Optional run name for the MLflow run.

    Attributes
    ----------
    featurizer : SimpleMoleculeMolGraphFeaturizer
        Molecular graph featurizer.
    mpnn : models.MPNN
        The underlying Chemprop MPNN model.
    trainer : pl.Trainer
        PyTorch Lightning trainer instance.
    scaler : StandardScaler
        Target normalization scaler fitted on training data.
    metrics : list
        Evaluation metrics (RMSE, MAE, R2).
    mlflow_run_id : str or None
        The MLflow run ID if tracking is enabled.

    Examples
    --------
    >>> model = ChempropModel(
    ...     df_train=train_df,
    ...     df_validation=val_df,
    ...     smiles_col="smiles",
    ...     target_cols=["logD", "solubility"],
    ...     mlflow_tracking=True,
    ...     mlflow_experiment_name="admet_chemprop",
    ... )
    >>> model.fit()
    >>> predictions = model.predict(test_df)
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame | None = None,
        df_test: pd.DataFrame | None = None,
        smiles_col: str = "smiles",
        target_cols: List[str] = [],
        target_weights: List[float] = [],
        output_dir: Path | None = None,
        progress_bar: bool = False,
        hyperparams: ChempropHyperparams | None = None,
        mlflow_tracking: bool = True,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "chemprop",
        mlflow_run_name: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        mlflow_parent_run_id: Optional[str] = None,
        mlflow_nested: bool = False,
        curriculum_config: CurriculumConfig | None = None,
        curriculum_state: CurriculumState | None = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize ChempropModel with data and configuration.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training dataframe containing SMILES and target columns.
        df_validation : pandas.DataFrame or None, optional
            Validation dataframe for early stopping.
        df_test : pandas.DataFrame or None, optional
            Test dataframe (stored but not used during training).
        smiles_col : str, default='smiles'
            Column name containing SMILES strings.
        target_cols : list[str], default=[]
            List of target column names.
        target_weights : list[float], default=[]
            Per-task loss weights.
        output_dir : pathlib.Path or None, optional
            Directory for saving checkpoints.
        progress_bar : bool, default=False
            Whether to display progress bar during training.
        hyperparams : ChempropHyperparams or None, optional
            Model hyperparameters.
        mlflow_tracking : bool, default=True
            Whether to enable MLflow tracking.
        mlflow_tracking_uri : str or None, optional
            MLflow tracking server URI. If None, uses default.
        mlflow_experiment_name : str, default='chemprop'
            MLflow experiment name.
        mlflow_run_name : str or None, optional
            Optional MLflow run name.
        mlflow_run_id : str or None, optional
            Existing MLflow run ID to attach to. If provided, the model
            will log to this existing run instead of creating a new one.
        mlflow_parent_run_id : str or None, optional
            Parent run ID for creating nested runs. Used for ensemble training.
        mlflow_nested : bool, default=False
            Whether to create a nested run under the parent_run_id.
        curriculum_config : CurriculumConfig or None, optional
            Curriculum learning configuration. If None, curriculum learning
            is disabled.
        curriculum_state : CurriculumState or None, optional
            Shared curriculum state for ensemble training. If None and
            curriculum_config is enabled, a new state is created.
        data_dir : str or None, optional
            Path to the data directory. Used to parse and log structured
            parameters like split_type, version, quality, cluster_method,
            split_method, split, and fold.
        """
        self.smiles_col: str = smiles_col
        self.target_cols: List[str] = target_cols
        self.target_weights: List[float] = target_weights
        self.output_dir: Path | None = output_dir
        self.progress_bar: bool = progress_bar
        self.hyperparams: ChempropHyperparams = hyperparams or ChempropHyperparams()
        self.data_dir: Optional[str] = data_dir

        # Curriculum learning configuration
        self.curriculum_config: CurriculumConfig | None = curriculum_config
        self.curriculum_state: CurriculumState | None = curriculum_state

        # Initialize curriculum state if config is enabled but state not provided
        if self.curriculum_config is not None and self.curriculum_config.enabled:
            if self.curriculum_state is None:
                self.curriculum_state = CurriculumState(
                    qualities=list(self.curriculum_config.qualities),
                    patience=self.curriculum_config.patience,
                )
            # Store quality column for later use
            self.quality_col: str | None = self.curriculum_config.quality_col
        else:
            self.quality_col = None

        # MLflow configuration
        self.mlflow_tracking: bool = mlflow_tracking
        self.mlflow_tracking_uri: Optional[str] = mlflow_tracking_uri
        self.mlflow_experiment_name: str = mlflow_experiment_name
        self.mlflow_run_name: Optional[str] = mlflow_run_name
        self.mlflow_run_id: Optional[str] = mlflow_run_id  # For attaching to existing run
        self.mlflow_parent_run_id: Optional[str] = mlflow_parent_run_id  # For nested runs
        self.mlflow_nested: bool = mlflow_nested
        self._mlflow_logger: Optional[MLFlowLogger] = None
        self._mlflow_client: Optional[MlflowClient] = None
        self._mlflow_run: Optional[ActiveRun] = None  # Active run context
        self._checkpoint_temp_dir: Optional[Path] = None

        # Initialize MLflow run at init to keep active context throughout lifecycle
        if self.mlflow_tracking:
            self._init_mlflow()

        if self.target_weights == []:
            self.target_weights = [1.0] * len(self.target_cols)
        self.metrics = [
            nn.metrics.MAE(self.target_weights),
            nn.metrics.MSE(self.target_weights),
            nn.metrics.RMSE(self.target_weights),
            nn.metrics.R2Score(self.target_weights),
        ]

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.agg = nn.NormAggregation()
        self.mp = nn.BondMessagePassing(
            d_h=self.hyperparams.message_hidden_dim,
            depth=self.hyperparams.depth,
            dropout=self.hyperparams.dropout,
        )
        self.ffn: Any = None
        self.mpnn: Any = None
        self.trainer: Any = None

        self.dataframes: Dict[str, pd.DataFrame | None] = {
            "train": df_train,
            "validation": df_validation,
            "test": df_test,
            "blind": None,  # Populated via from_config or directly
        }
        self.dataloaders: Dict[str, Any] = {
            "train": None,
            "validation": None,
            "test": None,
        }
        self.scaler: Any = None
        self.transform: Any = None

        self._prepare_dataloaders()
        self._prepare_model()
        self._prepare_trainer()

    @classmethod
    def from_config(
        cls,
        config: Union[ChempropConfig, DictConfig],
        df_train: Optional[pd.DataFrame] = None,
        df_validation: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
        df_blind: Optional[pd.DataFrame] = None,
    ) -> "ChempropModel":
        """
        Create a ChempropModel from an OmegaConf configuration.

        This factory method allows creating a model from a YAML configuration
        file, enabling reproducible and configurable experiments.

        Parameters
        ----------
        config : ChempropConfig or DictConfig
            Configuration object containing data, model, optimization,
            and MLflow settings. Can be created from a YAML file using
            ``OmegaConf.load()`` and ``OmegaConf.merge()``.
        df_train : pandas.DataFrame, optional
            Pre-loaded training dataframe. If None, will be loaded from
            ``config.data.train_file``.
        df_validation : pandas.DataFrame, optional
            Pre-loaded validation dataframe. If None and
            ``config.data.validation_file`` is set, will be loaded from file.
        df_test : pandas.DataFrame, optional
            Pre-loaded test dataframe. If None and ``config.data.test_file``
            is set, will be loaded from file.
        df_blind : pandas.DataFrame, optional
            Pre-loaded blind test dataframe. Stored as an attribute but not
            used during training.

        Returns
        -------
        ChempropModel
            Initialized model ready for training.

        Examples
        --------
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.merge(
        ...     OmegaConf.structured(ChempropConfig),
        ...     OmegaConf.load("experiment.yaml")
        ... )
        >>> model = ChempropModel.from_config(config)
        >>> model.fit()
        """
        # Load dataframes if not provided
        data_dir = Path(config.data.data_dir)
        if df_train is None:
            df_train = pd.read_csv(data_dir / "train.csv", low_memory=False)

        if df_validation is None:
            df_validation = pd.read_csv(data_dir / "validation.csv", low_memory=False)

        if df_test is None and config.data.test_file is not None:
            df_test = pd.read_csv(config.data.test_file, low_memory=False)

        if df_blind is None and config.data.blind_file is not None:
            df_blind = pd.read_csv(config.data.blind_file, low_memory=False)

        # Canonicalize SMILES
        df_train[config.data.smiles_col] = parallel_canonicalize_smiles(df_train[config.data.smiles_col].tolist())
        if df_validation is not None:
            df_validation[config.data.smiles_col] = parallel_canonicalize_smiles(
                df_validation[config.data.smiles_col].tolist()
            )
        if df_test is not None:
            df_test[config.data.smiles_col] = parallel_canonicalize_smiles(df_test[config.data.smiles_col].tolist())
        if df_blind is not None:
            df_blind[config.data.smiles_col] = parallel_canonicalize_smiles(df_blind[config.data.smiles_col].tolist())

        # Build hyperparams from config
        hyperparams = ChempropHyperparams(
            # Optimization
            init_lr=config.optimization.init_lr,
            max_lr=config.optimization.max_lr,
            final_lr=config.optimization.final_lr,
            warmup_epochs=config.optimization.warmup_epochs,
            patience=config.optimization.patience,
            max_epochs=config.optimization.max_epochs,
            batch_size=config.optimization.batch_size,
            num_workers=config.optimization.num_workers,
            seed=config.optimization.seed,
            criterion=config.optimization.criterion,
            # Model architecture
            depth=config.model.depth,
            message_hidden_dim=config.model.message_hidden_dim,
            dropout=config.model.dropout,
            num_layers=config.model.num_layers,
            hidden_dim=config.model.hidden_dim,
            batch_norm=config.model.batch_norm,
            ffn_type=config.model.ffn_type,
            trunk_n_layers=config.model.trunk_n_layers,
            trunk_hidden_dim=config.model.trunk_hidden_dim,
            n_experts=config.model.n_experts,
        )

        # Determine output directory
        output_dir = Path(config.data.output_dir) if config.data.output_dir else None

        # Create model instance
        model = cls(
            df_train=df_train,
            df_validation=df_validation,
            df_test=df_test,
            smiles_col=config.data.smiles_col,
            target_cols=list(config.data.target_cols),
            target_weights=list(config.data.target_weights) if config.data.target_weights else [],
            output_dir=output_dir,
            progress_bar=config.optimization.progress_bar,
            hyperparams=hyperparams,
            mlflow_tracking=config.mlflow.tracking,
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_experiment_name=config.mlflow.experiment_name,
            mlflow_run_name=config.mlflow.run_name,
            mlflow_run_id=config.mlflow.run_id,
            mlflow_parent_run_id=config.mlflow.parent_run_id,
            mlflow_nested=config.mlflow.nested,
            curriculum_config=config.curriculum,
            data_dir=config.data.data_dir,
        )

        # Store blind dataframe for later prediction
        model.dataframes["blind"] = df_blind

        return model

    def to_config(self) -> ChempropConfig:
        """
        Export current model configuration to a ChempropConfig object.

        Returns
        -------
        ChempropConfig
            Configuration object that can be saved to YAML using OmegaConf.

        Examples
        --------
        >>> config = model.to_config()
        >>> OmegaConf.save(config, "experiment_config.yaml")
        """
        return ChempropConfig(
            data=DataConfig(
                data_dir="",  # Not available after loading
                smiles_col=self.smiles_col,
                target_cols=list(self.target_cols),
                target_weights=list(self.target_weights),
                output_dir=str(self.output_dir) if self.output_dir else None,
            ),
            model=ModelConfig(
                depth=self.hyperparams.depth,
                message_hidden_dim=self.hyperparams.message_hidden_dim,
                dropout=self.hyperparams.dropout,
                num_layers=self.hyperparams.num_layers,
                hidden_dim=self.hyperparams.hidden_dim,
                batch_norm=self.hyperparams.batch_norm,
                ffn_type=self.hyperparams.ffn_type,
                trunk_n_layers=self.hyperparams.trunk_n_layers,
                trunk_hidden_dim=self.hyperparams.trunk_hidden_dim,
                n_experts=self.hyperparams.n_experts,
            ),
            optimization=OptimizationConfig(
                criterion=self.hyperparams.criterion,
                init_lr=self.hyperparams.init_lr,
                max_lr=self.hyperparams.max_lr,
                final_lr=self.hyperparams.final_lr,
                warmup_epochs=self.hyperparams.warmup_epochs,
                patience=self.hyperparams.patience,
                max_epochs=self.hyperparams.max_epochs,
                batch_size=self.hyperparams.batch_size,
                num_workers=self.hyperparams.num_workers,
                seed=self.hyperparams.seed,
                progress_bar=self.progress_bar,
            ),
            mlflow=MlflowConfig(
                tracking=self.mlflow_tracking,
                tracking_uri=self.mlflow_tracking_uri,
                experiment_name=self.mlflow_experiment_name,
                run_name=self.mlflow_run_name,
            ),
            curriculum=self.curriculum_config or CurriculumConfig(),
        )

    def _prepare_dataloaders(self) -> None:
        """
        Prepare PyTorch dataloaders for train/validation/test splits.

        This method converts dataframes to Chemprop MoleculeDatasets,
        fits a target scaler on training data, and creates dataloaders
        for each split.

        If curriculum learning is enabled, the training dataloader uses a
        WeightedRandomSampler based on the curriculum phase instead of
        uniform shuffling.
        """
        datapoints = {}
        datasets = {}

        # Store quality labels for curriculum sampling
        self._quality_labels: Dict[str, List[str] | None] = {
            "train": None,
            "validation": None,
            "test": None,
        }

        for split in ["train", "validation", "test"]:
            df = self.dataframes[split]
            if df is None:
                continue
            smis = df.loc[:, self.smiles_col].values
            ys = df.loc[:, self.target_cols].values

            # Canonicalize SMILES
            smis = parallel_canonicalize_smiles(smis.tolist())

            # Extract quality labels if curriculum is enabled
            if self.quality_col is not None and self.quality_col in df.columns:
                self._quality_labels[split] = df[self.quality_col].tolist()

            datapoints[split] = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
            datasets[split] = data.MoleculeDataset(datapoints[split], self.featurizer)

            if split == "train":
                self.scaler = datasets[split].normalize_targets()
                self.transform = nn.transforms.UnscaleTransform.from_standard_scaler(self.scaler)

            elif split == "validation":
                datasets[split].normalize_targets(self.scaler)

            # Build dataloader with curriculum sampling for training if enabled
            if split == "train" and self.curriculum_state is not None and self._quality_labels["train"] is not None:
                # Use dynamic curriculum-aware sampling that updates with phase changes
                seed = (
                    self.curriculum_config.seed
                    if self.curriculum_config and self.curriculum_config.seed
                    else self.hyperparams.seed
                )
                sampler = DynamicCurriculumSampler(
                    quality_labels=self._quality_labels["train"],
                    curriculum_state=self.curriculum_state,
                    num_samples=len(datasets[split]),
                    seed=seed,
                )
                drop_last = (len(datasets[split]) % self.hyperparams.batch_size) == 1
                self.dataloaders[split] = DataLoader(
                    datasets[split],
                    batch_size=self.hyperparams.batch_size,
                    sampler=sampler,
                    num_workers=self.hyperparams.num_workers,
                    collate_fn=data.collate_batch,
                    drop_last=drop_last,
                )
                logger.info(
                    "Dynamic curriculum sampling enabled for training: phase=%s, qualities=%s",
                    self.curriculum_state.phase,
                    self.curriculum_state.qualities,
                )
            elif split == "train" and self.hyperparams.task_sampling_alpha is not None:
                # Task-aware sampling for imbalanced multi-task learning
                logger.info("Using TaskAwareSampler with alpha=%s", self.hyperparams.task_sampling_alpha)
                sampler = TaskAwareSampler(
                    targets=ys,
                    alpha=self.hyperparams.task_sampling_alpha,
                    seed=self.hyperparams.seed,
                )
                drop_last = (len(datasets[split]) % self.hyperparams.batch_size) == 1
                self.dataloaders[split] = DataLoader(
                    datasets[split],
                    batch_size=self.hyperparams.batch_size,
                    sampler=sampler,
                    num_workers=self.hyperparams.num_workers,
                    collate_fn=data.collate_batch,
                    drop_last=drop_last,
                )
            else:
                # Standard dataloader (shuffle for train, no shuffle for val/test)
                self.dataloaders[split] = data.build_dataloader(
                    datasets[split],
                    batch_size=self.hyperparams.batch_size,
                    num_workers=self.hyperparams.num_workers,
                    shuffle=(split == "train"),
                    seed=self.hyperparams.seed,
                )

    def _prepare_model(self) -> None:
        """
        Build the MPNN model based on hyperparameters.

        Constructs the feed-forward network (FFN) based on ``ffn_type``
        hyperparameter and assembles the full MPNN with message passing,
        aggregation, and prediction components.

        Raises
        ------
        ValueError
            If ``ffn_type`` is not one of 'regression', 'mixture_of_experts',
            or 'branched'.
        """
        task_weights = torch.tensor(self.target_weights) if self.target_weights else None
        criterion = CriterionName.resolve(self.hyperparams.criterion, task_weights=task_weights)

        if self.hyperparams.ffn_type == "mixture_of_experts":
            self.ffn = MixtureOfExpertsRegressionFFN(
                n_tasks=len(self.target_cols),
                n_experts=self.hyperparams.n_experts,
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                n_layers=self.hyperparams.num_layers,
                dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=task_weights,
                output_transform=self.transform,
            )
        elif self.hyperparams.ffn_type == "branched":
            self.ffn = BranchedFFN(
                task_groups=[[i] for i in range(len(self.target_cols))],
                n_tasks=len(self.target_cols),
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                trunk_n_layers=self.hyperparams.trunk_n_layers,
                trunk_hidden_dim=self.hyperparams.trunk_hidden_dim,
                trunk_dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=task_weights,
                output_transform=self.transform,
            )
        elif self.hyperparams.ffn_type == "regression":
            self.ffn = RegressionFFN(
                n_tasks=len(self.target_cols),
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                n_layers=self.hyperparams.num_layers,
                dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=self.target_weights,
                output_transform=self.transform,
            )
        else:
            raise ValueError(f"Unsupported ffn_type: {self.hyperparams.ffn_type}")

        self.mpnn = models.MPNN(
            message_passing=self.mp,
            agg=self.agg,
            predictor=self.ffn,
            batch_norm=self.hyperparams.batch_norm,
            metrics=self.metrics,
            warmup_epochs=self.hyperparams.warmup_epochs,
            init_lr=self.hyperparams.init_lr,
            max_lr=self.hyperparams.max_lr,
            final_lr=self.hyperparams.final_lr,
        )

    def _init_mlflow(self) -> None:
        """
        Initialize MLflow tracking with an active run context.

        Creates the MLflow client and starts an active run that persists
        throughout the model's lifecycle. This allows all logging operations
        to use the same run context without repeatedly opening/closing runs.

        Supports three modes:
        1. Attach to existing run (mlflow_run_id is set)
        2. Create nested run under parent (mlflow_nested=True, mlflow_parent_run_id is set)
        3. Create new standalone run (default)
        """
        if self.mlflow_tracking_uri is not None:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        # Enable system metrics logging (CPU, memory, GPU, etc.)
        mlflow.enable_system_metrics_logging()

        # Determine which mode to use
        if self.mlflow_run_id is not None:
            # Mode 1: Attach to existing run (for ensemble nested runs)
            self._mlflow_run = mlflow.start_run(run_id=self.mlflow_run_id)
            logger.debug("MLflow attached to existing run: run_id=%s", self.mlflow_run_id)
        elif self.mlflow_nested and self.mlflow_parent_run_id is not None:
            # Mode 2: Create nested run under parent
            self._mlflow_run = mlflow.start_run(
                run_name=self.mlflow_run_name,
                nested=True,
                parent_run_id=self.mlflow_parent_run_id,
            )
            self.mlflow_run_id = self._mlflow_run.info.run_id
            logger.debug(
                "MLflow nested run started: run_id=%s, parent_run_id=%s",
                self.mlflow_run_id,
                self.mlflow_parent_run_id,
            )
        else:
            # Mode 3: Start a new standalone run
            self._mlflow_run = mlflow.start_run(run_name=self.mlflow_run_name)
            self.mlflow_run_id = self._mlflow_run.info.run_id
            logger.debug("MLflow new run started: run_id=%s", self.mlflow_run_id)

        # Create shared MlflowClient for direct API calls
        self._mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)

    def close(self) -> None:
        """
        Close the MLflow run and clean up resources.

        This should be called when you're done with the model to properly
        end the MLflow run. If not called explicitly, the run will remain
        active until the process exits.
        """
        # Clean up temp checkpoint directory if it was created
        if self._checkpoint_temp_dir is not None and self._checkpoint_temp_dir.exists():
            import shutil

            shutil.rmtree(self._checkpoint_temp_dir, ignore_errors=True)
            self._checkpoint_temp_dir = None

        # End the MLflow run
        if self._mlflow_run is not None:
            mlflow.end_run()
            logger.debug("MLflow run ended: run_id=%s", self.mlflow_run_id)
            self._mlflow_run = None

    def __del__(self) -> None:
        """Destructor to ensure MLflow run is closed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def _prepare_trainer(self) -> None:
        """
        Configure PyTorch Lightning trainer with callbacks.

        Sets up model checkpointing (saves best and last checkpoints)
        and early stopping based on validation loss. When MLflow tracking
        is enabled, uses MLflowModelCheckpoint to register models.
        """
        torch.set_float32_matmul_precision("medium")

        earlystopping = EarlyStopping(
            monitor="val_loss",
            patience=self.hyperparams.patience,
            mode="min",
        )

        # Configure logger and checkpointing based on mlflow_tracking setting
        pl_logger: MLFlowLogger | bool
        callbacks_list: List[Any] = [earlystopping]

        if self.mlflow_tracking and self.mlflow_run_id is not None:
            # Use existing MLflow run started in _init_mlflow()
            # MLFlowLogger with run_id will attach to the existing run
            self._mlflow_logger = MLFlowLogger(
                experiment_name=self.mlflow_experiment_name,
                run_id=self.mlflow_run_id,  # Attach to existing run
                tracking_uri=self.mlflow_tracking_uri,
                log_model=False,  # We handle model logging manually
                save_dir=None,  # Don't create local artifact directory
            )
            pl_logger = self._mlflow_logger

            # Use MLflow-enabled checkpointing with shared client
            # Use output_dir if provided, otherwise use a temp directory for checkpoints
            checkpoint_dir = self.output_dir
            if checkpoint_dir is None:
                # Create a temp directory that will be cleaned up
                self._checkpoint_temp_dir = Path(tempfile.mkdtemp(prefix="chemprop_ckpt_"))
                checkpoint_dir = self._checkpoint_temp_dir
            else:
                self._checkpoint_temp_dir = None

            checkpointing = MLflowModelCheckpoint(
                mlflow_client=self._mlflow_client,  # type: ignore[arg-type]
                run_id=self.mlflow_run_id,
                dirpath=checkpoint_dir,
                filename="best-{epoch:04}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
            )
            callbacks_list.append(checkpointing)

            # NOTE: EpochMetricsCallback removed - calling trainer.predict() inside
            # callbacks corrupts Lightning's internal state. Detailed metrics are
            # computed at training end via _log_evaluation_metrics() instead.
        else:
            pl_logger = True
            # Standard checkpointing without MLflow
            checkpointing = ModelCheckpoint(  # type: ignore[no-redef,assignment]
                dirpath=self.output_dir,
                filename="best-{epoch:04}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
            )
            callbacks_list.append(checkpointing)

        # Add curriculum callback if curriculum learning is enabled
        if self.curriculum_state is not None:
            # Get configuration options for curriculum callback
            reset_es = self.curriculum_config.reset_early_stopping_on_phase_change if self.curriculum_config else False
            log_per_quality = self.curriculum_config.log_per_quality_metrics if self.curriculum_config else True
            val_quality_labels = self._quality_labels.get("validation") if hasattr(self, "_quality_labels") else None

            curriculum_callback = CurriculumCallback(
                curr_state=self.curriculum_state,
                monitor_metric="val_loss",  # Use overall validation loss
                reset_early_stopping_on_phase_change=reset_es,
                log_per_quality_metrics=log_per_quality,
                quality_labels=val_quality_labels,
            )
            callbacks_list.append(curriculum_callback)
            logger.info(
                "Curriculum callback added: monitoring 'val_loss', " "reset_early_stopping=%s, log_per_quality=%s",
                reset_es,
                log_per_quality,
            )

        self.trainer = pl.Trainer(
            logger=pl_logger,
            enable_checkpointing=True,
            enable_progress_bar=self.progress_bar,
            accelerator="auto",
            devices=1,
            max_epochs=self.hyperparams.max_epochs,
            callbacks=callbacks_list,
        )

    def fit(self) -> bool:
        """
        Train the MPNN model.

        Runs training using the PyTorch Lightning trainer with the
        configured train and validation dataloaders. Handles keyboard
        interrupts gracefully, allowing the model to be used for
        prediction even if training is interrupted.

        When MLflow tracking is enabled, logs hyperparameters at the
        start of training, and logs metrics, artifacts, and registers
        the model upon completion.

        Returns
        -------
        bool
            True if training completed normally, False if interrupted.
        """
        # Log hyperparameters and dataset info at start of training
        if self.mlflow_tracking and self._mlflow_logger is not None:
            logger.debug(
                "MLflow tracking enabled: client=%s, run_id=%s",
                self._mlflow_client is not None,
                self.mlflow_run_id,
            )
            self._log_hyperparams()
            self._log_dataset_info()
        else:
            logger.debug("MLflow tracking skipped: tracking=%s, logger=%s", self.mlflow_tracking, self._mlflow_logger)

        completed = False
        try:
            logger.info("Starting training...")
            self.trainer.fit(
                self.mpnn,
                train_dataloaders=self.dataloaders["train"],
                val_dataloaders=self.dataloaders["validation"],
            )
            logger.info("trainer.fit() returned successfully")
            completed = True
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Model state preserved.")
            logger.info("You can still use model.predict() with the current weights.")
            completed = False
        except Exception as e:
            logger.error("Training failed with error: %s", e)
            completed = False
            # Re-raise exception after logging artifacts
            raise e
        finally:
            logger.info("Training finished: completed=%s", completed)

            # Log evaluation metrics and final artifacts
            if self.mlflow_tracking and self._mlflow_logger is not None:
                logger.debug("Post-training: logging evaluation metrics and artifacts")
                # Only log metrics if training completed or we have a best model
                try:
                    self._log_evaluation_metrics()
                    self._log_training_artifacts(completed)
                except Exception as log_err:
                    logger.error("Failed to log artifacts after training: %s", log_err)
            else:
                logger.debug("Post-training skipped: tracking=%s, logger=%s", self.mlflow_tracking, self._mlflow_logger)

            # Generate evaluation plots for train and validation sets
            try:
                self._generate_training_plots()
            except Exception as plot_err:
                logger.warning("Failed to generate training plots: %s", plot_err)

        return completed

    def _generate_training_plots(self) -> None:
        """
        Generate evaluation plots for training and validation sets.

        Creates parity plots comparing true vs predicted values for each
        target column on both training and validation datasets. Plots are
        logged to MLflow if active, saved to output_dir if specified.
        """
        # Check if we should generate plots
        should_save = self.mlflow_tracking or self.output_dir is not None
        if not should_save:
            return

        # Generate plots for training set
        df_train = self.dataframes.get("train")
        if df_train is not None:
            try:
                train_preds = self.predict(df_train)
                self._generate_evaluation_plots(df_train, train_preds, split="train")
            except Exception as e:
                logger.warning("Failed to generate training plots: %s", e)

        # Generate plots for validation set
        df_validation = self.dataframes.get("validation")
        if df_validation is not None:
            try:
                val_preds = self.predict(df_validation)
                self._generate_evaluation_plots(df_validation, val_preds, split="validation")
            except Exception as e:
                logger.warning("Failed to generate validation plots: %s", e)

    def _log_hyperparams(self) -> None:
        """
        Log model hyperparameters to MLflow.

        Logs all hyperparameters from the ChempropHyperparams dataclass,
        as well as additional model configuration parameters like
        smiles_col, target_cols, and parsed data_dir parameters.
        """
        if self._mlflow_logger is None:
            return

        # Log hyperparams from dataclass
        params = asdict(self.hyperparams)

        # Add additional configuration
        params["smiles_col"] = self.smiles_col
        params["target_cols"] = str(self.target_cols)
        params["n_targets"] = len(self.target_cols)

        # Parse and log data_dir parameters if available
        if self.data_dir is not None:
            from admet.util.utils import parse_data_dir_params

            data_params = parse_data_dir_params(self.data_dir)
            for key, value in data_params.items():
                if value is not None:
                    params[f"data.{key}"] = value

        self._mlflow_logger.log_hyperparams(params)

    def _log_dataset_info(self) -> None:
        """
        Log dataset information to MLflow.

        Logs dataset sizes as parameters and saves dataset statistics
        as CSV artifacts for reproducibility.
        """
        if self._mlflow_client is None or self.mlflow_run_id is None:
            logger.debug("_log_dataset_info skipped: mlflow_client or run_id is None")
            return

        logger.debug("Logging dataset info to MLflow run %s", self.mlflow_run_id)

        # Log dataset sizes
        df_train = self.dataframes.get("train")
        df_validation = self.dataframes.get("validation")
        df_test = self.dataframes.get("test")

        if df_train is not None:
            self._mlflow_client.log_param(self.mlflow_run_id, "train_size", len(df_train))
        if df_validation is not None:
            self._mlflow_client.log_param(self.mlflow_run_id, "validation_size", len(df_validation))
        if df_test is not None:
            self._mlflow_client.log_param(self.mlflow_run_id, "test_size", len(df_test))

        # Register datasets with MLflow (run is already active from _init_mlflow)
        if df_train is not None:
            train_dataset = mlflow.data.from_pandas(df_train, name="train")  # type: ignore[attr-defined]
            mlflow.log_input(train_dataset, context="training")
            logger.debug("Registered training dataset with MLflow")
        if df_validation is not None:
            val_dataset = mlflow.data.from_pandas(df_validation, name="validation")  # type: ignore[attr-defined]
            mlflow.log_input(val_dataset, context="validation")
            logger.debug("Registered validation dataset with MLflow")

        # Log dataset statistics as CSV artifact
        if df_train is not None:
            stats_rows = []
            for target in self.target_cols:
                if target in df_train.columns:
                    train_mean = df_train[target].mean()
                    train_std = df_train[target].std()
                    stats_rows.append({"split": "train", "target": target, "metric": "mean", "value": train_mean})
                    stats_rows.append({"split": "train", "target": target, "metric": "std", "value": train_std})
            if stats_rows:
                df_stats = pd.DataFrame(stats_rows)
                temp_dir = Path(tempfile.mkdtemp())
                csv_path = temp_dir / "train_dataset_stats.csv"
                df_stats.to_csv(csv_path, index=False)
                logger.debug("Logging dataset stats artifact: %s", csv_path)
                self._mlflow_client.log_artifact(self.mlflow_run_id, str(csv_path), artifact_path="metrics")
                csv_path.unlink(missing_ok=True)
                temp_dir.rmdir()

    def _log_evaluation_metrics(self) -> None:
        """
        Compute and log correlation metrics on validation set.

        Generates predictions on the validation set and computes
        correlation metrics (MAE, RMSE, R2, Pearson, Spearman, Kendall)
        for each target column using the stats.correlation function.

        When curriculum learning is enabled, also computes per-quality
        metrics for each quality level present in the validation data.
        """
        if self._mlflow_client is None or self.mlflow_run_id is None:
            return

        df_validation = self.dataframes.get("validation")
        if df_validation is None:
            return

        # Generate predictions on validation set
        try:
            preds_df = self.predict(df_validation)
        except Exception as e:
            logger.warning("Failed to generate validation predictions: %s", e)
            return

        # Compute correlation metrics for each target (overall)
        df_out = pd.DataFrame()
        for target in self.target_cols:
            if target not in df_validation.columns:
                continue

            y_true = np.asarray(df_validation[target].values)
            y_pred = np.asarray(preds_df[target].values)

            # Compute correlation metrics using stats module
            metrics = correlation(y_true, y_pred)

            # Add metrics to output dataframe
            for metric_name, metric_value in metrics.items():
                row = pd.DataFrame(
                    {
                        "split": ["validation"],
                        "target": [target],
                        "quality": ["overall"],
                        "metric": [metric_name],
                        "value": [metric_value],
                    }
                )
                df_out = pd.concat([df_out, row], ignore_index=True)

        # Compute per-quality metrics if curriculum learning is enabled
        if (
            self.quality_col is not None
            and self.quality_col in df_validation.columns
            and self.curriculum_state is not None
        ):
            quality_labels = df_validation[self.quality_col].tolist()
            quality_indices = get_quality_indices(quality_labels, self.curriculum_state.qualities)

            for quality, indices in quality_indices.items():
                if not indices:
                    continue

                # Subset the validation data for this quality
                df_quality = df_validation.iloc[indices]
                preds_quality = preds_df.iloc[indices]

                for target in self.target_cols:
                    if target not in df_quality.columns:
                        continue

                    y_true = np.asarray(df_quality[target].values)
                    y_pred = np.asarray(preds_quality[target].values)

                    # Skip if not enough samples for meaningful metrics
                    if len(y_true) < 2:
                        continue

                    # Compute correlation metrics
                    metrics = correlation(y_true, y_pred)

                    # Add metrics to output dataframe
                    for metric_name, metric_value in metrics.items():
                        row = pd.DataFrame(
                            {
                                "split": ["validation"],
                                "target": [target],
                                "quality": [quality],
                                "metric": [metric_name],
                                "value": [metric_value],
                            }
                        )
                        df_out = pd.concat([df_out, row], ignore_index=True)

                        # Also log quality-specific metrics directly to MLflow
                        mlflow_metric_name = _sanitize_mlflow_metric_name(f"val_{quality}_{target}_{metric_name}")
                        try:
                            self._mlflow_client.log_metric(
                                self.mlflow_run_id, mlflow_metric_name, float(metric_value)  # type: ignore[arg-type]
                            )
                        except Exception:
                            pass  # Silently ignore metric logging failures

            # Log quality distribution in validation set
            quality_counts = {q: len(indices) for q, indices in quality_indices.items() if indices}
            logger.info("Validation quality distribution: %s", quality_counts)

        # Log metrics as CSV artifact
        if not df_out.empty:
            df_out.reset_index(drop=True, inplace=True)
            temp_dir = Path(tempfile.mkdtemp())
            csv_path = temp_dir / "validation_metrics.csv"
            df_out.to_csv(csv_path, index=False)
            self._mlflow_client.log_artifact(self.mlflow_run_id, str(csv_path), artifact_path="metrics")
            csv_path.unlink(missing_ok=True)
            temp_dir.rmdir()

    def _log_training_artifacts(self, completed: bool) -> None:
        """
        Log training artifacts to MLflow.

        Saves model checkpoint and training configuration.

        Parameters
        ----------
        completed : bool
            Whether training completed normally (True) or was
            interrupted (False).
        """
        if self._mlflow_client is None or self.mlflow_run_id is None:
            logger.debug("_log_training_artifacts skipped: mlflow_client or run_id is None")
            return

        logger.debug("Logging training artifacts to MLflow run %s", self.mlflow_run_id)

        # Log training completion status
        self._mlflow_client.log_param(self.mlflow_run_id, "training_completed", completed)

        # Determine checkpoint directory (could be output_dir or temp dir)
        checkpoint_dir = self.output_dir if self.output_dir is not None else self._checkpoint_temp_dir

        # Log model checkpoints as artifacts
        if checkpoint_dir is not None and checkpoint_dir.exists():
            # Debug: List contents of checkpoint directory
            all_files = list(checkpoint_dir.iterdir())
            logger.debug("Checkpoint dir (%s): %s", checkpoint_dir, [f.name for f in all_files])

            # Log best checkpoint(s)
            best_checkpoints = list(checkpoint_dir.glob("best-*.ckpt"))
            logger.debug("Found %d best checkpoints", len(best_checkpoints))

            # Sort by modification time to get the most recent best checkpoint if multiple exist
            best_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if best_checkpoints:
                # Only log the single best checkpoint
                ckpt = best_checkpoints[0]
                logger.debug("Logging best checkpoint: %s", ckpt.name)
                self._mlflow_client.log_artifact(self.mlflow_run_id, str(ckpt), artifact_path="checkpoints")

                # Log best checkpoint path as param
                self._mlflow_client.log_param(self.mlflow_run_id, "best_checkpoint_path", str(ckpt))

            # Note: We do not log 'last.ckpt' to save space and only keep the best model.

        # Save hyperparameters as YAML artifact
        import yaml

        temp_dir = Path(tempfile.mkdtemp())
        yaml_path = temp_dir / "hyperparameters.yaml"
        params = asdict(self.hyperparams)
        params["smiles_col"] = self.smiles_col
        params["target_cols"] = self.target_cols
        with open(yaml_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        self._mlflow_client.log_artifact(self.mlflow_run_id, str(yaml_path), artifact_path="config")
        yaml_path.unlink(missing_ok=True)
        temp_dir.rmdir()

        # Register the model with MLflow (run is already active from _init_mlflow)
        if completed and self.mpnn is not None:
            mlflow.pytorch.log_model(
                self.mpnn,
                artifact_path="model",
                registered_model_name=None,  # Don't auto-register to model registry
            )
            logger.debug("Registered PyTorch model with MLflow")

    def predict(
        self,
        df: pd.DataFrame,
        generate_plots: bool = False,
        split_name: str = "test",
    ) -> pd.DataFrame:
        """
        Generate predictions for a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing SMILES column. Target columns
            are optional (used if present for dataset creation).
        generate_plots : bool, default=False
            Whether to generate parity plots for predictions when
            ground truth is available.
        split_name : str, default='test'
            Split name for labeling plots when generate_plots is True.

        Returns
        -------
        pandas.DataFrame
            Predictions with columns named after target columns.
        """
        smis = df.loc[:, self.smiles_col].values
        smis = parallel_canonicalize_smiles(smis.tolist())

        # gracefully handle labelled/unlabelled data
        try:
            ys = df.loc[:, self.target_cols].values
            datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
            logger.info("Split '%s': Generating predictions for %d labelled molecules", split_name, len(datapoints))
        except KeyError:
            ys = None
            datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
            logger.info("Split '%s': Generating predictions for %d unlabelled molecules", split_name, len(datapoints))

        dataset = data.MoleculeDataset(datapoints, self.featurizer)
        dataloader = data.build_dataloader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        results = self.trainer.predict(self.mpnn, dataloaders=dataloader)

        # Collect predictions from batches
        # results is a list of prediction tensors from each batch
        all_preds = []
        for batch_preds in results:
            # batch_preds is a tensor of shape [batch_size, n_tasks]
            if hasattr(batch_preds, "cpu"):
                preds = batch_preds.cpu().numpy()
            elif isinstance(batch_preds, dict) and "preds" in batch_preds:
                preds = batch_preds["preds"].cpu().numpy()
            else:
                preds = batch_preds
            all_preds.append(preds)

        all_preds_arr = np.vstack(all_preds)
        pred_df = pd.DataFrame(all_preds_arr, columns=[f"{t}" for t in self.target_cols])

        # Log prediction metrics if ground truth is available and MLflow tracking is enabled
        if self.mlflow_tracking and self._mlflow_logger is not None:
            self._log_prediction_metrics(df, pred_df, split=split_name)

        # Generate plots if requested and ground truth is available
        if generate_plots and ys is not None:
            self._generate_evaluation_plots(df, pred_df, split=split_name)

        # Log predictions CSV and submissions CSV with transformed values
        self._log_prediction_artifacts(df, pred_df, split=split_name)

        return pred_df

    def _log_prediction_artifacts(
        self,
        df: pd.DataFrame,
        pred_df: pd.DataFrame,
        split: str = "test",
    ) -> None:
        """
        Log prediction artifacts including submissions CSV and distribution stats.

        Creates and logs:
        1. Raw predictions CSV
        2. Submissions CSV with Log columns transformed by 10^x
        3. Distribution statistics CSV for both raw and transformed predictions

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing SMILES column.
        pred_df : pd.DataFrame
            Predictions dataframe.
        split : str, default='test'
            Split name for labeling artifacts.
        """
        # Determine where to save artifacts
        should_save = self.mlflow_tracking or self.output_dir is not None
        if not should_save:
            return

        # Create temp directory for artifacts if using MLflow
        temp_dir: Optional[Path] = None
        if self.mlflow_tracking and self._mlflow_client is not None:
            temp_dir = Path(tempfile.mkdtemp())
            artifact_dir = temp_dir
        elif self.output_dir is not None:
            artifact_dir = self.output_dir / "predictions"
            artifact_dir.mkdir(parents=True, exist_ok=True)
        else:
            return

        # Create predictions dataframe with SMILES
        predictions_csv = pd.DataFrame()
        if self.smiles_col in df.columns:
            predictions_csv[self.smiles_col] = df[self.smiles_col].values

        # Add raw predictions
        for target in self.target_cols:
            if target in pred_df.columns:
                predictions_csv[target] = pred_df[target].values

        # Save raw predictions
        raw_path = artifact_dir / f"{split}_predictions.csv"
        predictions_csv.to_csv(raw_path, index=False)
        logger.debug("Saved raw predictions to %s", raw_path)

        # Create submissions CSV with transformed values (10^x for Log columns)
        submissions_csv = pd.DataFrame()
        if self.smiles_col in df.columns:
            submissions_csv[self.smiles_col] = df[self.smiles_col].values

        for target in self.target_cols:
            if target in pred_df.columns:
                values = np.asarray(pred_df[target].values)
                if target.startswith("Log "):
                    # Transform Log columns: 10^x
                    transformed_name = target.replace("Log ", "")
                    submissions_csv[transformed_name] = np.power(10.0, values)
                else:
                    submissions_csv[target] = values

        submissions_path = artifact_dir / f"{split}_submissions.csv"
        submissions_csv.to_csv(submissions_path, index=False)
        logger.debug("Saved submissions to %s", submissions_path)

        # Compute and save distribution statistics
        stats_rows = []
        stats_path: Optional[Path] = None
        for target in self.target_cols:
            if target not in pred_df.columns:
                continue

            raw_values = np.asarray(pred_df[target].values)
            raw_stats = distribution(raw_values)

            # Add raw stats
            for stat_name, stat_value in raw_stats.items():
                stats_rows.append(
                    {
                        "split": split,
                        "target": target,
                        "transform": "raw",
                        "statistic": stat_name,
                        "value": stat_value,
                    }
                )

            # Add transformed stats for Log columns
            if target.startswith("Log "):
                transformed_name = target.replace("Log ", "")
                transformed_values = np.power(10.0, raw_values)
                transformed_stats = distribution(transformed_values)

                for stat_name, stat_value in transformed_stats.items():
                    stats_rows.append(
                        {
                            "split": split,
                            "target": transformed_name,
                            "transform": "10^x",
                            "statistic": stat_name,
                            "value": stat_value,
                        }
                    )

        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            stats_path = artifact_dir / f"{split}_distribution_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            logger.debug("Saved distribution stats to %s", stats_path)

        # Log artifacts to MLflow
        if self._mlflow_client is not None and self.mlflow_run_id is not None and temp_dir is not None:
            self._mlflow_client.log_artifact(self.mlflow_run_id, str(raw_path), artifact_path="predictions")
            self._mlflow_client.log_artifact(self.mlflow_run_id, str(submissions_path), artifact_path="predictions")
            if stats_rows:
                self._mlflow_client.log_artifact(self.mlflow_run_id, str(stats_path), artifact_path="predictions")

            # Clean up temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _log_prediction_metrics(self, df: pd.DataFrame, pred_df: pd.DataFrame, split: str = "predict") -> None:
        """
        Log correlation metrics for predictions if ground truth is available.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe that may contain ground truth target columns.
        pred_df : pd.DataFrame
            Predictions dataframe.
        split : str, default='predict'
            Split name prefix for metric logging.
        """
        if self._mlflow_client is None or self.mlflow_run_id is None:
            return

        # Check if any target columns have ground truth
        has_targets = any(t in df.columns for t in self.target_cols)
        if not has_targets:
            return

        df_out = pd.DataFrame()
        all_metrics: dict[str, list[float]] = {}  # Collect metrics for averaging

        for target in self.target_cols:
            if target not in df.columns:
                continue

            y_true = np.asarray(df[target].values)
            y_pred = np.asarray(pred_df[target].values)

            # Compute correlation metrics using stats module
            metrics = correlation(y_true, y_pred)

            # Log each metric to MLflow directly with test/ prefix
            for metric_name, metric_value in metrics.items():
                # Create sanitized metric key: test/<target>_<metric>
                safe_target = _sanitize_mlflow_metric_name(target)
                mlflow_key = f"{split}/{safe_target}_{metric_name}"
                try:
                    self._mlflow_client.log_metric(
                        self.mlflow_run_id, mlflow_key, float(metric_value)  # type: ignore[arg-type]
                    )
                except Exception:
                    pass  # Silently ignore metric logging failures

                # Collect for averaging
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                metric_val_float = float(metric_value)  # type: ignore[arg-type]
                if not np.isnan(metric_val_float):
                    all_metrics[metric_name].append(metric_val_float)

                # Save to dataframe for artifact
                row = pd.DataFrame(
                    {"split": [split], "target": [target], "metric": [metric_name], "value": [metric_value]}
                )
                df_out = pd.concat([df_out, row], ignore_index=True)

        # Log mean metrics across all targets
        for metric_name, values in all_metrics.items():
            if values:
                mean_value = float(np.mean(values))
                mlflow_key = f"{split}/mean_{metric_name}"
                try:
                    self._mlflow_client.log_metric(self.mlflow_run_id, mlflow_key, mean_value)
                except Exception:
                    pass

        if not df_out.empty:
            df_out.reset_index(drop=True, inplace=True)
            temp_dir = Path(tempfile.mkdtemp())
            csv_path = temp_dir / f"{split}_prediction_metrics.csv"
            df_out.to_csv(csv_path, index=False)
            self._mlflow_client.log_artifact(self.mlflow_run_id, str(csv_path), artifact_path="metrics")
            csv_path.unlink(missing_ok=True)
            temp_dir.rmdir()

    def _generate_evaluation_plots(
        self,
        df_true: pd.DataFrame,
        pred_df: pd.DataFrame,
        split: str = "validation",
    ) -> None:
        """
        Generate parity plots for model evaluation.

        Creates parity plots comparing true vs predicted values for each
        target column. Plots are logged to MLflow if active, saved to
        output_dir if specified, or skipped otherwise.

        Parameters
        ----------
        df_true : pd.DataFrame
            Ground truth dataframe with target columns.
        pred_df : pd.DataFrame
            Predictions dataframe.
        split : str, default='validation'
            Split name for labeling plots.
        """
        # Check if any target columns have ground truth
        has_targets = any(t in df_true.columns for t in self.target_cols)
        if not has_targets:
            return

        # Determine if we should save plots
        should_save = self.mlflow_tracking or self.output_dir is not None
        if not should_save:
            return

        # Create temp directory for plots if using MLflow
        temp_dir: Optional[Path] = None
        if self.mlflow_tracking and self._mlflow_logger is not None:
            temp_dir = Path(tempfile.mkdtemp())
            plot_dir = temp_dir / "plots" / split
        elif self.output_dir is not None:
            plot_dir = self.output_dir / "plots" / split
        else:
            return

        plot_dir.mkdir(parents=True, exist_ok=True)

        # Generate parity plots for each target
        for i, target in enumerate(self.target_cols):
            if target not in df_true.columns:
                continue

            y_true = np.asarray(df_true[target].values)
            y_pred = np.asarray(pred_df[target].values)

            from admet.plot import GLASBEY_PALETTE
            from admet.plot.latex import latex_sanitize

            fig, ax = plot_parity(
                y_true,
                y_pred,
                title=f"{latex_sanitize(target)} ({split})",
                color=GLASBEY_PALETTE[i % len(GLASBEY_PALETTE)],
            )

            # Create safe filename
            safe_name = target.replace(" ", "_").replace("/", "_").replace("<", "lt").replace(">", "gt")
            plot_path = plot_dir / f"parity_{safe_name}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Generate metric bar charts
        metrics_df = compute_metrics_df(df_true, pred_df, self.target_cols)
        if not metrics_df.empty:
            from admet.plot.latex import latex_sanitize as sanitize_label

            for metric_name in METRIC_NAMES:
                if metric_name not in metrics_df.columns:
                    continue

                values = np.asarray(metrics_df[metric_name].values)
                labels = [sanitize_label(ep) for ep in metrics_df.index]

                fig, _ = plot_metric_bar(
                    values,
                    labels,
                    metric_name,
                    title=f"{metric_name} ({split})",
                )

                safe_metric = metric_name.replace(" ", "_").replace("/", "_")
                plot_path = plot_dir / f"metric_{safe_metric}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

        # Save predictions as CSV artifact
        predictions_dir = plot_dir.parent.parent / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Create combined dataframe with SMILES, ground truth, and predictions
        predictions_csv = pd.DataFrame()
        if self.smiles_col in df_true.columns:
            predictions_csv[self.smiles_col] = df_true[self.smiles_col].values

        for target in self.target_cols:
            if target in df_true.columns:
                predictions_csv[f"{target}_true"] = df_true[target].values
            if target in pred_df.columns:
                predictions_csv[f"{target}_pred"] = pred_df[target].values

        predictions_path = predictions_dir / f"{split}_predictions.csv"
        predictions_csv.to_csv(predictions_path, index=False)

        # Log plots and predictions to MLflow
        if self._mlflow_client is not None and self.mlflow_run_id is not None:
            # Log all files in the plot directory
            for plot_file in plot_dir.iterdir():
                if plot_file.is_file():
                    self._mlflow_client.log_artifact(self.mlflow_run_id, str(plot_file), artifact_path=f"plots/{split}")
            # Log predictions CSV
            self._mlflow_client.log_artifact(self.mlflow_run_id, str(predictions_path), artifact_path="predictions")

            # Clean up temp directory (only exists when using MLflow)
            if temp_dir is not None:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)


def train_from_config(config_path: str, log_level: str = "INFO") -> None:
    """
    Train a ChempropModel from a YAML configuration file.

    This function loads the configuration from a YAML file, creates a model,
    trains it, and generates predictions for test and blind datasets if specified.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    log_level : str, default="INFO"
        Logging level. Options: "DEBUG", "INFO", "WARNING", "ERROR".

    Examples
    --------
    >>> train_from_config("configs/example_chemprop.yaml")
    >>> train_from_config("configs/experiment.yaml", log_level="DEBUG")
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading configuration from: %s", config_path)

    # Load configuration from YAML
    config = OmegaConf.merge(
        OmegaConf.structured(ChempropConfig),
        OmegaConf.load(config_path),
    )

    # Log the configuration
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Create model from config
    logger.info("Creating model from configuration...")
    model = ChempropModel.from_config(config)  # type: ignore[arg-type]

    # Train the model
    logger.info("Starting training...")
    model.fit()

    # Generate predictions for test set if available
    if model.dataframes["test"] is not None:
        logger.info("Generating predictions for test set...")
        _ = model.predict(
            model.dataframes["test"],
            generate_plots=True,
            split_name="test",
        )

    # Generate predictions for blind set if available
    if model.dataframes["blind"] is not None:
        logger.info("Generating predictions for blind set...")
        _ = model.predict(
            model.dataframes["blind"],
            generate_plots=False,
            split_name="blind",
        )

    # Export config for reproducibility
    exported_config = model.to_config()
    logger.info("Exported config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(exported_config)))

    # Close the MLflow run when done
    model.close()
    logger.info("Training complete!")


def main() -> None:
    """
    CLI entrypoint for training a ChempropModel from a YAML configuration.

    Usage
    -----
    python -m admet.model.chemprop.model --config configs/example_chemprop.yaml
    python -m admet.model.chemprop.model -c configs/experiment.yaml --log-level DEBUG
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Chemprop MPNN model from a YAML configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m admet.model.chemprop.model --config configs/example_chemprop.yaml
  python -m admet.model.chemprop.model -c configs/experiment.yaml --log-level DEBUG

Configuration file format:
  See configs/example_chemprop.yaml for a complete example.
  The configuration has four sections:
    - data: File paths and column specifications
    - model: Neural network architecture settings
    - optimization: Training hyperparameters
    - mlflow: Experiment tracking settings
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    train_from_config(args.config, args.log_level)


if __name__ == "__main__":
    main()
