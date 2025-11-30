"""
OmegaConf-based configuration for ChempropModel.

This module provides structured configuration classes that can be loaded
from YAML files using OmegaConf, allowing full specification of training runs.

Examples
--------
>>> from omegaconf import OmegaConf
>>> from admet.model.chemprop.config import ChempropConfig
>>>
>>> # Load from YAML file
>>> config = OmegaConf.structured(ChempropConfig)
>>> config = OmegaConf.merge(config, OmegaConf.load("config.yaml"))
>>>
>>> # Create model from config
>>> model = ChempropModel.from_config(config)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    """
    Configuration for data paths and columns.

    Parameters
    ----------
    data_dir : str
        Directory containing train.csv and validation.csv files.
        Expected structure: {data_dir}/train.csv and {data_dir}/validation.csv
    test_file : str, optional
        Path to test data CSV file.
    blind_file : str, optional
        Path to blind test data CSV file (no labels).
    smiles_col : str, default="SMILES"
        Column name containing SMILES strings.
    target_cols : List[str]
        List of target column names for multi-task prediction.
    target_weights : List[float], optional
        Per-task weights for the loss function. If empty, all tasks
        are weighted equally (1.0 each).
    output_dir : str, optional
        Directory to save model checkpoints and outputs. If None,
        uses a temporary directory.
    """

    data_dir: str = MISSING
    test_file: Optional[str] = None
    blind_file: Optional[str] = None
    smiles_col: str = "SMILES"
    target_cols: List[str] = field(default_factory=list)
    target_weights: List[float] = field(default_factory=list)
    output_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """
    Configuration for Chemprop MPNN architecture.

    Parameters
    ----------
    depth : int, default=5
        Number of message passing iterations.
    message_hidden_dim : int, default=600
        Hidden dimension for message passing layer.
    dropout : float, default=0.1
        Dropout probability in feed-forward networks.
    num_layers : int, default=2
        Number of feed-forward layers.
    hidden_dim : int, default=600
        Hidden dimension for feed-forward layers.
    batch_norm : bool, default=True
        Whether to use batch normalization in MPNN.
    ffn_type : str, default="regression"
        Type of feed-forward network. Options: 'regression',
        'mixture_of_experts', 'branched'.
    trunk_n_layers : int, default=2
        Number of trunk layers for branched FFN.
    trunk_hidden_dim : int, default=600
        Hidden dimension for trunk layers in branched FFN.
    n_experts : int, default=4
        Number of experts for mixture of experts FFN.
    """

    depth: int = 5
    message_hidden_dim: int = 600
    dropout: float = 0.1
    num_layers: int = 2
    hidden_dim: int = 600
    batch_norm: bool = True
    ffn_type: str = "regression"
    trunk_n_layers: int = 2
    trunk_hidden_dim: int = 600
    n_experts: int = 4


@dataclass
class OptimizationConfig:
    """
    Configuration for training optimization.

    Parameters
    ----------
    criterion : str, default="MAE"
        Loss function. Options: "MAE", "MSE", "RMSE", "SID", "BCE",
        "CrossEntropy", "Dirichlet", "Evidential", "MVE", "Quantile".
    init_lr : float, default=1.0e-4
        Initial learning rate.
    max_lr : float, default=1.0e-3
        Maximum learning rate (peak of OneCycle).
    final_lr : float, default=1.0e-4
        Final learning rate at end of training.
    warmup_epochs : int, default=5
        Number of warmup epochs.
    patience : int, default=15
        Early stopping patience (epochs without improvement).
    max_epochs : int, default=150
        Maximum number of training epochs.
    batch_size : int, default=32
        Training batch size.
    num_workers : int, default=0
        Number of data loader workers.
    seed : int, default=12345
        Random seed for reproducibility.
    progress_bar : bool, default=False
        Whether to show training progress bar.
    """

    criterion: str = "MAE"
    init_lr: float = 1.0e-4
    max_lr: float = 1.0e-3
    final_lr: float = 1.0e-4
    warmup_epochs: int = 5
    patience: int = 15
    max_epochs: int = 150
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 12345
    progress_bar: bool = False


@dataclass
class MlflowConfig:
    """
    Configuration for MLflow experiment tracking.

    Parameters
    ----------
    tracking : bool, default=True
        Whether to enable MLflow tracking for params, metrics, and artifacts.
    tracking_uri : str, optional
        MLflow tracking server URI. If None, uses default local storage.
    experiment_name : str, default="chemprop"
        MLflow experiment name to log runs under.
    run_name : str, optional
        Optional run name for the MLflow run. If None, MLflow generates one.
    """

    tracking: bool = True
    tracking_uri: Optional[str] = None
    experiment_name: str = "chemprop"
    run_name: Optional[str] = None


@dataclass
class ChempropConfig:
    """
    Complete configuration for a Chemprop training run.

    This configuration can be loaded from a YAML file using OmegaConf
    and passed to ChempropModel.from_config() to create a fully
    configured model.

    Parameters
    ----------
    data : DataConfig
        Data paths and column configuration.
    model : ModelConfig
        Model architecture configuration.
    optimization : OptimizationConfig
        Training optimization configuration.
    mlflow : MlflowConfig
        MLflow tracking configuration.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from admet.model.chemprop.config import ChempropConfig
    >>>
    >>> # Load from YAML
    >>> config = OmegaConf.merge(
    ...     OmegaConf.structured(ChempropConfig),
    ...     OmegaConf.load("config.yaml")
    ... )
    >>>
    >>> # Or create programmatically
    >>> config = ChempropConfig(
    ...     data=DataConfig(
    ...         train_file="train.csv",
    ...         target_cols=["LogD", "Log KSOL"],
    ...     ),
    ...     model=ModelConfig(depth=4, hidden_dim=512),
    ...     optimization=OptimizationConfig(max_epochs=100),
    ...     mlflow=MlflowConfig(experiment_name="my_experiment"),
    ... )
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
