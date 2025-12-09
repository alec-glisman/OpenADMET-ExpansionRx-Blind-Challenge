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
    aggregation : str, default="mean"
        Aggregation method for message passing. Options: 'mean', 'sum', 'norm'.
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
    aggregation: str = "mean"


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
    task_sampling_alpha: Optional[float] = None
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
    task_sampling_alpha: Optional[float] = None


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
    run_id : str, optional
        Existing MLflow run ID to attach to. If provided, the model will
        log to this existing run instead of creating a new one.
    parent_run_id : str, optional
        Parent run ID for creating nested runs. Used for ensemble training
        where individual models are nested under a parent ensemble run.
    nested : bool, default=False
        Whether to create a nested run under the parent_run_id.
    """

    tracking: bool = True
    tracking_uri: Optional[str] = None
    experiment_name: str = "chemprop"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    nested: bool = False


@dataclass
class CurriculumConfig:
    """
    Configuration for quality-aware curriculum learning.

    Curriculum learning progressively adjusts the sampling of training data
    based on quality labels. The curriculum proceeds through phases:
    - warmup: Focus on high-quality data
    - expand: Gradually include medium-quality data
    - robust: Include low-quality data for robustness
    - polish: Return focus to high-quality data

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable curriculum learning. When disabled, all samples
        are weighted equally regardless of quality.
    quality_col : str, default="Quality"
        Column name in the data containing quality labels (e.g., "high",
        "medium", "low").
    qualities : List[str], default=["high", "medium", "low"]
        Ordered list of quality levels from highest to lowest quality.
        The curriculum adapts to the number of qualities provided.
    patience : int, default=5
        Number of epochs without improvement in overall validation loss
        before advancing to the next curriculum phase.
    seed : int, default=42
        Random seed for reproducible curriculum sampling.
    strategy : str, default="sampled"
        Curriculum strategy: "sampled" uses weighted random sampling to
        control which data points appear in each batch; "weighted" applies
        quality-based loss weights (not yet implemented).
    reset_early_stopping_on_phase_change : bool, default=False
        Whether to reset early stopping patience when advancing to a new
        curriculum phase. This allows the model more time to adapt to
        the new data distribution.
    log_per_quality_metrics : bool, default=True
        Whether to log per-quality validation metrics (e.g., val_loss_high,
        val_loss_medium) in addition to overall metrics.
    """

    enabled: bool = False
    quality_col: str = "Quality"
    qualities: List[str] = field(default_factory=lambda: ["high", "medium", "low"])
    patience: int = 5
    seed: int = 42
    strategy: str = "sampled"
    reset_early_stopping_on_phase_change: bool = False
    log_per_quality_metrics: bool = True


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
    curriculum : CurriculumConfig
        Curriculum learning configuration for quality-aware sampling.

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
    ...     curriculum=CurriculumConfig(enabled=True, quality_col="Quality"),
    ... )
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass
class EnsembleDataConfig:
    """
    Configuration for ensemble data paths and columns.

    Parameters
    ----------
    data_dir : str
        Root directory containing split_*/fold_*/ subdirectories.
        Expected structure: {data_dir}/split_*/fold_*/train.csv
    test_file : str, optional
        Path to test data CSV file (global for all ensemble members).
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
        Directory to save ensemble outputs. If None, uses temp directory.
    splits : List[int], optional
        Specific split indices to use. If None, uses all found splits.
    folds : List[int], optional
        Specific fold indices to use. If None, uses all found folds.
    """

    data_dir: str = MISSING
    test_file: Optional[str] = None
    blind_file: Optional[str] = None
    smiles_col: str = "SMILES"
    target_cols: List[str] = field(default_factory=list)
    target_weights: List[float] = field(default_factory=list)
    output_dir: Optional[str] = None
    splits: Optional[List[int]] = None
    folds: Optional[List[int]] = None


@dataclass
class EnsembleConfig:
    """
    Complete configuration for ensemble Chemprop training.

    This configuration extends the single-model config to support
    training multiple models across splits and folds with Ray-based
    parallelization.

    Parameters
    ----------
    data : EnsembleDataConfig
        Ensemble data paths and column configuration.
    model : ModelConfig
        Model architecture configuration (shared across ensemble).
    optimization : OptimizationConfig
        Training optimization configuration (shared across ensemble).
    mlflow : MlflowConfig
        MLflow tracking configuration.
    curriculum : CurriculumConfig
        Curriculum learning configuration. When enabled, all ensemble
        members share the same curriculum schedule for consistent
        quality-aware sampling.
    max_parallel : int, default=1
        Maximum number of models to train in parallel.
        Set based on available GPU memory.
    ray_num_cpus : int, optional
        Number of CPUs to allocate to Ray. If None, uses all available.
    ray_num_gpus : int, optional
        Number of GPUs to allocate to Ray. If None, auto-detects.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from admet.model.chemprop.config import EnsembleConfig
    >>>
    >>> config = OmegaConf.merge(
    ...     OmegaConf.structured(EnsembleConfig),
    ...     OmegaConf.load("ensemble_config.yaml")
    ... )
    >>> ensemble = ChempropEnsemble.from_config(config)
    >>> ensemble.train_all()
    """

    data: EnsembleDataConfig = field(default_factory=EnsembleDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    max_parallel: int = 1
    ray_num_cpus: Optional[int] = None
    ray_num_gpus: Optional[int] = None
