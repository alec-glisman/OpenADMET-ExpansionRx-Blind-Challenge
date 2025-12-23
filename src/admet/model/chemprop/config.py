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
from typing import Dict, List, Optional

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
class TaskOversamplingConfig:
    """
    Configuration for task-aware oversampling of sparse tasks.

    Task oversampling adjusts sampling weights using inverse-power scheduling
    to give more weight to samples with labels for rare tasks, helping to
    balance learning across tasks with very different label counts.

    Parameters
    ----------
    alpha : float, default=0.5
        Power law exponent controlling oversampling strength.
        - alpha=0: Uniform task sampling (no rebalancing)
        - alpha=0.5: Moderate rebalancing (default)
        - alpha=1: Full inverse-proportional (rare tasks heavily favored)
        Valid range [0, 1]. Values outside this range will trigger a warning.
    """

    alpha: float = 0.5


@dataclass
class CurriculumConfig:
    """
    Configuration for quality-aware curriculum learning.

    Curriculum learning progressively adjusts the sampling of training data
    based on quality labels. The curriculum proceeds through phases:
    - warmup: Focus on high-quality data
    - expand: Gradually include medium-quality data
    - robust: Include low-quality data for robustness
    - polish: Return focus to high-quality data (but maintain some diversity)

    Count Normalization
    -------------------
    When `count_normalize=True` (default), the phase proportions represent the
    actual fraction of training samples you want from each quality level,
    regardless of dataset sizes. This compensates for imbalanced datasets.

    For example, with High=5k, Medium=100k, Low=15k samples:
    - warmup_proportions: [0.8, 0.15, 0.05] means 80% of batches contain
      high-quality samples, even though high-quality is only 4% of raw data.

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
    count_normalize : bool, default=True
        If True, interpret weights as target proportions and automatically
        adjust for dataset size imbalance. If False, apply weights directly
        to samples (legacy behavior where larger datasets dominate).
    min_high_quality_proportion : float, default=0.25
        Minimum proportion of high-quality data in any phase. Acts as a safety
        floor to prevent catastrophic forgetting of high-quality patterns.
    warmup_proportions : List[float], optional
        Target proportions for warmup phase [high, medium, low].
        Default: [0.80, 0.15, 0.05] for 3 qualities.
    expand_proportions : List[float], optional
        Target proportions for expand phase [high, medium, low].
        Default: [0.60, 0.30, 0.10] for 3 qualities.
    robust_proportions : List[float], optional
        Target proportions for robust phase [high, medium, low].
        Default: [0.50, 0.35, 0.15] for 3 qualities.
    polish_proportions : List[float], optional
        Target proportions for polish phase [high, medium, low].
        Default: [0.70, 0.20, 0.10] for 3 qualities.
    phase_weights_two_quality : Dict[str, List[float]], optional
        Custom phase weights for 2 quality levels. If None, uses defaults.
    phase_weights_three_quality : Dict[str, List[float]], optional
        Custom phase weights for 3 quality levels. If None, uses defaults.
    """

    enabled: bool = False
    quality_col: str = "Quality"
    qualities: List[str] = field(default_factory=lambda: ["high", "medium", "low"])
    patience: int = 5
    seed: int = 42
    save_plots: bool = False
    plot_dpi: int = 150
    strategy: str = "sampled"
    reset_early_stopping_on_phase_change: bool = False
    log_per_quality_metrics: bool = True
    phase_weights_two_quality: Optional[Dict[str, List[float]]] = None
    phase_weights_three_quality: Optional[Dict[str, List[float]]] = None

    # Count normalization settings
    count_normalize: bool = True
    min_high_quality_proportion: float = 0.25

    # HPO-friendly target proportions for each phase (when count_normalize=True)
    # These are the actual proportions of training samples you want from each quality
    warmup_proportions: Optional[List[float]] = None
    expand_proportions: Optional[List[float]] = None
    robust_proportions: Optional[List[float]] = None
    polish_proportions: Optional[List[float]] = None

    # Metric alignment: monitor high-quality metrics for curriculum and early stopping
    # Examples: "val/mae/high", "val_loss", "val/rmse/high"
    monitor_metric: str = "val_loss"
    early_stopping_metric: Optional[str] = None  # If None, uses monitor_metric

    # Adaptive curriculum: auto-adjust proportions based on per-quality performance
    adaptive_enabled: bool = False
    adaptive_improvement_threshold: float = 0.02  # 2% relative improvement required
    adaptive_max_adjustment: float = 0.1  # Max 10% adjustment per phase transition
    adaptive_lookback_epochs: int = 5  # Compare current vs N epochs ago

    # Loss weighting: scale gradients by quality level
    loss_weighting_enabled: bool = False
    loss_weights: Optional[Dict[str, float]] = None  # e.g., {"high": 1.0, "medium": 0.5, "low": 0.3}


@dataclass
class JointSamplingConfig:
    """
    Configuration for unified sampling combining task-aware and curriculum-aware strategies.

    JointSampling combines two complementary sampling strategies via multiplicative
    weight composition:

    1. **Task oversampling**: Rebalances sampling across tasks with different label
       counts using inverse-power weighting (controlled by alpha).
    2. **Curriculum learning**: Adjusts sampling based on data quality labels that
       change with curriculum phase progression.

    The joint weight for sample i is computed as:
        w_joint[i] = w_task[i] × w_curriculum[i]

    For multi-task samples, the "primary" task (used for weight computation) is
    the rarest task among those the sample has labels for.

    Parameters
    ----------
    enabled : bool, default=False
        Master switch for joint sampling. When False, standard shuffling is used.
    task_oversampling : TaskOversamplingConfig
        Configuration for task-aware oversampling.
    curriculum : CurriculumConfig
        Configuration for curriculum-aware sampling.
    num_samples : Optional[int], default=None
        Number of samples per epoch. If None, uses dataset length.
    seed : int, default=42
        Base random seed for sampling reproducibility.
    increment_seed_per_epoch : bool, default=True
        If True, increments seed each epoch for sampling variety.
        If False, uses same seed each epoch for reproducibility.
    log_to_mlflow : bool, default=True
        Whether to log sampling statistics (entropy, effective samples, etc.)
        to MLflow each epoch.

    Examples
    --------
    >>> from admet.model.chemprop.config import JointSamplingConfig
    >>>
    >>> # Task oversampling only
    >>> config = JointSamplingConfig(
    ...     enabled=True,
    ...     task_oversampling=TaskOversamplingConfig(alpha=0.3),
    ...     curriculum=CurriculumConfig(enabled=False),
    ... )
    >>>
    >>> # Curriculum learning only
    >>> config = JointSamplingConfig(
    ...     enabled=True,
    ...     task_oversampling=TaskOversamplingConfig(alpha=0.0),
    ...     curriculum=CurriculumConfig(enabled=True, quality_col="Quality"),
    ... )
    >>>
    >>> # Both strategies combined
    >>> config = JointSamplingConfig(
    ...     enabled=True,
    ...     task_oversampling=TaskOversamplingConfig(alpha=0.5),
    ...     curriculum=CurriculumConfig(enabled=True, quality_col="Quality"),
    ... )
    """

    enabled: bool = False
    task_oversampling: TaskOversamplingConfig = field(default_factory=TaskOversamplingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    num_samples: Optional[int] = None
    seed: int = 42
    increment_seed_per_epoch: bool = True
    log_to_mlflow: bool = True


@dataclass
class TaskAffinityConfig:
    """
    Configuration for task affinity grouping (legacy pre-training approach).

    Task affinity grouping implements the TAG algorithm from
    "Efficiently Identifying Task Groupings for Multi-Task Learning"
    (Fifty et al., NeurIPS 2021). It computes inter-task affinity scores
    by measuring gradient similarity between tasks during a short training
    run, then clusters tasks into groups that benefit from being trained
    together.

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable task affinity grouping. When enabled, a short
        pre-training phase computes task affinities before the main
        training loop.
    affinity_epochs : int, default=1
        Number of epochs for the affinity computation phase.
    affinity_batch_size : int, default=64
        Batch size during affinity computation.
    affinity_lr : float, default=1e-3
        Learning rate during affinity computation.
    n_groups : int, default=3
        Number of task groups to create via clustering.
    clustering_method : str, default="agglomerative"
        Clustering algorithm: "agglomerative" or "spectral".
    affinity_type : str, default="cosine"
        Type of affinity to compute: "cosine" or "dot_product".
    seed : int, default=42
        Random seed for reproducibility.
    """

    enabled: bool = False
    affinity_epochs: int = 1
    affinity_batch_size: int = 64
    affinity_lr: float = 1.0e-3
    n_groups: int = 3
    clustering_method: str = "agglomerative"
    affinity_type: str = "cosine"
    seed: int = 42


@dataclass
class InterTaskAffinityConfig:
    """
    Configuration for inter-task affinity computation during training.

    This implements the lookahead-based inter-task affinity from the TAG paper:

        Z^t_{ij} = 1 - L_j(θ^{t+1}_{s|i}) / L_j(θ^t_s)

    Where θ^{t+1}_{s|i} = θ^t_s - η∇_{θ_s} L_i is the shared parameter update
    from task i's gradient. This measures how task i's update affects task j.

    Unlike the legacy :class:`TaskAffinityConfig`, this computes affinity
    during the main training loop (not as a separate pre-training phase) and
    logs per-step and epoch-level metrics to MLflow.

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable inter-task affinity computation during training.
    compute_every_n_steps : int, default=1
        Compute affinity every N training steps. Higher values reduce
        computational overhead but provide less granular measurements.
    log_every_n_steps : int, default=100
        Log running average affinity to MLflow every N steps.
    log_epoch_summary : bool, default=True
        Log epoch-level summary statistics (mean, std) of affinity matrix.
    log_step_matrices : bool, default=False
        Log individual step affinity matrices. WARNING: High volume.
    lookahead_lr : float, default=0.001
        Learning rate η for computing the lookahead parameter update.
    use_optimizer_lr : bool, default=True
        If True, uses the current optimizer learning rate for lookahead.
    shared_param_patterns : List[str], optional
        Patterns that explicitly mark encoder parameters as shared.
    exclude_param_patterns : List[str]
        Patterns to exclude from shared parameters (task-specific layers).
    n_groups : Optional[int], default=None
        If provided, cluster tasks into this many groups using the final
        affinity matrix (TAG paper grouping step).
    clustering_method : str, default="agglomerative"
        Clustering algorithm for grouping: "agglomerative" or "spectral".
    clustering_linkage : str, default="average"
        Linkage for agglomerative clustering when grouping tasks.
    device : str, default="auto"
        Device hint for affinity computation. "auto" selects CUDA if available.
    log_to_mlflow : bool, default=True
        Whether to log affinity metrics to MLflow.
    save_plots : bool, default=False
        Save affinity matrix plots as MLflow artifacts when enabled.
    plot_formats : List[str], default=["png"]
        Image formats to emit when saving affinity plots.
    plot_dpi : int, default=150
        DPI for any saved plots.

    See Also
    --------
    admet.model.chemprop.inter_task_affinity : Full implementation details.
    """

    enabled: bool = False
    compute_every_n_steps: int = 1
    log_every_n_steps: int = 100
    log_epoch_summary: bool = True
    log_step_matrices: bool = False
    lookahead_lr: float = 0.001
    use_optimizer_lr: bool = True
    shared_param_patterns: List[str] = field(default_factory=list)
    exclude_param_patterns: List[str] = field(default_factory=lambda: ["predictor", "ffn", "output", "head", "readout"])
    n_groups: Optional[int] = None
    clustering_method: str = "agglomerative"
    clustering_linkage: str = "average"
    device: str = "auto"
    log_to_mlflow: bool = True
    save_plots: bool = False
    plot_formats: List[str] = field(default_factory=lambda: ["png"])
    plot_dpi: int = 150


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
    joint_sampling : JointSamplingConfig
        Unified configuration for task-aware oversampling and curriculum learning.
    task_affinity : TaskAffinityConfig
        Task affinity grouping configuration for multi-task learning.
    inter_task_affinity : InterTaskAffinityConfig
        Inter-task affinity computation during training (TAG paper lookahead).

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
    ...     joint_sampling=JointSamplingConfig(
    ...         enabled=True,
    ...         task_oversampling=TaskOversamplingConfig(alpha=0.5),
    ...         curriculum=CurriculumConfig(enabled=True, quality_col="Quality"),
    ...     ),
    ...     task_affinity=TaskAffinityConfig(enabled=True, n_groups=2),
    ...     inter_task_affinity=InterTaskAffinityConfig(enabled=True),
    ... )
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)
    inter_task_affinity: InterTaskAffinityConfig = field(default_factory=InterTaskAffinityConfig)


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
class RayConfig:
    """
    Configuration for Ray parallelization settings.

    Parameters
    ----------
    max_parallel : int, default=1
        Maximum number of models to train in parallel.
        Set based on available GPU memory. For example, if you have
        1 GPU and each model needs 0.5 GPU, set max_parallel=2.
    num_cpus : int, optional
        Number of CPUs to allocate to Ray. If None, uses all available CPUs.
    num_gpus : int, optional
        Number of GPUs to allocate to Ray. If None, auto-detects available GPUs.

    Examples
    --------
    >>> ray = RayConfig(max_parallel=2, num_cpus=8, num_gpus=1)
    """

    max_parallel: int = 1
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None


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
    joint_sampling : JointSamplingConfig
        Unified configuration for task-aware oversampling and curriculum learning.
        When curriculum is enabled, all ensemble members share the same curriculum
        schedule for consistent quality-aware sampling.
    ray : RayConfig
        Ray parallelization configuration for distributed ensemble training.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from admet.model.chemprop.config import EnsembleConfig
    >>>
    >>> config = OmegaConf.merge(
    ...     OmegaConf.structured(EnsembleConfig),
    ...     OmegaConf.load("ensemble_config.yaml")
    ... )
    >>> ensemble = ModelEnsemble.from_config(config)
    >>> ensemble.train_all()
    """

    data: EnsembleDataConfig = field(default_factory=EnsembleDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)
    inter_task_affinity: InterTaskAffinityConfig = field(default_factory=InterTaskAffinityConfig)
    ray: RayConfig = field(default_factory=RayConfig)
