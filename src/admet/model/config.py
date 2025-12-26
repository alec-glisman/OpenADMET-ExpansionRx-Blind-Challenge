"""Unified configuration dataclasses for all ADMET models.

This module provides base configuration classes that all model-specific
configs inherit from, ensuring consistent structure and behavior.

Key classes:
    - UnifiedModelConfig: Master config for any model type
    - UnifiedDataConfig: Universal data configuration
    - UnifiedMlflowConfig: MLflow tracking configuration
    - UnifiedOptimizationConfig: Training optimization settings
    - JointSamplingConfig: Task oversampling + curriculum learning
    - TaskAffinityConfig: Task affinity grouping
    - InterTaskAffinityConfig: Inter-task affinity computation

The `model.type` field discriminates which model-specific parameters to use.
Training strategies (JointSampling, Curriculum) are at root level and
validated for model-type compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class BaseDataConfig:
    """Base configuration for data paths and columns.

    Shared across all model types. Model-specific data configs
    can inherit and extend this class.

    Parameters:
        data_dir: Directory containing train.csv and validation.csv files.
        test_file: Path to test data CSV file.
        blind_file: Path to blind test data CSV file (no labels).
        smiles_col: Column name containing SMILES strings.
        target_cols: List of target column names for multi-task prediction.
        target_weights: Per-task weights for the loss function.
        output_dir: Directory to save model checkpoints and outputs.
    """

    data_dir: str = MISSING
    test_file: str | None = None
    blind_file: str | None = None
    smiles_col: str = "SMILES"
    target_cols: list[str] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    output_dir: str | None = None


@dataclass
class BaseMlflowConfig:
    """Base configuration for MLflow experiment tracking.

    Shared across all model types to ensure consistent tracking behavior.

    Parameters:
        enabled: Whether to enable MLflow tracking.
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        run_name: Optional run name for the MLflow run.
        run_id: Existing MLflow run ID to attach to.
        parent_run_id: Parent run ID for nested runs (ensemble).
        nested: Whether to create a nested run.
        log_model: Whether to log the trained model as artifact.
    """

    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "admet"
    run_name: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    nested: bool = False
    log_model: bool = True


# Valid model types (documentation and validation)
# OmegaConf does not support Literal types, so we use str with runtime validation
MODEL_TYPES = ("chemprop", "chemeleon", "xgboost", "lightgbm", "catboost")
PYTORCH_MODEL_TYPES = ("chemprop", "chemeleon")
CLASSICAL_MODEL_TYPES = ("xgboost", "lightgbm", "catboost")


@dataclass
class BaseModelConfig:
    """Base configuration inherited by all model-specific configs.

    This class defines the common structure for model configurations.
    The actual model parameters are nested under a model-type-specific
    section (e.g., config.model.chemprop, config.model.xgboost).

    Parameters:
        type: Model type discriminator ("chemprop", "xgboost", etc.).
        data: Data configuration section.
        mlflow: MLflow tracking configuration.
    """

    type: str = MISSING
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
    mlflow: BaseMlflowConfig = field(default_factory=BaseMlflowConfig)


# ============================================================================
# Fingerprint Configuration (for classical models)
# ============================================================================


@dataclass
class MorganFingerprintConfig:
    """Configuration for Morgan (circular) fingerprints.

    Parameters:
        radius: Morgan fingerprint radius.
        n_bits: Number of bits in the fingerprint.
        use_chirality: Whether to include chirality information.
        use_bond_types: Whether to distinguish bond types.
        use_features: Whether to use feature-based invariants.
    """

    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = False
    use_bond_types: bool = True
    use_features: bool = False


@dataclass
class RDKitFingerprintConfig:
    """Configuration for RDKit topological fingerprints.

    Parameters:
        min_path: Minimum path length.
        max_path: Maximum path length.
        n_bits: Number of bits in the fingerprint.
        branched_paths: Whether to include branched paths.
    """

    min_path: int = 1
    max_path: int = 7
    n_bits: int = 2048
    branched_paths: bool = True


@dataclass
class MACCSConfig:
    """Configuration for MACCS keys fingerprints.

    MACCS keys have a fixed 167-bit representation.
    """

    pass


@dataclass
class MordredConfig:
    """Configuration for Mordred molecular descriptors.

    Parameters:
        ignore_3d: Whether to ignore 3D descriptors (requires conformer).
        normalize: Whether to normalize descriptor values.
    """

    ignore_3d: bool = True
    normalize: bool = True


# Type alias for fingerprint types (documentation only - not used in dataclass fields)
# OmegaConf does not support Literal types, so we use str with runtime validation
FINGERPRINT_TYPES = ("morgan", "rdkit", "maccs", "mordred")


@dataclass
class FingerprintConfig:
    """Configuration for molecular fingerprint/descriptor generation.

    Used by classical models (XGBoost, LightGBM, CatBoost) that require
    fixed-length feature vectors instead of molecular graphs.

    Parameters:
        type: Fingerprint type to use ("morgan", "rdkit", "maccs", "mordred").
        morgan: Morgan fingerprint settings.
        rdkit: RDKit fingerprint settings.
        maccs: MACCS keys settings.
        mordred: Mordred descriptor settings.
    """

    type: str = "morgan"  # One of FINGERPRINT_TYPES
    morgan: MorganFingerprintConfig = field(default_factory=MorganFingerprintConfig)
    rdkit: RDKitFingerprintConfig = field(default_factory=RDKitFingerprintConfig)
    maccs: MACCSConfig = field(default_factory=MACCSConfig)
    mordred: MordredConfig = field(default_factory=MordredConfig)


# ============================================================================
# Chemprop-specific Configuration
# ============================================================================


@dataclass
class ChempropModelParams:
    """Chemprop-specific model architecture parameters.

    Parameters:
        depth: Number of message passing iterations.
        message_hidden_dim: Hidden dimension for message passing.
        dropout: Dropout probability.
        num_layers: Number of feed-forward layers.
        hidden_dim: Hidden dimension for FFN layers.
        batch_norm: Whether to use batch normalization.
        ffn_type: Type of feed-forward network.
        trunk_n_layers: Number of trunk layers for branched FFN.
        trunk_hidden_dim: Hidden dimension for trunk layers.
        n_experts: Number of experts for mixture of experts FFN.
        aggregation: Aggregation method for message passing.
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
class ChempropOptimizationParams:
    """Chemprop optimization/training parameters.

    Parameters:
        criterion: Loss function name.
        init_lr: Initial learning rate.
        max_lr: Maximum learning rate (OneCycle peak).
        final_lr: Final learning rate.
        warmup_epochs: Number of warmup epochs.
        patience: Early stopping patience.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        num_workers: Data loader workers.
        seed: Random seed.
        progress_bar: Whether to show progress bar.
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


# ============================================================================
# Chemeleon-specific Configuration
# ============================================================================


@dataclass
class UnfreezeScheduleConfig:
    """Configuration for gradual unfreezing of encoder/decoder.

    Supports scheduled unfreezing where the encoder (message passing)
    starts frozen and is optionally unfrozen at a specified epoch.

    Parameters:
        freeze_encoder: Whether to freeze the encoder initially.
        unfreeze_encoder_epoch: Epoch at which to unfreeze encoder.
        unfreeze_at_epoch: Alias for unfreeze_encoder_epoch (YAML compatibility).
        unfreeze_encoder_lr_multiplier: LR multiplier for unfrozen encoder.
        freeze_decoder_initially: Whether to freeze decoder initially.
        unfreeze_decoder_epoch: Epoch at which to unfreeze decoder.
    """

    freeze_encoder: bool = True
    unfreeze_encoder_epoch: Optional[int] = None
    unfreeze_at_epoch: Optional[int] = None  # Alias for YAML compatibility
    unfreeze_encoder_lr_multiplier: float = 0.1
    freeze_decoder_initially: bool = False
    unfreeze_decoder_epoch: Optional[int] = None


@dataclass
class ChemeleonHeadConfig:
    """Configuration for Chemeleon prediction head.

    Parameters:
        hidden_dims: List of hidden dimensions for FFN layers.
        dropout: Dropout probability.
        activation: Activation function name.
    """

    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    activation: str = "relu"


@dataclass
class ChemeleonModelParams:
    """Chemeleon-specific model parameters.

    Parameters:
        checkpoint_path: Path to pretrained checkpoint or "auto" for download.
        zenodo_id: Zenodo record ID for downloading pretrained weights.
        zenodo_filename: Filename within the Zenodo record.
        model_cache_dir: Directory to cache downloaded models.
        unfreeze_encoder: Whether to unfreeze encoder (alternative to schedule).
        unfreeze_schedule: Gradual unfreezing configuration.
        head: Prediction head configuration.
        ffn_type: FFN architecture type ('regression', 'mixture_of_experts', 'branched').
        ffn_hidden_dim: Hidden dimension for FFN layers (legacy, use head.hidden_dims).
        ffn_num_layers: Number of FFN layers (legacy).
        dropout: Dropout probability (legacy, use head.dropout).
        batch_norm: Whether to use batch normalization.
        n_experts: Number of experts for MoE architecture.
        trunk_n_layers: Number of trunk layers for branched architecture.
        trunk_hidden_dim: Hidden dimension for trunk in branched architecture.
    """

    checkpoint_path: str = "auto"
    zenodo_id: Optional[str] = None
    zenodo_filename: Optional[str] = None
    model_cache_dir: Optional[str] = None
    unfreeze_encoder: bool = False
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    head: ChemeleonHeadConfig = field(default_factory=ChemeleonHeadConfig)
    ffn_type: str = "regression"
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    batch_norm: bool = False
    n_experts: Optional[int] = None
    trunk_n_layers: Optional[int] = None
    trunk_hidden_dim: Optional[int] = None


# ============================================================================
# Classical Model Configurations
# ============================================================================


@dataclass
class XGBoostModelParams:
    """XGBoost-specific model parameters.

    Parameters:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
        min_child_weight: Minimum sum of instance weight in child.
        gamma: Minimum loss reduction for split.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        random_state: Random seed (alias: seed).
        n_jobs: Number of parallel jobs (-1 = all cores).
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class LightGBMModelParams:
    """LightGBM-specific model parameters.

    Parameters:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (-1 for no limit).
        learning_rate: Boosting learning rate.
        num_leaves: Maximum number of leaves.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
        min_child_samples: Minimum samples in leaf.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        random_state: Random seed.
        n_jobs: Number of parallel jobs (-1 = all cores).
        verbose: Verbosity level (-1 = silent).
    """

    n_estimators: int = 100
    max_depth: int = -1
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


@dataclass
class CatBoostModelParams:
    """CatBoost-specific model parameters.

    Parameters:
        iterations: Number of boosting iterations.
        depth: Tree depth.
        learning_rate: Boosting learning rate.
        l2_leaf_reg: L2 regularization coefficient.
        subsample: Subsample ratio (bootstrap_type must be Bernoulli/MVS).
        rsm: Random subspace method (column sampling).
        random_strength: Randomness for scoring splits.
        random_seed: Random seed.
        verbose: Logging verbosity.
        thread_count: Number of threads (-1 = all cores).
    """

    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    subsample: Optional[float] = None
    rsm: Optional[float] = None
    random_strength: float = 1.0
    random_seed: int = 42
    verbose: bool = False
    thread_count: int = -1


# ============================================================================
# Training Strategy Configurations (Model-Agnostic)
# ============================================================================


@dataclass
class TaskOversamplingConfig:
    """Configuration for task-aware oversampling of sparse tasks.

    Task oversampling adjusts sampling weights using inverse-power scheduling
    to give more weight to samples with labels for rare tasks.

    Parameters:
        alpha: Power law exponent controlling oversampling strength.
            0=uniform, 0.5=moderate (default), 1=full inverse-proportional.
    """

    alpha: float = 0.5


@dataclass
class CurriculumConfig:
    """Configuration for quality-aware curriculum learning.

    Curriculum learning progressively adjusts sampling based on quality labels.
    Phases: warmup -> expand -> robust -> polish.

    Note: Only supported for PyTorch-based models (chemprop, chemeleon).

    Parameters:
        enabled: Whether to enable curriculum learning.
        quality_col: Column name containing quality labels.
        qualities: Ordered list of quality levels (highest to lowest).
        patience: Epochs without improvement before phase advance.
        seed: Random seed for reproducible sampling.
        strategy: "sampled" uses weighted random sampling.
        count_normalize: If True, interpret weights as target proportions.
        min_high_quality_proportion: Minimum proportion of high-quality data.
        monitor_metric: Metric to monitor for phase advancement.
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
    count_normalize: bool = True
    min_high_quality_proportion: float = 0.25
    warmup_proportions: Optional[List[float]] = None
    expand_proportions: Optional[List[float]] = None
    robust_proportions: Optional[List[float]] = None
    polish_proportions: Optional[List[float]] = None
    monitor_metric: str = "val_loss"
    early_stopping_metric: Optional[str] = None
    adaptive_enabled: bool = False
    adaptive_improvement_threshold: float = 0.02
    adaptive_max_adjustment: float = 0.1
    adaptive_lookback_epochs: int = 5
    loss_weighting_enabled: bool = False
    loss_weights: Optional[Dict[str, float]] = None


@dataclass
class JointSamplingConfig:
    """Configuration for unified task-aware and curriculum-aware sampling.

    Combines task oversampling (inverse-power weighting by task sparsity)
    with curriculum learning (quality-based phase progression).

    Note: Only supported for PyTorch-based models (chemprop, chemeleon).

    Parameters:
        enabled: Master switch for joint sampling.
        task_oversampling: Task-aware oversampling configuration.
        curriculum: Curriculum learning configuration.
        num_samples: Samples per epoch (None = dataset length).
        seed: Base random seed.
        increment_seed_per_epoch: If True, varies sampling each epoch.
        log_to_mlflow: Log sampling statistics to MLflow.
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
    """Configuration for task affinity grouping (TAG algorithm).

    Computes inter-task affinity scores via gradient similarity during
    a short pre-training phase, then clusters tasks into groups.

    Note: Only supported for PyTorch-based models (chemprop, chemeleon).

    Parameters:
        enabled: Whether to enable task affinity grouping.
        affinity_epochs: Epochs for affinity computation phase.
        affinity_batch_size: Batch size during affinity computation.
        affinity_lr: Learning rate during affinity computation.
        n_groups: Number of task groups to create.
        clustering_method: "agglomerative" or "spectral".
        affinity_type: "cosine" or "dot_product".
        seed: Random seed.
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
    """Configuration for inter-task affinity computation during training.

    Implements lookahead-based inter-task affinity from the TAG paper,
    computed during the main training loop.

    Note: Only supported for PyTorch-based models (chemprop, chemeleon).

    Parameters:
        enabled: Whether to enable inter-task affinity computation.
        compute_every_n_steps: Compute affinity every N steps.
        log_every_n_steps: Log running average every N steps.
        log_epoch_summary: Log epoch-level summary statistics.
        lookahead_lr: Learning rate for lookahead parameter update.
        use_optimizer_lr: Use current optimizer LR for lookahead.
        n_groups: Number of task groups for clustering.
        clustering_method: "agglomerative" or "spectral".
        log_to_mlflow: Log affinity metrics to MLflow.
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


# ============================================================================
# Unified Configuration (Master Schema)
# ============================================================================


@dataclass
class UnifiedDataConfig:
    """Universal data configuration for all model types.

    Consolidates data paths, columns, and ensemble settings.

    Parameters:
        data_dir: Directory containing train.csv and validation.csv files.
        train_file: Path to training data (alternative to data_dir).
        validation_file: Path to validation data.
        test_file: Path to test data CSV file.
        blind_file: Path to blind test data CSV file.
        output_dir: Directory to save outputs.
        smiles_col: Column name containing SMILES strings.
        target_cols: List of target column names.
        target_weights: Per-task weights for loss function.
        quality_col: Column for curriculum learning quality labels.
        splits: Specific split indices to use (None = all).
        folds: Specific fold indices to use (None = all).
    """

    data_dir: str = MISSING
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    blind_file: Optional[str] = None
    output_dir: Optional[str] = None
    smiles_col: str = "SMILES"
    target_cols: List[str] = field(default_factory=list)
    target_weights: List[float] = field(default_factory=list)
    quality_col: Optional[str] = None
    splits: Optional[List[int]] = None
    folds: Optional[List[int]] = None


@dataclass
class UnifiedMlflowConfig:
    """Universal MLflow tracking configuration.

    Uses consistent field names across all models.
    Note: 'enabled' is the canonical field; 'tracking' is an alias for
    backward compatibility with older configs.

    Parameters:
        enabled: Whether to enable MLflow tracking.
        tracking: Alias for enabled (for backward compatibility).
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        run_name: Optional run name.
        run_id: Existing run ID to attach to.
        parent_run_id: Parent run ID for nested runs.
        nested: Whether to create a nested run.
        log_model: Whether to log trained model as artifact.
    """

    enabled: bool = True
    tracking: Optional[bool] = None  # Alias for enabled
    tracking_uri: Optional[str] = None
    experiment_name: str = "admet"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    nested: bool = False
    log_model: bool = True


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler.

    Parameters:
        type: Scheduler type ('cosine', 'step', 'exponential', 'onecycle').
        warmup_epochs: Number of warmup epochs.
        step_size: Step size for StepLR scheduler.
        gamma: Decay factor for StepLR/ExponentialLR.
    """

    type: str = "cosine"
    warmup_epochs: int = 5
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class UnifiedOptimizationConfig:
    """Universal optimization configuration.

    Contains superset of parameters for all model types.
    Neural models use LR scheduling; classical models ignore those fields.

    Parameters:
        seed: Random seed.
        progress_bar: Whether to show progress bar.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Training batch size.
        num_workers: Data loader workers.
        learning_rate: Learning rate (alias for init_lr in YAML configs).
        weight_decay: Weight decay for optimizer.
        init_lr: Initial learning rate.
        max_lr: Maximum learning rate (OneCycle peak).
        final_lr: Final learning rate.
        warmup_epochs: Number of warmup epochs.
        scheduler: Learning rate scheduler configuration.
        criterion: Loss function name.
    """

    seed: int = 42
    progress_bar: bool = False
    max_epochs: int = 150
    patience: int = 15
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: Optional[float] = None  # Alias for init_lr
    weight_decay: float = 0.0
    init_lr: float = 1.0e-4
    max_lr: float = 1.0e-3
    final_lr: float = 1.0e-4
    warmup_epochs: int = 5
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    criterion: str = "MAE"


@dataclass
class RayConfig:
    """Configuration for Ray parallelization.

    Parameters:
        max_parallel: Maximum models to train in parallel.
        num_cpus: CPUs to allocate to Ray (None = auto).
        num_gpus: GPUs to allocate to Ray (None = auto).
    """

    max_parallel: int = 1
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None


@dataclass
class EnsembleSection:
    """Configuration for ensemble training mode.

    Parameters:
        enabled: Whether to use ensemble training mode.
        n_models: Number of models to train (for simple ensemble without splits).
        aggregation: Method to aggregate predictions ("mean" or "median").
        use_splits: Whether to use split/fold directory structure.
        splits: Specific split indices to use (None = all).
        folds: Specific fold indices to use (None = all).
    """

    enabled: bool = False
    n_models: int = 5
    aggregation: str = "mean"
    use_splits: bool = True
    splits: Optional[List[int]] = None
    folds: Optional[List[int]] = None


@dataclass
class ModelSection:
    """Model type discriminator and model-specific parameters.

    The 'type' field determines which nested config section is used.

    Parameters:
        type: Model type identifier.
        chemprop: Chemprop-specific parameters.
        chemeleon: Chemeleon-specific parameters.
        xgboost: XGBoost-specific parameters.
        lightgbm: LightGBM-specific parameters.
        catboost: CatBoost-specific parameters.
        fingerprint: Fingerprint config for classical models.
    """

    type: str = "chemprop"
    chemprop: ChempropModelParams = field(default_factory=ChempropModelParams)
    chemeleon: ChemeleonModelParams = field(default_factory=ChemeleonModelParams)
    xgboost: XGBoostModelParams = field(default_factory=XGBoostModelParams)
    lightgbm: LightGBMModelParams = field(default_factory=LightGBMModelParams)
    catboost: CatBoostModelParams = field(default_factory=CatBoostModelParams)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)


@dataclass
class UnifiedModelConfig:
    """Complete unified model configuration.

    Single schema for all model types. The 'model.type' field determines
    which model-specific section is used.

    Structure:
        model:
          type: "chemprop"  # or "chemeleon", "xgboost", "lightgbm", "catboost"
          chemprop: { ... }  # Model-specific params
        data: { ... }
        optimization: { ... }
        mlflow: { ... }
        ensemble: { ... }  # Ensemble training mode
        joint_sampling: { ... }  # Training strategies
        ray: { ... }

    Parameters:
        model: Model type and parameters section.
        data: Data configuration.
        optimization: Training optimization configuration.
        mlflow: MLflow tracking configuration.
        ensemble: Ensemble training configuration.
        joint_sampling: Unified sampling configuration.
        task_affinity: Task affinity grouping configuration.
        inter_task_affinity: Inter-task affinity configuration.
        ray: Ray parallelization configuration.
    """

    model: ModelSection = field(default_factory=ModelSection)
    data: UnifiedDataConfig = field(default_factory=UnifiedDataConfig)
    optimization: UnifiedOptimizationConfig = field(default_factory=UnifiedOptimizationConfig)
    mlflow: UnifiedMlflowConfig = field(default_factory=UnifiedMlflowConfig)
    ensemble: EnsembleSection = field(default_factory=EnsembleSection)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)
    inter_task_affinity: InterTaskAffinityConfig = field(default_factory=InterTaskAffinityConfig)
    ray: RayConfig = field(default_factory=RayConfig)


# ============================================================================
# Configuration Validation
# ============================================================================


class ConfigValidationError(ValueError):
    """Raised when configuration is invalid for the specified model type."""

    pass


def _get_config_attr(obj, attr: str, default):
    """Helper to get attribute from dict, DictConfig, or dataclass."""
    if obj is None:
        return default
    if isinstance(obj, (dict, DictConfig)):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def validate_model_config(config: DictConfig | UnifiedModelConfig) -> None:
    """Validate configuration for model-type compatibility.

    Raises ConfigValidationError if incompatible options are enabled.

    Parameters:
        config: Configuration to validate.

    Raises:
        ConfigValidationError: If curriculum/sampling enabled for classical models.
    """
    if isinstance(config, UnifiedModelConfig):
        config = OmegaConf.structured(config)

    model_type = _get_config_attr(_get_config_attr(config, "model", {}), "type", "chemprop")
    is_classical = model_type in CLASSICAL_MODEL_TYPES

    joint_sampling = _get_config_attr(config, "joint_sampling", {})

    if is_classical:
        js_enabled = _get_config_attr(joint_sampling, "enabled", False)

        if js_enabled:
            curriculum = _get_config_attr(joint_sampling, "curriculum", {})
            curriculum_enabled = _get_config_attr(curriculum, "enabled", False)

            if curriculum_enabled:
                raise ConfigValidationError(
                    f"Curriculum learning is not supported for {model_type} models. "
                    f"Curriculum requires PyTorch DataLoader which classical models "
                    f"do not use. Set joint_sampling.curriculum.enabled=false or use "
                    f"a neural model (chemprop, chemeleon)."
                )

            task_oversampling = _get_config_attr(joint_sampling, "task_oversampling", {})
            alpha = _get_config_attr(task_oversampling, "alpha", 0)

            if alpha > 0:
                raise ConfigValidationError(
                    f"Task oversampling is not supported for {model_type} models. "
                    f"Task oversampling requires PyTorch DataLoader. "
                    f"Set joint_sampling.task_oversampling.alpha=0 or use a neural model."
                )

    inter_task = _get_config_attr(config, "inter_task_affinity", {})
    inter_task_enabled = _get_config_attr(inter_task, "enabled", False)

    if is_classical and inter_task_enabled:
        raise ConfigValidationError(
            f"Inter-task affinity is not supported for {model_type} models. "
            f"Inter-task affinity requires gradient computation. "
            f"Set inter_task_affinity.enabled=false or use a neural model."
        )

    task_affinity = _get_config_attr(config, "task_affinity", {})
    task_affinity_enabled = _get_config_attr(task_affinity, "enabled", False)

    if is_classical and task_affinity_enabled:
        raise ConfigValidationError(
            f"Task affinity grouping is not supported for {model_type} models. "
            f"Set task_affinity.enabled=false or use a neural model."
        )


def get_structured_config_for_model_type(model_type: str) -> DictConfig:
    """Get structured config with appropriate defaults for model type.

    Returns a base configuration that can be merged with user config
    to ensure all fields have appropriate defaults.

    Parameters:
        model_type: Model type identifier.

    Returns:
        Structured config with model-type-appropriate defaults.

    Raises:
        ValueError: If model_type is unknown.

    Examples:
        >>> base = get_structured_config_for_model_type("xgboost")
        >>> user_config = OmegaConf.load("my_config.yaml")
        >>> config = OmegaConf.merge(base, user_config)
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Valid: {MODEL_TYPES}")

    base = OmegaConf.structured(UnifiedModelConfig)
    base.model.type = model_type

    if model_type in CLASSICAL_MODEL_TYPES:
        base.joint_sampling.enabled = False
        base.joint_sampling.task_oversampling.alpha = 0.0
        base.joint_sampling.curriculum.enabled = False
        base.task_affinity.enabled = False
        base.inter_task_affinity.enabled = False

    return base
