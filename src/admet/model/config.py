"""Unified configuration dataclasses for all ADMET models.

This module provides base configuration classes that all model-specific
configs inherit from, ensuring consistent structure and behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omegaconf import MISSING


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


# Type alias for model types
ModelType = Literal["chemprop", "chemeleon", "xgboost", "lightgbm", "catboost"]


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


# Type alias for fingerprint types
FingerprintType = Literal["morgan", "rdkit", "maccs", "mordred"]


@dataclass
class FingerprintConfig:
    """Configuration for molecular fingerprint/descriptor generation.

    Used by classical models (XGBoost, LightGBM, CatBoost) that require
    fixed-length feature vectors instead of molecular graphs.

    Parameters:
        type: Fingerprint type to use.
        morgan: Morgan fingerprint settings.
        rdkit: RDKit fingerprint settings.
        maccs: MACCS keys settings.
        mordred: Mordred descriptor settings.
    """

    type: FingerprintType = "morgan"
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
        unfreeze_encoder_lr_multiplier: LR multiplier for unfrozen encoder.
        freeze_decoder_initially: Whether to freeze decoder initially.
        unfreeze_decoder_epoch: Epoch at which to unfreeze decoder.
    """

    freeze_encoder: bool = True
    unfreeze_encoder_epoch: int | None = None
    unfreeze_encoder_lr_multiplier: float = 0.1
    freeze_decoder_initially: bool = False
    unfreeze_decoder_epoch: int | None = None


@dataclass
class ChemeleonModelParams:
    """Chemeleon-specific model parameters.

    Parameters:
        checkpoint_path: Path to pretrained checkpoint or "auto" for download.
        unfreeze_schedule: Gradual unfreezing configuration.
        ffn_type: FFN architecture type ('regression', 'mixture_of_experts', 'branched').
        ffn_hidden_dim: Hidden dimension for FFN layers.
        ffn_num_layers: Number of FFN layers.
        dropout: Dropout probability.
        batch_norm: Whether to use batch normalization.
        n_experts: Number of experts for MoE architecture (only used when ffn_type='mixture_of_experts').
        trunk_n_layers: Number of trunk layers for branched architecture (only used when ffn_type='branched').
        trunk_hidden_dim: Hidden dimension for trunk in branched architecture.
    """

    checkpoint_path: str = "auto"
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    ffn_type: str = "regression"
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    batch_norm: bool = False
    n_experts: int | None = None
    trunk_n_layers: int | None = None
    trunk_hidden_dim: int | None = None


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
        seed: Random seed.
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
    seed: int = 42


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
        seed: Random seed.
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
    seed: int = 42


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
        random_seed: Random seed.
        verbose: Logging verbosity.
    """

    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    subsample: float | None = None
    rsm: float | None = None
    random_seed: int = 42
    verbose: bool = False
