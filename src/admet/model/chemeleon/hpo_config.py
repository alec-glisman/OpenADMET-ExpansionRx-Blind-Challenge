"""HPO Configuration module for CheMeleon hyperparameter optimization.

This module provides OmegaConf-compatible dataclasses for configuring
Ray Tune hyperparameter optimization of CheMeleon models.

CheMeleon HPO differs from Chemprop HPO in that:
- The message passing encoder is frozen (no MPNN hyperparameters to tune)
- Focus is on FFN architecture and training dynamics
- Gradual unfreezing schedule can optionally be tuned
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class ParameterSpace:
    """Configuration for a single hyperparameter search space.

    Attributes:
        type: Distribution type - one of "uniform", "loguniform", "choice",
            "quniform", "randint", "qrandint"
        low: Lower bound for uniform/loguniform/quniform/randint/qrandint
        high: Upper bound for uniform/loguniform/quniform/randint/qrandint
        values: List of values for choice distribution
        q: Quantization step for quniform/qrandint distribution
        conditional_on: Parameter name this depends on (for conditional spaces)
        conditional_values: Parameter values that enable this space
    """

    type: str = MISSING
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None
    q: float | None = None
    conditional_on: str | None = None
    conditional_values: list[Any] | None = None


@dataclass
class ChemeleonSearchSpaceConfig:
    """Configuration for the CheMeleon hyperparameter search space.

    Defines which parameters to tune and their search ranges.
    All parameters are optional - only specified parameters will be tuned.

    Note: Message passing parameters are NOT tunable since the encoder is frozen.

    Attributes:
        learning_rate: Max learning rate search space
        lr_warmup_ratio: Ratio of init_lr to max_lr
        lr_final_ratio: Ratio of final_lr to max_lr
        warmup_epochs: Number of warmup epochs search space
        patience: Early stopping patience search space
        dropout: Dropout rate search space
        ffn_type: FFN architecture type ('regression', 'mixture_of_experts', 'branched')
        ffn_num_layers: FFN layers search space
        ffn_hidden_dim: FFN hidden dimension search space
        batch_size: Batch size search space
        n_experts: Number of experts (MoE only, conditional)
        trunk_n_layers: Trunk depth (branched only, conditional)
        trunk_hidden_dim: Trunk hidden dimension (branched only, conditional)
        batch_norm: Whether to use batch normalization
    """

    # Learning rate schedule
    learning_rate: ParameterSpace | None = None
    lr_warmup_ratio: ParameterSpace | None = None
    lr_final_ratio: ParameterSpace | None = None
    warmup_epochs: ParameterSpace | None = None
    patience: ParameterSpace | None = None

    # Regularization
    dropout: ParameterSpace | None = None

    # FFN architecture
    ffn_type: ParameterSpace | None = None
    ffn_num_layers: ParameterSpace | None = None
    ffn_hidden_dim: ParameterSpace | None = None
    batch_size: ParameterSpace | None = None
    batch_norm: ParameterSpace | None = None

    # MoE-specific (conditional on ffn_type == 'mixture_of_experts')
    n_experts: ParameterSpace | None = None

    # Branched-specific (conditional on ffn_type == 'branched')
    trunk_n_layers: ParameterSpace | None = None
    trunk_hidden_dim: ParameterSpace | None = None


@dataclass
class ASHAConfig:
    """Configuration for the ASHA scheduler.

    Attributes:
        metric: Metric to optimize (e.g., "val_mae")
        mode: Optimization mode - "min" or "max"
        max_t: Maximum training epochs
        grace_period: Minimum epochs before early stopping
        reduction_factor: Factor for successive halving
        brackets: Number of brackets for ASHA
    """

    metric: str = "val_mae"
    mode: str = "min"
    max_t: int = 100
    grace_period: int = 15
    reduction_factor: int = 3
    brackets: int = 1


@dataclass
class ResourceConfig:
    """Configuration for Ray Tune resource allocation.

    Attributes:
        num_samples: Number of HPO trials to run
        cpus_per_trial: CPU cores per trial
        gpus_per_trial: GPU fraction per trial
        max_concurrent_trials: Maximum concurrent trials
    """

    num_samples: int = 50
    cpus_per_trial: int = 4
    gpus_per_trial: float = 0.25
    max_concurrent_trials: int | None = None


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning from HPO results.

    Attributes:
        top_k: Number of top configurations for ensemble training
        full_epochs: Number of epochs for full ensemble training
        ensemble_size: Number of models per ensemble configuration
    """

    top_k: int = 5
    full_epochs: int = 100
    ensemble_size: int = 5


@dataclass
class ChemeleonHPOConfig:
    """Main HPO configuration for CheMeleon model.

    Attributes:
        experiment_name: Name for the HPO experiment
        data_path: Path to training data CSV
        val_data_path: Path to validation data CSV
        smiles_column: Column name for SMILES strings
        target_columns: List of target column names
        output_dir: Directory for HPO outputs
        checkpoint_path: Path to CheMeleon checkpoint or "auto"
        freeze_encoder: Whether to freeze encoder during training
        search_space: Hyperparameter search space configuration
        asha: ASHA scheduler configuration
        resources: Resource allocation configuration
        transfer_learning: Transfer learning configuration
        seed: Random seed for reproducibility
        ray_storage_path: Path for Ray Tune storage
        mlflow_tracking_uri: MLflow tracking URI
    """

    experiment_name: str = MISSING
    data_path: str = MISSING
    smiles_column: str = "smiles"
    target_columns: list[str] = field(default_factory=list)
    output_dir: str = "outputs/hpo_chemeleon"

    # Optional paths
    val_data_path: str | None = None
    ray_storage_path: str | None = None
    mlflow_tracking_uri: str | None = None

    # CheMeleon-specific
    checkpoint_path: str = "auto"
    freeze_encoder: bool = True

    # Sub-configurations
    search_space: ChemeleonSearchSpaceConfig = field(default_factory=ChemeleonSearchSpaceConfig)
    asha: ASHAConfig = field(default_factory=ASHAConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    transfer_learning: TransferLearningConfig = field(default_factory=TransferLearningConfig)

    # Reproducibility
    seed: int = 42


# Default search space for CheMeleon HPO
DEFAULT_CHEMELEON_SEARCH_SPACE = ChemeleonSearchSpaceConfig(
    learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-3),
    dropout=ParameterSpace(type="uniform", low=0.0, high=0.3),
    ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts", "branched"]),
    ffn_num_layers=ParameterSpace(type="randint", low=1, high=4),
    ffn_hidden_dim=ParameterSpace(type="choice", values=[128, 256, 300, 512]),
    batch_size=ParameterSpace(type="choice", values=[16, 32, 64]),
    n_experts=ParameterSpace(
        type="randint",
        low=2,
        high=8,
        conditional_on="ffn_type",
        conditional_values=["mixture_of_experts"],
    ),
    trunk_n_layers=ParameterSpace(
        type="randint",
        low=1,
        high=3,
        conditional_on="ffn_type",
        conditional_values=["branched"],
    ),
    trunk_hidden_dim=ParameterSpace(
        type="choice",
        values=[128, 256, 384, 512],
        conditional_on="ffn_type",
        conditional_values=["branched"],
    ),
)
