"""HPO Configuration module for Chemprop hyperparameter optimization.

This module provides OmegaConf-compatible dataclasses for configuring
Ray Tune hyperparameter optimization of Chemprop models.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class ParameterSpace:
    """Configuration for a single hyperparameter search space.

    Attributes:
        type: Distribution type - one of "uniform", "loguniform", "choice", "quniform"
        low: Lower bound for uniform/loguniform/quniform distributions
        high: Upper bound for uniform/loguniform/quniform distributions
        values: List of values for choice distribution
        q: Quantization step for quniform distribution
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
class SearchSpaceConfig:
    """Configuration for the hyperparameter search space.

    Defines which parameters to tune and their search ranges.
    All parameters are optional - only specified parameters will be tuned.

    Attributes:
        learning_rate: Learning rate search space
        weight_decay: Weight decay search space
        dropout: Dropout rate search space
        depth: Message passing depth search space
        hidden_dim: Hidden dimension search space
        ffn_num_layers: FFN layers search space
        ffn_hidden_dim: FFN hidden dimension search space
        batch_size: Batch size search space
        ffn_type: FFN architecture type search space
        n_experts: Number of experts (MoE only) search space
        trunk_depth: Trunk depth (branched only) search space
        trunk_hidden_dim: Trunk hidden dimension (branched only) search space
        aggregation: Message aggregation function search space
        aggregation_norm: Aggregation normalization search space
        target_weights: Per-endpoint loss weights search space (applied to each target)
    """

    learning_rate: ParameterSpace | None = None
    weight_decay: ParameterSpace | None = None
    dropout: ParameterSpace | None = None
    depth: ParameterSpace | None = None
    hidden_dim: ParameterSpace | None = None
    ffn_num_layers: ParameterSpace | None = None
    ffn_hidden_dim: ParameterSpace | None = None
    batch_size: ParameterSpace | None = None
    ffn_type: ParameterSpace | None = None
    n_experts: ParameterSpace | None = None
    trunk_depth: ParameterSpace | None = None
    trunk_hidden_dim: ParameterSpace | None = None
    aggregation: ParameterSpace | None = None
    aggregation_norm: ParameterSpace | None = None
    target_weights: ParameterSpace | None = None


@dataclass
class ASHAConfig:
    """Configuration for the ASHA (Asynchronous Successive Halving) scheduler.

    Attributes:
        metric: Metric to optimize (e.g., "val_mae")
        mode: Optimization mode - "min" or "max"
        max_t: Maximum training epochs
        grace_period: Minimum epochs before early stopping
        reduction_factor: Factor for successive halving (typically 2 or 3)
        brackets: Number of brackets for ASHA
    """

    metric: str = "val_mae"
    mode: str = "min"
    max_t: int = 150
    grace_period: int = 10
    reduction_factor: int = 3
    brackets: int = 1


@dataclass
class ResourceConfig:
    """Configuration for Ray Tune resource allocation.

    Attributes:
        num_samples: Number of HPO trials to run
        cpus_per_trial: CPU cores per trial
        gpus_per_trial: GPU fraction per trial (0.25 = 4 trials per GPU)
        max_concurrent_trials: Maximum concurrent trials (None = auto)
    """

    num_samples: int = 50
    cpus_per_trial: int = 2
    gpus_per_trial: float = 0.25
    max_concurrent_trials: int | None = None


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning from HPO results.

    After HPO completes, the top-k configurations can be used to train
    full ensembles with the best hyperparameters.

    Attributes:
        top_k: Number of top configurations to use for ensemble training
        full_epochs: Number of epochs for full ensemble training
        ensemble_size: Number of models per ensemble configuration
    """

    top_k: int = 5
    full_epochs: int = 150
    ensemble_size: int = 5


@dataclass
class HPOConfig:
    """Main HPO configuration combining all sub-configurations.

    Attributes:
        experiment_name: Name for the HPO experiment (used in MLflow/Ray)
        data_path: Path to training data CSV
        val_data_path: Path to validation data CSV (optional)
        smiles_column: Column name for SMILES strings
        target_columns: List of target column names
        output_dir: Directory for HPO outputs
        search_space: Hyperparameter search space configuration
        asha: ASHA scheduler configuration
        resources: Resource allocation configuration
        transfer_learning: Transfer learning configuration
        base_config_path: Path to base Chemprop config YAML (optional)
        seed: Random seed for reproducibility
        ray_storage_path: Path for Ray Tune storage (optional)
    """

    experiment_name: str = MISSING
    data_path: str = MISSING
    smiles_column: str = "smiles"
    target_columns: list[str] = field(default_factory=list)
    output_dir: str = "outputs/hpo"

    # Optional paths
    val_data_path: str | None = None
    base_config_path: str | None = None
    ray_storage_path: str | None = None

    # Sub-configurations
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    asha: ASHAConfig = field(default_factory=ASHAConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    transfer_learning: TransferLearningConfig = field(default_factory=TransferLearningConfig)

    # Reproducibility
    seed: int = 42
