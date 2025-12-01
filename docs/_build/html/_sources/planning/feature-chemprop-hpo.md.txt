---
goal: Implement Ray Tune Hyperparameter Optimization for Chemprop Models
version: 1.0
date_created: 2025-11-30
last_updated: 2025-11-30
owner: alec-glisman
status: Planned
tags:
  - feature
  - hyperparameter-optimization
  - ray-tune
  - chemprop
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This implementation plan describes adding hyperparameter optimization (HPO) capabilities to the Chemprop model training pipeline using Ray Tune with ASHA scheduler. The HPO system will optimize `ChempropHyperparams` values using validation loss as the primary metric, support conditional search spaces for architecture-specific parameters, and enable transfer learning by training full ensembles with top-k configurations.

## 1. Requirements & Constraints

### Functional Requirements

- **REQ-001**: HPO pipeline must be fully configurable via YAML files processed by OmegaConf
- **REQ-002**: Use validation loss (`val_loss`) as the primary optimization metric (minimize)
- **REQ-003**: Support conditional search spaces for FFN architecture-specific parameters
- **REQ-004**: Use ASHA scheduler for efficient early stopping of poor trials
- **REQ-005**: Support 4 concurrent trials per GPU with fractional GPU allocation
- **REQ-006**: Seeds must be configurable and reproducible across trials
- **REQ-007**: Use single train/validation split during HPO for efficiency
- **REQ-008**: Select top-k configurations and train full CV ensembles for each
- **REQ-009**: Log all HPO trials and metrics to MLflow for analysis
- **REQ-010**: Provide CLI entry point and bash script for running HPO

### Security Requirements

- **SEC-001**: No sensitive data in configuration files or logs

### Constraints

- **CON-001**: Must integrate with existing `ChempropModel` and `ChempropHyperparams` classes
- **CON-002**: Must not break existing single model and ensemble training workflows
- **CON-003**: GPU memory constraint: 4 trials sharing 1 GPU (0.25 GPU fraction per trial)
- **CON-004**: Must use Ray Tune >= 2.9.0 for stable ASHA implementation

### Guidelines

- **GUD-001**: Follow existing code patterns from `ensemble.py` for Ray integration
- **GUD-002**: Use dataclasses with OmegaConf for all configuration structures
- **GUD-003**: Implement comprehensive logging at DEBUG, INFO, WARNING levels
- **GUD-004**: Include docstrings following numpydoc format for all public APIs

### Patterns to Follow

- **PAT-001**: Configuration pattern from `config.py` (dataclass + OmegaConf)
- **PAT-002**: Factory pattern from `ChempropModel.from_config()`
- **PAT-003**: MLflow integration pattern from `ensemble.py`
- **PAT-004**: CLI pattern from `model.py` main() function

## 2. Implementation Steps

### Implementation Phase 1: Configuration Infrastructure

- GOAL-001: Create HPO configuration dataclasses and YAML schema

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `src/admet/model/chemprop/hpo_config.py` with `HPOConfig` dataclass | | |
| TASK-002 | Define `SearchSpaceConfig` dataclass for individual parameter search spaces | | |
| TASK-003 | Define `ASHAConfig` dataclass for scheduler configuration | | |
| TASK-004 | Define `ResourceConfig` dataclass for GPU/CPU allocation | | |
| TASK-005 | Define `TransferLearningConfig` dataclass for top-k ensemble training | | |
| TASK-006 | Create `configs/hpo_chemprop.yaml` example configuration file | | |
| TASK-007 | Add unit tests for configuration loading and validation | | |

**TASK-001 Details: HPOConfig Dataclass**

```python
@dataclass
class HPOConfig:
    """Root configuration for hyperparameter optimization."""
    
    # Base model configuration (fixed parameters)
    data: DataConfig
    model: ModelConfig  # Default values, overridden by search space
    optimization: OptimizationConfig  # Default values, overridden by search space
    mlflow: MlflowConfig
    
    # HPO-specific settings
    search_space: SearchSpaceConfig
    scheduler: ASHAConfig
    resources: ResourceConfig
    transfer_learning: TransferLearningConfig
    
    # General HPO settings
    num_samples: int = 100  # Total number of trials
    seed: int = 12345  # Global seed for reproducibility
    metric: str = "val_loss"  # Metric to optimize
    mode: str = "min"  # "min" or "max"
```

**TASK-002 Details: SearchSpaceConfig Dataclass**

```python
@dataclass
class ParameterSpace:
    """Configuration for a single parameter's search space."""
    type: str  # "uniform", "loguniform", "choice", "randint", "fixed"
    low: Optional[float] = None  # For uniform, loguniform, randint
    high: Optional[float] = None  # For uniform, loguniform, randint
    choices: Optional[List[Any]] = None  # For choice
    value: Optional[Any] = None  # For fixed
    condition: Optional[str] = None  # Conditional expression, e.g., "ffn_type == 'branched'"

@dataclass
class SearchSpaceConfig:
    """Search space definitions for hyperparameters."""
    
    # Learning rate schedule
    init_lr: Optional[ParameterSpace] = None
    max_lr: Optional[ParameterSpace] = None
    final_lr: Optional[ParameterSpace] = None
    
    # Training dynamics
    warmup_epochs: Optional[ParameterSpace] = None
    batch_size: Optional[ParameterSpace] = None
    
    # Message passing architecture
    depth: Optional[ParameterSpace] = None
    message_hidden_dim: Optional[ParameterSpace] = None
    
    # FFN architecture
    dropout: Optional[ParameterSpace] = None
    num_layers: Optional[ParameterSpace] = None
    hidden_dim: Optional[ParameterSpace] = None
    ffn_type: Optional[ParameterSpace] = None
    
    # Conditional: only when ffn_type == "branched"
    trunk_n_layers: Optional[ParameterSpace] = None
    trunk_hidden_dim: Optional[ParameterSpace] = None
    
    # Conditional: only when ffn_type == "mixture_of_experts"
    n_experts: Optional[ParameterSpace] = None
    
    # Loss function
    criterion: Optional[ParameterSpace] = None
    
    # MPNN settings
    batch_norm: Optional[ParameterSpace] = None
```

**TASK-003 Details: ASHAConfig Dataclass**

```python
@dataclass
class ASHAConfig:
    """Configuration for ASHA scheduler."""
    max_t: int = 150  # Maximum epochs (same as max_epochs)
    grace_period: int = 10  # Minimum epochs before pruning
    reduction_factor: int = 3  # Halving rate for trials
    brackets: int = 1  # Number of brackets (1 = standard ASHA)
```

**TASK-004 Details: ResourceConfig Dataclass**

```python
@dataclass
class ResourceConfig:
    """Resource allocation for HPO trials."""
    gpus_per_trial: float = 0.25  # Fractional GPU (4 trials per GPU)
    cpus_per_trial: int = 2  # CPUs per trial
    max_concurrent_trials: int = 4  # Maximum parallel trials
```

**TASK-005 Details: TransferLearningConfig Dataclass**

```python
@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning with top-k configs."""
    enabled: bool = True
    top_k: int = 3  # Number of best configurations to use
    train_full_cv: bool = True  # Train full cross-validation ensembles
    ensemble_splits: Optional[List[int]] = None  # Splits for ensemble (null = all)
    ensemble_folds: Optional[List[int]] = None  # Folds for ensemble (null = all)
```

**TASK-006 Details: Example YAML Configuration**

Create `configs/hpo_chemprop.yaml`:

```yaml
# Hyperparameter Optimization Configuration for Chemprop
# ======================================================
# This YAML configures Ray Tune HPO with ASHA scheduler.
#
# Usage:
#   python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml

# Data configuration (fixed during HPO)
data:
  # Single split/fold for HPO efficiency
  data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0"
  test_file: "assets/dataset/set/local_test.csv"
  blind_file: null  # Not used during HPO
  smiles_col: "SMILES"
  target_cols:
    - "LogD"
    - "Log KSOL"
    - "Log HLM CLint"
    - "Log MLM CLint"
    - "Log Caco-2 Permeability Papp A>B"
    - "Log Caco-2 Permeability Efflux"
    - "Log MPPB"
    - "Log MBPB"
    - "Log MGMB"
  target_weights:
    - 0.5
    - 2.0
    - 3.0
    - 3.0
    - 3.0
    - 2.0
    - 2.0
    - 3.0
    - 4.0
  output_dir: null

# Default model configuration (overridden by search space)
model:
  depth: 5
  message_hidden_dim: 600
  num_layers: 2
  hidden_dim: 600
  dropout: 0.1
  batch_norm: true
  ffn_type: "regression"
  trunk_n_layers: 2
  trunk_hidden_dim: 600
  n_experts: 4

# Default optimization configuration (overridden by search space)
optimization:
  criterion: "MSE"
  init_lr: 1.0e-4
  max_lr: 1.0e-3
  final_lr: 1.0e-4
  warmup_epochs: 5
  patience: 15
  max_epochs: 150
  batch_size: 32
  num_workers: 0
  seed: 12345  # Base seed, trials get seed + trial_id
  progress_bar: false

# MLflow tracking
mlflow:
  tracking: true
  tracking_uri: "http://127.0.0.1:8080"
  experiment_name: "chemprop_hpo"
  run_name: null  # Auto-generated

# Search space definitions
search_space:
  # Learning rate (log-uniform scale)
  init_lr:
    type: "loguniform"
    low: 1.0e-5
    high: 1.0e-3
  max_lr:
    type: "loguniform"
    low: 1.0e-4
    high: 1.0e-2
  final_lr:
    type: "loguniform"
    low: 1.0e-6
    high: 1.0e-4
  
  # Training dynamics
  warmup_epochs:
    type: "randint"
    low: 1
    high: 10
  batch_size:
    type: "choice"
    choices: [16, 32, 64, 128]
  
  # Message passing
  depth:
    type: "randint"
    low: 2
    high: 6
  message_hidden_dim:
    type: "choice"
    choices: [256, 300, 512, 600, 768]
  
  # FFN architecture
  dropout:
    type: "uniform"
    low: 0.0
    high: 0.4
  num_layers:
    type: "randint"
    low: 1
    high: 4
  hidden_dim:
    type: "choice"
    choices: [256, 300, 512, 600, 768]
  ffn_type:
    type: "choice"
    choices: ["regression", "mixture_of_experts", "branched"]
  
  # Conditional: branched FFN
  trunk_n_layers:
    type: "randint"
    low: 1
    high: 3
    condition: "ffn_type == 'branched'"
  trunk_hidden_dim:
    type: "choice"
    choices: [256, 300, 512, 600]
    condition: "ffn_type == 'branched'"
  
  # Conditional: mixture of experts
  n_experts:
    type: "randint"
    low: 2
    high: 8
    condition: "ffn_type == 'mixture_of_experts'"
  
  # Loss function
  criterion:
    type: "choice"
    choices: ["MSE", "MAE", "RMSE"]
  
  # MPNN
  batch_norm:
    type: "choice"
    choices: [true, false]

# ASHA scheduler configuration
scheduler:
  max_t: 150  # Max epochs
  grace_period: 10  # Min epochs before pruning
  reduction_factor: 3
  brackets: 1

# Resource allocation
resources:
  gpus_per_trial: 0.25  # 4 trials per GPU
  cpus_per_trial: 2
  max_concurrent_trials: 4

# Transfer learning settings
transfer_learning:
  enabled: true
  top_k: 3  # Train ensembles with top 3 configurations
  train_full_cv: true
  # Use all splits/folds for ensemble training
  ensemble_splits: null
  ensemble_folds: null

# General HPO settings
num_samples: 100
seed: 12345
metric: "val_loss"
mode: "min"
```

### Implementation Phase 2: Search Space Builder

- GOAL-002: Implement search space construction from configuration

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Create `src/admet/model/chemprop/hpo_search_space.py` | | |
| TASK-009 | Implement `build_search_space()` function to convert config to Ray Tune format | | |
| TASK-010 | Implement conditional search space logic using `tune.sample_from()` | | |
| TASK-011 | Add validation for search space configuration | | |
| TASK-012 | Add unit tests for search space building | | |

**TASK-008-010 Details: Search Space Builder Module**

```python
# src/admet/model/chemprop/hpo_search_space.py
"""
Search space construction for Chemprop HPO.

This module converts SearchSpaceConfig dataclass to Ray Tune search space
format, including support for conditional parameters.
"""

from typing import Any, Dict, Optional
from ray import tune

from admet.model.chemprop.hpo_config import SearchSpaceConfig, ParameterSpace


def _build_parameter_space(param: ParameterSpace) -> Any:
    """Convert a ParameterSpace to Ray Tune search space."""
    if param.type == "uniform":
        return tune.uniform(param.low, param.high)
    elif param.type == "loguniform":
        return tune.loguniform(param.low, param.high)
    elif param.type == "choice":
        return tune.choice(param.choices)
    elif param.type == "randint":
        return tune.randint(param.low, param.high + 1)  # +1 for inclusive
    elif param.type == "fixed":
        return param.value
    else:
        raise ValueError(f"Unknown search space type: {param.type}")


def build_search_space(
    config: SearchSpaceConfig,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Ray Tune search space from configuration.
    
    Parameters
    ----------
    config : SearchSpaceConfig
        Search space configuration from YAML.
    defaults : Dict[str, Any]
        Default values for parameters not in search space.
    
    Returns
    -------
    Dict[str, Any]
        Ray Tune compatible search space dictionary.
    """
    search_space = {}
    
    # Non-conditional parameters
    param_names = [
        "init_lr", "max_lr", "final_lr", "warmup_epochs", "batch_size",
        "depth", "message_hidden_dim", "dropout", "num_layers", "hidden_dim",
        "ffn_type", "criterion", "batch_norm"
    ]
    
    for name in param_names:
        param = getattr(config, name, None)
        if param is not None and param.condition is None:
            search_space[name] = _build_parameter_space(param)
        elif name in defaults:
            search_space[name] = defaults[name]
    
    # Conditional parameters using tune.sample_from
    # trunk_n_layers: only when ffn_type == "branched"
    trunk_n_layers_param = getattr(config, "trunk_n_layers", None)
    if trunk_n_layers_param is not None:
        search_space["trunk_n_layers"] = tune.sample_from(
            lambda spec: (
                tune.randint(trunk_n_layers_param.low, trunk_n_layers_param.high + 1).sample()
                if spec.config.get("ffn_type") == "branched"
                else defaults.get("trunk_n_layers", 2)
            )
        )
    
    # trunk_hidden_dim: only when ffn_type == "branched"
    trunk_hidden_dim_param = getattr(config, "trunk_hidden_dim", None)
    if trunk_hidden_dim_param is not None:
        search_space["trunk_hidden_dim"] = tune.sample_from(
            lambda spec: (
                tune.choice(trunk_hidden_dim_param.choices).sample()
                if spec.config.get("ffn_type") == "branched"
                else defaults.get("trunk_hidden_dim", 600)
            )
        )
    
    # n_experts: only when ffn_type == "mixture_of_experts"
    n_experts_param = getattr(config, "n_experts", None)
    if n_experts_param is not None:
        search_space["n_experts"] = tune.sample_from(
            lambda spec: (
                tune.randint(n_experts_param.low, n_experts_param.high + 1).sample()
                if spec.config.get("ffn_type") == "mixture_of_experts"
                else defaults.get("n_experts", 4)
            )
        )
    
    return search_space
```

### Implementation Phase 3: Trainable Function and Callbacks

- GOAL-003: Implement Ray Tune trainable function with metric reporting

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-013 | Create `src/admet/model/chemprop/hpo_trainable.py` | | |
| TASK-014 | Implement `RayTuneReportCallback` for PyTorch Lightning | | |
| TASK-015 | Implement `train_chemprop_trial()` trainable function | | |
| TASK-016 | Add checkpoint saving/loading for trial recovery | | |
| TASK-017 | Add unit tests for trainable function | | |

**TASK-013-016 Details: Trainable Module**

```python
# src/admet/model/chemprop/hpo_trainable.py
"""
Ray Tune trainable function for Chemprop HPO.

This module provides the trainable function that Ray Tune calls for each
trial, including a Lightning callback for metric reporting.
"""

from typing import Any, Dict, Optional
import tempfile
from pathlib import Path

import ray
from ray import tune
from ray.train import Checkpoint
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from admet.model.chemprop.model import ChempropModel, ChempropHyperparams


class RayTuneReportCallback(Callback):
    """
    PyTorch Lightning callback that reports metrics to Ray Tune.
    
    Reports validation loss after each epoch to enable ASHA pruning.
    """
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics to Ray Tune after validation."""
        metrics = {}
        
        # Get validation loss
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            if hasattr(val_loss, "item"):
                metrics["val_loss"] = val_loss.item()
            else:
                metrics["val_loss"] = float(val_loss)
        
        # Get other validation metrics
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val_"):
                if hasattr(value, "item"):
                    metrics[key] = value.item()
                else:
                    metrics[key] = float(value)
        
        # Report to Ray Tune
        if metrics:
            # Create checkpoint for fault tolerance
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = Path(temp_dir) / "checkpoint.ckpt"
                trainer.save_checkpoint(checkpoint_path)
                checkpoint = Checkpoint.from_directory(temp_dir)
                ray.train.report(metrics, checkpoint=checkpoint)


def train_chemprop_trial(config: Dict[str, Any]) -> None:
    """
    Trainable function for a single HPO trial.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Trial configuration containing:
        - Hyperparameters sampled from search space
        - Fixed parameters (data paths, target cols, etc.)
        - Trial metadata (seed, trial_id)
    """
    import pandas as pd
    from admet.model.chemprop.model import ChempropModel, ChempropHyperparams
    
    # Extract fixed config
    data_config = config["_fixed"]["data"]
    mlflow_config = config["_fixed"]["mlflow"]
    base_seed = config["_fixed"]["base_seed"]
    trial_id = config.get("_trial_id", 0)
    
    # Compute trial-specific seed
    trial_seed = base_seed + trial_id
    
    # Build hyperparams from sampled config
    hyperparams = ChempropHyperparams(
        init_lr=config.get("init_lr", 1e-4),
        max_lr=config.get("max_lr", 1e-3),
        final_lr=config.get("final_lr", 1e-4),
        warmup_epochs=config.get("warmup_epochs", 5),
        patience=config["_fixed"]["patience"],  # Fixed during HPO
        max_epochs=config["_fixed"]["max_epochs"],
        batch_size=config.get("batch_size", 32),
        num_workers=0,  # Avoid DataLoader conflicts in Ray
        seed=trial_seed,
        depth=config.get("depth", 5),
        message_hidden_dim=config.get("message_hidden_dim", 600),
        dropout=config.get("dropout", 0.1),
        num_layers=config.get("num_layers", 2),
        hidden_dim=config.get("hidden_dim", 600),
        criterion=config.get("criterion", "MSE"),
        ffn_type=config.get("ffn_type", "regression"),
        trunk_n_layers=config.get("trunk_n_layers", 2),
        trunk_hidden_dim=config.get("trunk_hidden_dim", 600),
        n_experts=config.get("n_experts", 4),
        batch_norm=config.get("batch_norm", True),
    )
    
    # Load data
    data_dir = Path(data_config["data_dir"])
    df_train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    df_validation = pd.read_csv(data_dir / "validation.csv", low_memory=False)
    
    # Load test set if specified
    df_test = None
    if data_config.get("test_file"):
        df_test = pd.read_csv(data_config["test_file"], low_memory=False)
    
    # Create model with Ray Tune callback
    model = ChempropModel(
        df_train=df_train,
        df_validation=df_validation,
        df_test=df_test,
        smiles_col=data_config["smiles_col"],
        target_cols=list(data_config["target_cols"]),
        target_weights=list(data_config.get("target_weights", [])),
        output_dir=None,  # Use temp directory
        progress_bar=False,
        hyperparams=hyperparams,
        mlflow_tracking=mlflow_config.get("tracking", False),
        mlflow_tracking_uri=mlflow_config.get("tracking_uri"),
        mlflow_experiment_name=mlflow_config.get("experiment_name", "chemprop_hpo"),
        mlflow_run_name=f"trial_{trial_id}",
        mlflow_nested=True,  # Nest under HPO parent run
        mlflow_parent_run_id=config["_fixed"].get("mlflow_parent_run_id"),
    )
    
    # Add Ray Tune callback to trainer
    model.trainer.callbacks.append(RayTuneReportCallback())
    
    # Train model
    model.fit()
    
    # Close MLflow run
    model.close()
```

### Implementation Phase 4: HPO Runner

- GOAL-004: Implement main HPO orchestration with ASHA scheduler

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | Create `src/admet/model/chemprop/hpo.py` main runner module | | |
| TASK-019 | Implement `ChempropHPO` class for orchestrating HPO | | |
| TASK-020 | Implement ASHA scheduler setup with configuration | | |
| TASK-021 | Implement MLflow parent run for HPO tracking | | |
| TASK-022 | Implement results collection and best config export | | |
| TASK-023 | Add CLI entry point with argparse | | |
| TASK-024 | Add integration tests for HPO runner | | |

**TASK-018-023 Details: HPO Runner Module**

```python
# src/admet/model/chemprop/hpo.py
"""
Hyperparameter optimization runner for Chemprop models.

This module provides the main entry point for running HPO with Ray Tune,
using ASHA scheduler for efficient trial pruning.

Usage
-----
python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ResultGrid

from admet.model.chemprop.hpo_config import HPOConfig
from admet.model.chemprop.hpo_search_space import build_search_space
from admet.model.chemprop.hpo_trainable import train_chemprop_trial

logger = logging.getLogger("admet.model.chemprop.hpo")


class ChempropHPO:
    """
    Hyperparameter optimization orchestrator for Chemprop models.
    
    Parameters
    ----------
    config : HPOConfig or DictConfig
        HPO configuration loaded from YAML.
    
    Attributes
    ----------
    config : HPOConfig
        The HPO configuration.
    results : ResultGrid or None
        Ray Tune results after HPO completes.
    best_configs : List[Dict[str, Any]]
        Top-k configurations sorted by validation loss.
    """
    
    def __init__(self, config: HPOConfig | DictConfig) -> None:
        self.config = config
        self.results: Optional[ResultGrid] = None
        self.best_configs: List[Dict[str, Any]] = []
        self._mlflow_client: Optional[MlflowClient] = None
        self._parent_run_id: Optional[str] = None
    
    @classmethod
    def from_config(cls, config: HPOConfig | DictConfig) -> "ChempropHPO":
        """Create HPO runner from configuration."""
        return cls(config)
    
    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking for HPO."""
        if not self.config.mlflow.tracking:
            return
        
        if self.config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        self._mlflow_client = MlflowClient()
        
        # Start parent run for HPO
        parent_run = mlflow.start_run(run_name=self.config.mlflow.run_name or "hpo_run")
        self._parent_run_id = parent_run.info.run_id
        
        # Log HPO configuration
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        mlflow.log_params(_flatten_dict(config_dict, max_depth=2))
        
        logger.info("Started MLflow HPO parent run: %s", self._parent_run_id)
    
    def _build_scheduler(self) -> ASHAScheduler:
        """Build ASHA scheduler from configuration."""
        return ASHAScheduler(
            metric=self.config.metric,
            mode=self.config.mode,
            max_t=self.config.scheduler.max_t,
            grace_period=self.config.scheduler.grace_period,
            reduction_factor=self.config.scheduler.reduction_factor,
            brackets=self.config.scheduler.brackets,
        )
    
    def _build_search_space(self) -> Dict[str, Any]:
        """Build Ray Tune search space from configuration."""
        # Get default values from model/optimization config
        defaults = {
            **asdict(self.config.model) if hasattr(self.config.model, '__dataclass_fields__') else dict(self.config.model),
            **asdict(self.config.optimization) if hasattr(self.config.optimization, '__dataclass_fields__') else dict(self.config.optimization),
        }
        
        search_space = build_search_space(self.config.search_space, defaults)
        
        # Add fixed configuration that all trials need
        search_space["_fixed"] = {
            "data": OmegaConf.to_container(self.config.data, resolve=True),
            "mlflow": OmegaConf.to_container(self.config.mlflow, resolve=True),
            "base_seed": self.config.seed,
            "patience": self.config.optimization.patience,
            "max_epochs": self.config.optimization.max_epochs,
            "mlflow_parent_run_id": self._parent_run_id,
        }
        
        return search_space
    
    def run(self) -> ResultGrid:
        """
        Run hyperparameter optimization.
        
        Returns
        -------
        ResultGrid
            Ray Tune results containing all trial information.
        """
        # Initialize MLflow
        self._init_mlflow()
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Initialized Ray cluster")
        
        # Build search space and scheduler
        search_space = self._build_search_space()
        scheduler = self._build_scheduler()
        
        # Configure resources
        trainable_with_resources = tune.with_resources(
            train_chemprop_trial,
            resources={
                "gpu": self.config.resources.gpus_per_trial,
                "cpu": self.config.resources.cpus_per_trial,
            }
        )
        
        logger.info("Starting HPO with %d samples", self.config.num_samples)
        logger.info("Resources: %.2f GPU, %d CPU per trial", 
                   self.config.resources.gpus_per_trial,
                   self.config.resources.cpus_per_trial)
        
        # Run HPO
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric=self.config.metric,
                mode=self.config.mode,
                scheduler=scheduler,
                num_samples=self.config.num_samples,
                max_concurrent_trials=self.config.resources.max_concurrent_trials,
            ),
            run_config=ray.train.RunConfig(
                name="chemprop_hpo",
                verbose=1,
            ),
        )
        
        self.results = tuner.fit()
        
        # Collect best configurations
        self._collect_best_configs()
        
        # Log results to MLflow
        self._log_results()
        
        return self.results
    
    def _collect_best_configs(self) -> None:
        """Collect top-k configurations from HPO results."""
        if self.results is None:
            return
        
        # Get all results sorted by metric
        results_df = self.results.get_dataframe()
        results_df = results_df.sort_values(self.config.metric, ascending=(self.config.mode == "min"))
        
        # Extract top-k configurations
        top_k = self.config.transfer_learning.top_k
        for i in range(min(top_k, len(results_df))):
            result = self.results[i]
            config = result.config.copy()
            # Remove internal keys
            config.pop("_fixed", None)
            config.pop("_trial_id", None)
            
            self.best_configs.append({
                "rank": i + 1,
                "val_loss": result.metrics.get(self.config.metric),
                "config": config,
            })
        
        logger.info("Collected top %d configurations", len(self.best_configs))
    
    def _log_results(self) -> None:
        """Log HPO results to MLflow."""
        if self._mlflow_client is None or self._parent_run_id is None:
            return
        
        # Log best configurations
        for i, best in enumerate(self.best_configs):
            self._mlflow_client.log_metric(
                self._parent_run_id,
                f"top_{i+1}_val_loss",
                best["val_loss"]
            )
        
        # Save best configs as artifact
        import yaml
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp())
        configs_path = temp_dir / "best_configs.yaml"
        with open(configs_path, "w") as f:
            yaml.dump(self.best_configs, f, default_flow_style=False)
        
        self._mlflow_client.log_artifact(
            self._parent_run_id,
            str(configs_path),
            artifact_path="hpo_results"
        )
        
        logger.info("Logged HPO results to MLflow")
    
    def train_top_k_ensembles(self) -> None:
        """
        Train full CV ensembles with top-k configurations.
        
        This implements transfer learning by taking the best HPO configs
        and training complete ensembles across all splits and folds.
        """
        if not self.config.transfer_learning.enabled:
            logger.info("Transfer learning disabled, skipping ensemble training")
            return
        
        if not self.best_configs:
            logger.warning("No best configs available, run HPO first")
            return
        
        from admet.model.chemprop.ensemble import ChempropEnsemble
        
        logger.info("Training ensembles with top %d configurations", len(self.best_configs))
        
        for i, best in enumerate(self.best_configs):
            logger.info("Training ensemble %d/%d (val_loss=%.4f)", 
                       i + 1, len(self.best_configs), best["val_loss"])
            
            # Build ensemble config from best HPO config
            ensemble_config = self._build_ensemble_config(best["config"], rank=i + 1)
            
            # Train ensemble
            ensemble = ChempropEnsemble.from_config(ensemble_config)
            ensemble.train_all()
            ensemble.close()
            
            logger.info("Completed ensemble %d/%d", i + 1, len(self.best_configs))
    
    def _build_ensemble_config(self, hpo_config: Dict[str, Any], rank: int) -> DictConfig:
        """Build ensemble config from HPO best config."""
        from admet.model.chemprop.config import EnsembleConfig
        
        # Start with base config
        ensemble_config = OmegaConf.structured(EnsembleConfig)
        
        # Copy data config but update data_dir to ensemble root
        ensemble_config.data = self.config.data.copy()
        # Extract parent directory (remove /split_X/fold_Y from path)
        data_dir = Path(self.config.data.data_dir)
        ensemble_data_dir = data_dir.parent.parent  # Go up from fold_X/split_Y
        ensemble_config.data.data_dir = str(ensemble_data_dir)
        
        # Apply splits/folds filter if specified
        ensemble_config.data.splits = self.config.transfer_learning.ensemble_splits
        ensemble_config.data.folds = self.config.transfer_learning.ensemble_folds
        
        # Apply best hyperparameters to model config
        ensemble_config.model.depth = hpo_config.get("depth", self.config.model.depth)
        ensemble_config.model.message_hidden_dim = hpo_config.get("message_hidden_dim", self.config.model.message_hidden_dim)
        ensemble_config.model.num_layers = hpo_config.get("num_layers", self.config.model.num_layers)
        ensemble_config.model.hidden_dim = hpo_config.get("hidden_dim", self.config.model.hidden_dim)
        ensemble_config.model.dropout = hpo_config.get("dropout", self.config.model.dropout)
        ensemble_config.model.batch_norm = hpo_config.get("batch_norm", self.config.model.batch_norm)
        ensemble_config.model.ffn_type = hpo_config.get("ffn_type", self.config.model.ffn_type)
        ensemble_config.model.trunk_n_layers = hpo_config.get("trunk_n_layers", self.config.model.trunk_n_layers)
        ensemble_config.model.trunk_hidden_dim = hpo_config.get("trunk_hidden_dim", self.config.model.trunk_hidden_dim)
        ensemble_config.model.n_experts = hpo_config.get("n_experts", self.config.model.n_experts)
        
        # Apply best hyperparameters to optimization config
        ensemble_config.optimization.criterion = hpo_config.get("criterion", self.config.optimization.criterion)
        ensemble_config.optimization.init_lr = hpo_config.get("init_lr", self.config.optimization.init_lr)
        ensemble_config.optimization.max_lr = hpo_config.get("max_lr", self.config.optimization.max_lr)
        ensemble_config.optimization.final_lr = hpo_config.get("final_lr", self.config.optimization.final_lr)
        ensemble_config.optimization.warmup_epochs = hpo_config.get("warmup_epochs", self.config.optimization.warmup_epochs)
        ensemble_config.optimization.batch_size = hpo_config.get("batch_size", self.config.optimization.batch_size)
        
        # MLflow config for ensemble
        ensemble_config.mlflow = self.config.mlflow.copy()
        ensemble_config.mlflow.run_name = f"ensemble_top_{rank}"
        
        return ensemble_config
    
    def close(self) -> None:
        """Clean up resources."""
        if self._parent_run_id:
            mlflow.end_run()
            logger.info("Ended MLflow HPO run")
        
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shut down Ray cluster")


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".", max_depth: int = 3) -> Dict[str, Any]:
    """Flatten nested dictionary for MLflow params."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and max_depth > 0:
            items.extend(_flatten_dict(v, new_key, sep, max_depth - 1).items())
        else:
            str_val = str(v)
            if len(str_val) > 250:
                str_val = str_val[:247] + "..."
            items.append((new_key, str_val))
    return dict(items)


def run_hpo_from_config(config_path: str, log_level: str = "INFO") -> None:
    """
    Run HPO from a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to HPO YAML configuration file.
    log_level : str, default="INFO"
        Logging level.
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Loading HPO configuration from: %s", config_path)
    
    # Load configuration
    config = OmegaConf.merge(
        OmegaConf.structured(HPOConfig),
        OmegaConf.load(config_path),
    )
    
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))
    
    # Create and run HPO
    hpo = ChempropHPO.from_config(config)
    hpo.run()
    
    # Train ensembles with best configs
    if config.transfer_learning.enabled:
        hpo.train_top_k_ensembles()
    
    hpo.close()
    logger.info("HPO complete!")


def main() -> None:
    """CLI entrypoint for HPO."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for Chemprop models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml
  python -m admet.model.chemprop.hpo -c configs/hpo.yaml --log-level DEBUG
        """,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to HPO YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    run_hpo_from_config(args.config, args.log_level)


if __name__ == "__main__":
    main()
```

### Implementation Phase 5: Bash Script and Documentation

- GOAL-005: Create runner script and update documentation

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-025 | Create `scripts/run_chemprop_hpo.sh` bash script | | |
| TASK-026 | Update `scripts/README.md` with HPO documentation | | |
| TASK-027 | Update `scripts/lib/common.sh` if needed for HPO | | |
| TASK-028 | Add HPO section to main README.md | | |

**TASK-025 Details: HPO Runner Script**

```bash
#!/usr/bin/env bash
# =============================================================================
# Chemprop Hyperparameter Optimization Script
# =============================================================================
# This script runs Ray Tune HPO for Chemprop models and optionally trains
# full CV ensembles with the best configurations.
#
# Usage:
#   ./scripts/run_chemprop_hpo.sh [--dry-run] [--config PATH] [--log-level LEVEL]
#
# Options:
#   --dry-run     Print commands without executing
#   --config      Path to HPO config (default: configs/hpo_chemprop.yaml)
#   --log-level   Set logging level (DEBUG, INFO, WARNING, ERROR)
#
# =============================================================================

set -euo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# Configuration
PROJECT_ROOT="$(get_project_root "$SCRIPT_DIR")"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/hpo_chemprop.yaml"

# Default options
DRY_RUN=false
CONFIG_FILE="$DEFAULT_CONFIG"
LOG_LEVEL="INFO"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--config PATH] [--log-level LEVEL]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check config file exists
check_config_exists "$CONFIG_FILE" || exit 1

log_info "Starting Chemprop HPO"
log_info "Config file: $CONFIG_FILE"

# Build command
CMD="python -m admet.model.chemprop.hpo --config $CONFIG_FILE --log-level $LOG_LEVEL"

# Execute command
if execute_command "$CMD" "$DRY_RUN" "HPO run"; then
    log_success "HPO completed successfully"
else
    log_error "HPO failed"
    exit 1
fi
```

### Implementation Phase 6: Testing

- GOAL-006: Implement comprehensive tests for HPO system

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-029 | Create `tests/admet/model/chemprop/test_hpo_config.py` | | |
| TASK-030 | Create `tests/admet/model/chemprop/test_hpo_search_space.py` | | |
| TASK-031 | Create `tests/admet/model/chemprop/test_hpo_trainable.py` | | |
| TASK-032 | Create `tests/integration/test_hpo_integration.py` | | |
| TASK-033 | Add smoke test for full HPO pipeline (5 trials, 5 epochs) | | |

## 3. Alternatives

- **ALT-001**: Use Optuna directly instead of Ray Tune - Rejected because Ray Tune provides better GPU sharing and integrates with existing ensemble code
- **ALT-002**: Use HyperBand instead of ASHA - ASHA is simpler and performs similarly for our use case
- **ALT-003**: Use Population-Based Training (PBT) - Too complex for initial implementation, can be added later
- **ALT-004**: Optimize per-target weights alongside hyperparameters - Deferred to future work to keep scope manageable

## 4. Dependencies

- **DEP-001**: `ray[tune]>=2.9.0` - Ray Tune framework
- **DEP-002**: `optuna>=3.0.0` - Optional, for Bayesian search algorithm
- **DEP-003**: Existing `ChempropModel` and `ChempropHyperparams` classes
- **DEP-004**: Existing `ChempropEnsemble` for transfer learning
- **DEP-005**: MLflow for experiment tracking
- **DEP-006**: OmegaConf for configuration management

## 5. Files

- **FILE-001**: `src/admet/model/chemprop/hpo_config.py` - HPO configuration dataclasses
- **FILE-002**: `src/admet/model/chemprop/hpo_search_space.py` - Search space builder
- **FILE-003**: `src/admet/model/chemprop/hpo_trainable.py` - Ray Tune trainable function
- **FILE-004**: `src/admet/model/chemprop/hpo.py` - Main HPO runner
- **FILE-005**: `configs/hpo_chemprop.yaml` - Example HPO configuration
- **FILE-006**: `scripts/run_chemprop_hpo.sh` - Bash runner script
- **FILE-007**: `scripts/README.md` - Updated with HPO documentation
- **FILE-008**: `tests/admet/model/chemprop/test_hpo_config.py` - Config tests
- **FILE-009**: `tests/admet/model/chemprop/test_hpo_search_space.py` - Search space tests
- **FILE-010**: `tests/admet/model/chemprop/test_hpo_trainable.py` - Trainable tests
- **FILE-011**: `tests/integration/test_hpo_integration.py` - Integration tests

## 6. Testing

- **TEST-001**: Unit test configuration loading and validation
- **TEST-002**: Unit test search space generation for all parameter types
- **TEST-003**: Unit test conditional search space logic
- **TEST-004**: Unit test RayTuneReportCallback metric reporting
- **TEST-005**: Integration test HPO with 5 trials, 5 epochs on synthetic data
- **TEST-006**: Smoke test full pipeline including transfer learning
- **TEST-007**: Test MLflow logging for HPO runs
- **TEST-008**: Test YAML config roundtrip (load, modify, save)

## 7. Risks & Assumptions

### Risks

- **RISK-001**: GPU memory contention with 4 concurrent trials - Mitigation: Start with 0.25 GPU fraction, adjust based on testing
- **RISK-002**: Ray and PyTorch Lightning callback conflicts - Mitigation: Test callback integration thoroughly
- **RISK-003**: MLflow nested run complexity - Mitigation: Follow existing ensemble.py pattern
- **RISK-004**: Long HPO runtime (100 trials × 150 epochs) - Mitigation: ASHA pruning should reduce effective compute by 3-5x

### Assumptions

- **ASSUMPTION-001**: Single train/validation split is representative enough for HPO
- **ASSUMPTION-002**: Validation loss is a good proxy for test performance
- **ASSUMPTION-003**: Best HPO config generalizes across different data splits
- **ASSUMPTION-004**: 4 trials per GPU fits in memory with batch_size ≤ 128

## 8. Related Specifications / Further Reading

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [ASHA Paper](https://arxiv.org/abs/1810.05934)
- [Chemprop Documentation](https://chemprop.readthedocs.io/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- Existing files: `src/admet/model/chemprop/ensemble.py`, `src/admet/model/chemprop/config.py`
