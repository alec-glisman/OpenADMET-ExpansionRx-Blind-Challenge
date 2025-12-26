<!-- markdownlint-disable-file -->

# Task Details: Unified Configuration and Ensemble Harmonization

## Research Reference

**Source Research**: #file:../research/20251226-config-harmonization-update-research.md

---

## Phase 1: Unified Configuration Schema

### Task 1.1: Create UnifiedModelConfig master dataclass

Create the master configuration dataclass that provides a single schema for all model types.

- **Files**:
  - `src/admet/model/config.py` - Add UnifiedModelConfig and supporting classes

- **Implementation**:

```python
# Add to src/admet/model/config.py

@dataclass
class UnifiedDataConfig:
    """Universal data configuration for all model types.

    Consolidates data paths, columns, and ensemble settings into a single
    config that works across Chemprop, classical models, and Chemeleon.
    """
    # Core paths
    data_dir: str = MISSING
    train_file: str | None = None
    validation_file: str | None = None
    test_file: str | None = None
    blind_file: str | None = None
    output_dir: str | None = None

    # Column configuration
    smiles_col: str = "SMILES"
    target_cols: list[str] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    quality_col: str | None = None  # For curriculum learning

    # Ensemble settings (split/fold discovery)
    splits: list[int] | None = None
    folds: list[int] | None = None


@dataclass
class UnifiedMlflowConfig:
    """Universal MLflow tracking configuration.

    Uses consistent field names across all models.
    Note: 'enabled' is the canonical field (not 'tracking').
    """
    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "admet"
    run_name: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    nested: bool = False
    log_model: bool = True


@dataclass
class UnifiedOptimizationConfig:
    """Universal optimization configuration.

    Contains superset of parameters for all model types. Neural models
    use learning rate scheduling; tree-based models ignore those fields.
    """
    # Common to all models
    seed: int = 42
    progress_bar: bool = False

    # Neural network training
    max_epochs: int = 150
    patience: int = 15
    batch_size: int = 32
    num_workers: int = 0

    # Learning rate (neural models) - use max_lr as primary
    init_lr: float = 1.0e-4
    max_lr: float = 1.0e-3
    final_lr: float = 1.0e-4
    warmup_epochs: int = 5

    # Loss function
    criterion: str = "MAE"


@dataclass
class RayConfig:
    """Configuration for Ray parallelization."""
    max_parallel: int = 1
    num_cpus: int | None = None
    num_gpus: int | None = None


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
        joint_sampling: { ... }  # Training strategies
        ray: { ... }
    """
    # Model type and parameters
    model: ModelSection = field(default_factory=lambda: ModelSection())

    # Core configs
    data: UnifiedDataConfig = field(default_factory=UnifiedDataConfig)
    optimization: UnifiedOptimizationConfig = field(default_factory=UnifiedOptimizationConfig)
    mlflow: UnifiedMlflowConfig = field(default_factory=UnifiedMlflowConfig)

    # Training strategies (model-agnostic, but validated per model type)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)
    inter_task_affinity: InterTaskAffinityConfig = field(default_factory=InterTaskAffinityConfig)

    # Ensemble/Ray settings
    ray: RayConfig = field(default_factory=RayConfig)


@dataclass
class ModelSection:
    """Model type discriminator and model-specific parameters.

    The 'type' field determines which nested config section is used.
    """
    type: str = "chemprop"

    # Model-specific parameters (only the matching type is used)
    chemprop: ChempropModelParams = field(default_factory=ChempropModelParams)
    chemeleon: ChemeleonModelParams = field(default_factory=ChemeleonModelParams)
    xgboost: XGBoostModelParams = field(default_factory=XGBoostModelParams)
    lightgbm: LightGBMModelParams = field(default_factory=LightGBMModelParams)
    catboost: CatBoostModelParams = field(default_factory=CatBoostModelParams)

    # Fingerprint config (for classical models)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
```

- **Success**:
  - UnifiedModelConfig passes OmegaConf structured validation
  - Can load any existing YAML config with proper structure
  - Model type discriminator correctly selects model params
  - All field names are consistent (no aliases)

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md (Lines 60-150) - Config analysis

- **Dependencies**:
  - Existing model param dataclasses must exist

---

### Task 1.2: Move training strategy configs to model-agnostic location

Move JointSamplingConfig, CurriculumConfig, TaskAffinityConfig from chemprop/config.py to model/config.py.

- **Files**:
  - `src/admet/model/config.py` - Add training strategy configs
  - `src/admet/model/chemprop/config.py` - Remove moved configs, add imports

- **Implementation**:

Move these dataclasses from `chemprop/config.py` to `model/config.py`:
- `TaskOversamplingConfig`
- `CurriculumConfig`
- `JointSamplingConfig`
- `TaskAffinityConfig`
- `InterTaskAffinityConfig`

Update imports in `chemprop/config.py`:
```python
from admet.model.config import (
    JointSamplingConfig,
    CurriculumConfig,
    TaskAffinityConfig,
    InterTaskAffinityConfig,
    TaskOversamplingConfig,
)
```

- **Success**:
  - All training strategy configs in `model/config.py`
  - `chemprop/config.py` imports from `model/config.py`
  - No circular import issues
  - Existing code continues to work

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md (Lines 100-140) - Feature analysis

- **Dependencies**:
  - Task 1.1: UnifiedModelConfig structure defined

---

### Task 1.3: Create model-type-aware config validation

Add validation that checks training strategy compatibility with model type.

- **Files**:
  - `src/admet/model/config.py` - Add validate_config() function

- **Implementation**:

```python
def validate_model_config(config: UnifiedModelConfig) -> None:
    """Validate configuration for model-type compatibility.

    Raises ConfigValidationError if incompatible options are enabled.

    Parameters
    ----------
    config : UnifiedModelConfig
        Configuration to validate.

    Raises
    ------
    ConfigValidationError
        If curriculum/sampling enabled for classical models.
    """
    model_type = config.model.type
    is_neural = model_type in ("chemprop", "chemeleon")
    is_classical = model_type in ("xgboost", "lightgbm", "catboost")

    # Curriculum learning requires PyTorch DataLoader
    if is_classical and config.joint_sampling.enabled:
        if config.joint_sampling.curriculum.enabled:
            raise ConfigValidationError(
                f"Curriculum learning is not supported for {model_type} models. "
                f"Curriculum requires PyTorch DataLoader which classical models do not use. "
                f"Set joint_sampling.curriculum.enabled=false or use a neural model (chemprop, chemeleon)."
            )
        if config.joint_sampling.task_oversampling.alpha > 0:
            raise ConfigValidationError(
                f"Task oversampling is not supported for {model_type} models. "
                f"Task oversampling requires PyTorch DataLoader. "
                f"Set joint_sampling.task_oversampling.alpha=0 or use a neural model."
            )

    # Inter-task affinity requires gradient access
    if is_classical and config.inter_task_affinity.enabled:
        raise ConfigValidationError(
            f"Inter-task affinity is not supported for {model_type} models. "
            f"Inter-task affinity requires gradient computation. "
            f"Set inter_task_affinity.enabled=false or use a neural model."
        )

    # Task affinity pre-training requires neural model
    if is_classical and config.task_affinity.enabled:
        raise ConfigValidationError(
            f"Task affinity grouping is not supported for {model_type} models. "
            f"Set task_affinity.enabled=false or use a neural model."
        )


class ConfigValidationError(ValueError):
    """Raised when configuration is invalid for the specified model type."""
    pass
```

- **Success**:
  - Clear error messages when incompatible options enabled
  - Validation runs before model creation
  - Neural models (chemprop, chemeleon) can use all features
  - Classical models error on curriculum/sampling/affinity

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md (Lines 50-80) - Feature compatibility

- **Dependencies**:
  - Task 1.1: UnifiedModelConfig must exist
  - Task 1.2: Training strategy configs in model/config.py

---

### Task 1.4: Write config schema tests

Create comprehensive tests for the unified config schema.

- **Files**:
  - `tests/test_unified_config.py` - NEW FILE

- **Implementation**:

```python
"""Tests for unified configuration schema."""

import pytest
from omegaconf import OmegaConf

from admet.model.config import (
    UnifiedModelConfig,
    ConfigValidationError,
    validate_model_config,
)


class TestUnifiedModelConfig:
    """Test UnifiedModelConfig schema."""

    def test_can_create_structured_config(self):
        """Test OmegaConf.structured() works."""
        config = OmegaConf.structured(UnifiedModelConfig)
        assert config.model.type == "chemprop"

    def test_can_merge_with_yaml(self, tmp_path):
        """Test merging with YAML file."""
        yaml_content = '''
model:
  type: xgboost
  xgboost:
    n_estimators: 200
data:
  smiles_col: SMILES
  target_cols: [LogD]
'''
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        base = OmegaConf.structured(UnifiedModelConfig)
        override = OmegaConf.load(yaml_file)
        config = OmegaConf.merge(base, override)

        assert config.model.type == "xgboost"
        assert config.model.xgboost.n_estimators == 200


class TestConfigValidation:
    """Test model-type-aware validation."""

    def test_classical_with_curriculum_raises(self):
        """Classical models cannot use curriculum."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "xgboost"
        config.joint_sampling.enabled = True
        config.joint_sampling.curriculum.enabled = True

        with pytest.raises(ConfigValidationError, match="not supported for xgboost"):
            validate_model_config(config)

    def test_neural_with_curriculum_ok(self):
        """Neural models can use curriculum."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "chemeleon"
        config.joint_sampling.enabled = True
        config.joint_sampling.curriculum.enabled = True

        validate_model_config(config)  # Should not raise
```

- **Success**:
  - Tests cover all model types
  - Tests verify validation errors
  - Tests verify YAML loading
  - All tests pass

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md - Config structure

- **Dependencies**:
  - Task 1.1-1.3 complete

---

## Phase 2: Unified Ensemble Implementation

### Task 2.1: Create merged ModelEnsemble class

Merge `src/admet/model/ensemble.py` (generic) and `src/admet/model/chemprop/ensemble.py` (feature-rich) into a single unified ensemble class.

- **Files**:
  - `src/admet/model/ensemble.py` - Rewrite with merged functionality

- **Implementation**:

```python
"""Unified ensemble training for all ADMET model types.

This module provides a model-agnostic ensemble class that works with any
model type registered in the ModelRegistry, featuring:
- Ray-based parallel training
- Split/fold discovery from data directory
- MLflow nested run tracking
- Aggregated predictions with uncertainty

Examples
--------
>>> from omegaconf import OmegaConf
>>> from admet.model.ensemble import ModelEnsemble
>>>
>>> config = OmegaConf.load("ensemble_config.yaml")
>>> ensemble = ModelEnsemble.from_config(config)
>>> ensemble.train_all()
>>> predictions, uncertainty = ensemble.predict_with_uncertainty(test_smiles)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf

from admet.model.base import BaseModel
from admet.model.config import (
    UnifiedModelConfig,
    validate_model_config,
    get_structured_config_for_model_type,
)
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelEnsemble(MLflowMixin):
    """Unified ensemble of ADMET models.

    Creates and manages an ensemble of models of any registered type.
    Supports training across multiple data splits/folds with Ray-based
    parallelization, MLflow tracking, and aggregated predictions.

    Parameters
    ----------
    config : DictConfig
        Unified configuration with model, data, optimization, and mlflow settings.

    Attributes
    ----------
    models : dict[str, BaseModel]
        Dictionary mapping model keys (split_X_fold_Y) to fitted models.
    model_type : str
        Type of model in the ensemble.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize ensemble from configuration."""
        self.config = config
        self.models: dict[str, BaseModel] = {}

        # Extract model type
        self.model_type = config.model.type

        # Validate config for model type
        validate_model_config(config)

        # Extract settings
        self._ray_config = config.get("ray", {})
        self._mlflow_config = config.get("mlflow", {})

        # Initialize MLflow if enabled
        if self._mlflow_config.get("enabled", False):
            self.init_mlflow(self._mlflow_config)

    @classmethod
    def from_config(cls, config: DictConfig) -> "ModelEnsemble":
        """Create ensemble from configuration file or dict."""
        # Ensure we have structured config with defaults
        base_config = get_structured_config_for_model_type(config.model.type)
        merged = OmegaConf.merge(base_config, config)
        OmegaConf.resolve(merged)
        return cls(merged)

    def discover_splits_folds(self) -> List[Tuple[int, int, Path, Path]]:
        """Discover split/fold combinations from data directory.

        Returns list of (split_idx, fold_idx, train_path, val_path) tuples.
        """
        # ... implementation from chemprop/ensemble.py ...

    def train_all(self) -> "ModelEnsemble":
        """Train all ensemble members in parallel using Ray."""
        # ... implementation merging both ensembles ...

    def predict(self, smiles: List[str]) -> np.ndarray:
        """Generate aggregated ensemble predictions."""
        # ... implementation ...

    def predict_with_uncertainty(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty (mean, std)."""
        # ... implementation ...
```

- **Success**:
  - Single ModelEnsemble class handles all model types
  - Ray parallelization works for all models
  - MLflow nested runs work correctly
  - Split/fold discovery from data directory

- **Research References**:
  - [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) - Feature-rich implementation
  - [src/admet/model/ensemble.py](src/admet/model/ensemble.py) - Simple implementation

- **Dependencies**:
  - Phase 1 complete: UnifiedModelConfig must exist

---

### Task 2.2: Implement model-type-aware config merging

Create factory function to get appropriate base config for each model type.

- **Files**:
  - `src/admet/model/config.py` - Add get_structured_config_for_model_type()

- **Implementation**:

```python
def get_structured_config_for_model_type(model_type: str) -> DictConfig:
    """Get appropriate structured config base for model type.

    This ensures config merging uses the correct defaults for each model type.

    Parameters
    ----------
    model_type : str
        Model type: "chemprop", "chemeleon", "xgboost", "lightgbm", "catboost"

    Returns
    -------
    DictConfig
        Structured config with appropriate defaults.
    """
    # Use UnifiedModelConfig as base for all types
    base = OmegaConf.structured(UnifiedModelConfig)
    base.model.type = model_type

    # Apply model-type-specific defaults
    if model_type in ("xgboost", "lightgbm", "catboost"):
        # Classical models: disable neural-specific features
        base.joint_sampling.enabled = False
        base.task_affinity.enabled = False
        base.inter_task_affinity.enabled = False

    return base
```

- **Success**:
  - Factory returns correct base config for each model type
  - Classical models have sampling/affinity disabled by default
  - Config merging preserves user overrides

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md (Lines 148-160) - Config factory

- **Dependencies**:
  - Task 1.1: UnifiedModelConfig must exist

---

### Task 2.3: Implement Ray parallelization for all model types

Update the Ray remote function to handle any model type.

- **Files**:
  - `src/admet/model/ensemble.py` - Implement train_single_model Ray task

- **Implementation**:

```python
@ray.remote
def train_single_model(
    config_dict: Dict[str, Any],
    split_idx: int,
    fold_idx: int,
    train_file: str,
    val_file: str,
    parent_run_id: Optional[str],
) -> Tuple[str, Dict[str, float], Optional[pd.DataFrame]]:
    """Train a single model as a Ray task.

    Works with any model type registered in ModelRegistry.
    """
    import lightning.pytorch as pl

    # Get model type and appropriate base config
    model_type = config_dict.get("model", {}).get("type", "chemprop")
    base_config = get_structured_config_for_model_type(model_type)
    override_config = OmegaConf.create(config_dict)
    config = OmegaConf.merge(base_config, override_config)
    OmegaConf.resolve(config)

    model_key = f"split_{split_idx}_fold_{fold_idx}"

    # Seed for reproducibility
    pl.seed_everything(config.optimization.seed, workers=True)

    # Configure MLflow nested run
    config.mlflow.parent_run_id = parent_run_id
    config.mlflow.nested = parent_run_id is not None

    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file) if Path(val_file).exists() else None

    smiles_col = config.data.smiles_col
    target_cols = list(config.data.target_cols)

    train_smiles = train_df[smiles_col].tolist()
    train_y = train_df[target_cols].values

    val_smiles = val_df[smiles_col].tolist() if val_df is not None else None
    val_y = val_df[target_cols].values if val_df is not None else None

    # Create and train model via registry
    model = ModelRegistry.create(config)
    model.fit(train_smiles, train_y, val_smiles=val_smiles, val_y=val_y)

    # Collect metrics
    metrics = model.get_metrics() if hasattr(model, "get_metrics") else {}

    return model_key, metrics, None  # Return model separately if needed
```

- **Success**:
  - Ray task works for all model types
  - Proper config merging per model type
  - MLflow nested runs work
  - Data loading is model-agnostic

- **Research References**:
  - [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) (Lines 600-700) - Existing Ray implementation

- **Dependencies**:
  - Task 2.1: ModelEnsemble class structure
  - Task 2.2: get_structured_config_for_model_type()

---

### Task 2.4: Update MLflow integration for unified ensemble

Ensure MLflow logging works consistently across all model types.

- **Files**:
  - `src/admet/model/ensemble.py` - Update MLflow integration

- **Implementation**:

The ensemble should:
1. Create a parent MLflow run
2. Each model creates a nested child run
3. Aggregate metrics logged to parent run
4. Model artifacts saved per child run

```python
def train_all(self) -> "ModelEnsemble":
    """Train all ensemble members with MLflow tracking."""
    # Initialize Ray
    ray_config = self._ray_config
    if not ray.is_initialized():
        ray.init(
            num_cpus=ray_config.get("num_cpus"),
            num_gpus=ray_config.get("num_gpus"),
        )

    # Start parent MLflow run
    parent_run_id = None
    if self._mlflow_config.get("enabled", False):
        import mlflow
        mlflow.set_tracking_uri(self._mlflow_config.get("tracking_uri"))
        mlflow.set_experiment(self._mlflow_config.get("experiment_name", "ensemble"))

        with mlflow.start_run(run_name=self._mlflow_config.get("run_name")) as run:
            parent_run_id = run.info.run_id

            # Log ensemble config
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_splits_folds", len(self.discover_splits_folds()))

            # Train models
            self._train_with_ray(parent_run_id)

            # Log aggregate metrics
            self._log_aggregate_metrics()
    else:
        self._train_with_ray(parent_run_id=None)

    return self
```

- **Success**:
  - Parent run created for ensemble
  - Child runs for each model
  - Aggregate metrics in parent
  - Works for all model types

- **Research References**:
  - [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) - MLflow integration

- **Dependencies**:
  - Task 2.1-2.3 complete

---

### Task 2.5: Write ensemble integration tests

Create tests for the unified ensemble class.

- **Files**:
  - `tests/test_unified_ensemble.py` - NEW FILE

- **Implementation**:

```python
"""Tests for unified ModelEnsemble class."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from admet.model.ensemble import ModelEnsemble


class TestModelEnsemble:
    """Test unified ModelEnsemble."""

    @pytest.fixture
    def xgboost_config(self, tmp_path):
        """Create XGBoost ensemble config."""
        return OmegaConf.create({
            "model": {"type": "xgboost", "xgboost": {"n_estimators": 10}},
            "data": {
                "data_dir": str(tmp_path),
                "smiles_col": "SMILES",
                "target_cols": ["target"],
            },
            "mlflow": {"enabled": False},
            "ray": {"max_parallel": 1},
        })

    def test_from_config_xgboost(self, xgboost_config):
        """Test creating XGBoost ensemble."""
        ensemble = ModelEnsemble.from_config(xgboost_config)
        assert ensemble.model_type == "xgboost"

    def test_curriculum_disabled_for_classical(self, xgboost_config):
        """Test curriculum raises error for classical."""
        xgboost_config.joint_sampling.enabled = True
        xgboost_config.joint_sampling.curriculum.enabled = True

        with pytest.raises(ConfigValidationError):
            ModelEnsemble.from_config(xgboost_config)
```

- **Success**:
  - Tests cover all model types
  - Tests verify Ray parallelization
  - Tests verify MLflow integration
  - Tests verify validation errors

- **Research References**:
  - `tests/test_ensemble_generic.py` - Existing generic ensemble tests
  - `tests/test_ensemble_chemprop.py` - Existing chemprop ensemble tests

- **Dependencies**:
  - Task 2.1-2.4 complete

---

## Phase 3: Model Adapter Updates

### Task 3.1: Update ChempropAdapter for unified config

Update the ChempropAdapter to work with UnifiedModelConfig.

- **Files**:
  - `src/admet/model/chemprop/adapter.py` - Update for unified config

- **Implementation**:

```python
@ModelRegistry.register("chemprop")
class ChempropModelAdapter(BaseModel, MLflowMixin):
    """BaseModel adapter for ChempropModel with unified config support."""

    model_type = "chemprop"

    def __init__(self, config: DictConfig) -> None:
        """Initialize adapter with unified configuration."""
        super().__init__(config)
        self._model: ChempropModel | None = None

        # Extract from unified config structure
        self._smiles_col = config.data.smiles_col
        self._target_cols = list(config.data.target_cols)

        # Model params from model.chemprop section
        self._model_params = config.model.chemprop

        # Training strategy configs (now at root level)
        self._joint_sampling = config.get("joint_sampling")
        self._task_affinity = config.get("task_affinity")
        self._inter_task_affinity = config.get("inter_task_affinity")

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChempropModelAdapter":
        """Create adapter from unified config."""
        # Ensure proper config structure
        from admet.model.config import get_structured_config_for_model_type

        base = get_structured_config_for_model_type("chemprop")
        merged = OmegaConf.merge(base, config)
        OmegaConf.resolve(merged)

        return cls(merged)
```

- **Success**:
  - Adapter works with UnifiedModelConfig
  - Extracts model params from `model.chemprop`
  - Accesses training strategies from root level
  - Maintains full ChempropModel functionality

- **Research References**:
  - [src/admet/model/chemprop/adapter.py](src/admet/model/chemprop/adapter.py) - Current implementation

- **Dependencies**:
  - Phase 1 complete
  - Phase 2 complete

---

### Task 3.2: Update ChemeleonModel for unified config and JointSampling

Update ChemeleonModel to use unified config and support JointSampling.

- **Files**:
  - `src/admet/model/chemeleon/model.py` - Update for unified config and JointSampling

- **Implementation**:

```python
@ModelRegistry.register("chemeleon")
class ChemeleonModel(BaseModel, MLflowMixin):
    """Chemeleon foundation model with unified config support."""

    model_type = "chemeleon"

    def __init__(self, config: DictConfig) -> None:
        """Initialize with unified configuration."""
        super().__init__(config)

        # Extract from unified config structure
        self._model_params = config.model.chemeleon
        self._smiles_col = config.data.smiles_col
        self._target_cols = list(config.data.target_cols)

        # JointSampling support (Chemeleon uses PyTorch DataLoader)
        self._joint_sampling_config = config.get("joint_sampling")

        # Initialize JointSampler if enabled
        self._joint_sampler = None
        if self._joint_sampling_config and self._joint_sampling_config.enabled:
            self._setup_joint_sampling()

    def _setup_joint_sampling(self) -> None:
        """Configure JointSampler for Chemeleon training."""
        from admet.model.chemprop.joint_sampler import JointSampler

        # JointSampler will be created during fit() when we have data
        self._use_joint_sampling = True

    def _create_train_dataloader(self, dataset, batch_size: int) -> DataLoader:
        """Create training DataLoader with optional JointSampler."""
        if self._use_joint_sampling and self._joint_sampler is not None:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self._joint_sampler,
                num_workers=self.config.optimization.num_workers,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.optimization.num_workers,
        )
```

- **Success**:
  - Chemeleon uses unified config structure
  - JointSampler works for curriculum/task oversampling
  - DataLoader creation respects sampling config
  - Maintains existing Chemeleon functionality

- **Research References**:
  - [src/admet/model/chemeleon/model.py](src/admet/model/chemeleon/model.py) - Current implementation
  - [src/admet/model/chemprop/joint_sampler.py](src/admet/model/chemprop/joint_sampler.py) - JointSampler

- **Dependencies**:
  - Phase 1 complete
  - JointSampler must exist in accessible location

---

### Task 3.3: Update ClassicalModelBase for unified config

Update classical models to use unified config structure.

- **Files**:
  - `src/admet/model/classical/base.py` - Update for unified config

- **Implementation**:

```python
class ClassicalModelBase(BaseModel, MLflowMixin):
    """Base class for classical ML models with unified config support."""

    model_type: str = "classical"

    def __init__(self, config: DictConfig) -> None:
        """Initialize with unified configuration."""
        self.config = config
        self._is_fitted = False
        self._model = None

        # Extract from unified config structure
        self._smiles_col = config.data.smiles_col
        self._target_cols = list(config.data.target_cols)

        # Model params from model.<type> section
        self._model_params = self._get_model_params()

        # Fingerprint config from model.fingerprint
        self._fingerprint_config = config.model.fingerprint
        self.fingerprint_generator = FingerprintGenerator(self._fingerprint_config)

        # MLflow setup
        if config.mlflow.enabled:
            self.init_mlflow(config.mlflow)

    def _get_model_params(self) -> DictConfig:
        """Get model-specific parameters from unified config."""
        model_type = self.model_type
        return getattr(self.config.model, model_type, {})
```

- **Success**:
  - Classical models use unified config structure
  - Model params from `model.<type>` section
  - Fingerprint config from `model.fingerprint`
  - Maintains existing functionality

- **Research References**:
  - [src/admet/model/classical/base.py](src/admet/model/classical/base.py) - Current implementation

- **Dependencies**:
  - Phase 1 complete

---

### Task 3.4: Add curriculum validation error for classical models

Ensure clear error message when curriculum enabled for classical models.

- **Files**:
  - `src/admet/model/classical/base.py` - Add validation in __init__

- **Implementation**:

```python
def __init__(self, config: DictConfig) -> None:
    """Initialize with unified configuration."""
    # Validate config before initialization
    from admet.model.config import validate_model_config, ConfigValidationError

    try:
        validate_model_config(config)
    except ConfigValidationError as e:
        # Re-raise with model-type context
        raise ConfigValidationError(
            f"Configuration error for {self.model_type} model: {e}"
        ) from e

    # ... rest of initialization ...
```

The error message should be clear and actionable:
```
ConfigValidationError: Curriculum learning is not supported for xgboost models.
Curriculum requires PyTorch DataLoader which classical models do not use.
Set joint_sampling.curriculum.enabled=false or use a neural model (chemprop, chemeleon).
```

- **Success**:
  - Clear error at model creation time
  - Error message explains why and how to fix
  - Validation runs for all classical models

- **Research References**:
  - Task 1.3: validate_model_config() implementation

- **Dependencies**:
  - Task 1.3: validate_model_config() must exist
  - Task 3.3: ClassicalModelBase updated

---

## Phase 4: Registry and Factory Updates

### Task 4.1: Update ModelRegistry.create() for unified config

Update ModelRegistry to use unified config validation.

- **Files**:
  - `src/admet/model/registry.py` - Update create() method

- **Implementation**:

```python
@classmethod
def create(cls, config: DictConfig) -> BaseModel:
    """Create model instance from unified configuration.

    Validates configuration for model-type compatibility before
    instantiating the model.

    Parameters
    ----------
    config : DictConfig
        Unified configuration with model.type field.

    Returns
    -------
    BaseModel
        Instantiated and validated model.

    Raises
    ------
    ConfigValidationError
        If configuration is incompatible with model type.
    ValueError
        If model type is unknown.
    """
    from admet.model.config import validate_model_config

    # Validate config before creating model
    validate_model_config(config)

    model_type = config.model.type
    if model_type not in cls._registry:
        available = list(cls._registry.keys())
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")

    model_cls = cls._registry[model_type]
    logger.info(f"Creating {model_type} model")
    return model_cls.from_config(config)
```

- **Success**:
  - Config validation before model creation
  - Clear error messages for invalid configs
  - Works with all registered model types

- **Research References**:
  - [src/admet/model/registry.py](src/admet/model/registry.py) - Current implementation

- **Dependencies**:
  - Task 1.3: validate_model_config() must exist

---

### Task 4.2: Create get_structured_config_for_model_type() factory

Implement the factory function for model-type-specific base configs.

- **Files**:
  - `src/admet/model/config.py` - Add factory function

- **Implementation**:

```python
def get_structured_config_for_model_type(model_type: str) -> DictConfig:
    """Get structured config with appropriate defaults for model type.

    Returns a base configuration that can be merged with user config
    to ensure all fields have appropriate defaults.

    Parameters
    ----------
    model_type : str
        Model type identifier.

    Returns
    -------
    DictConfig
        Structured config with model-type-appropriate defaults.

    Examples
    --------
    >>> base = get_structured_config_for_model_type("xgboost")
    >>> user_config = OmegaConf.load("my_config.yaml")
    >>> config = OmegaConf.merge(base, user_config)
    """
    valid_types = ("chemprop", "chemeleon", "xgboost", "lightgbm", "catboost")
    if model_type not in valid_types:
        raise ValueError(f"Unknown model type: {model_type}. Valid: {valid_types}")

    # Create structured base with UnifiedModelConfig
    base = OmegaConf.structured(UnifiedModelConfig)
    base.model.type = model_type

    # Model-type-specific defaults
    if model_type in ("xgboost", "lightgbm", "catboost"):
        # Classical models: disable unsupported features
        base.joint_sampling.enabled = False
        base.joint_sampling.task_oversampling.alpha = 0.0
        base.joint_sampling.curriculum.enabled = False
        base.task_affinity.enabled = False
        base.inter_task_affinity.enabled = False

    elif model_type == "chemeleon":
        # Chemeleon: enable joint sampling by default
        base.joint_sampling.enabled = True

    return base
```

- **Success**:
  - Factory returns correct defaults for each model type
  - Classical models have neural-only features disabled
  - Config merging preserves user overrides
  - Used by ensemble and registry

- **Research References**:
  - #file:../research/20251226-config-harmonization-update-research.md

- **Dependencies**:
  - Task 1.1: UnifiedModelConfig must exist

---

### Task 4.3: Update from_config() methods across all models

Ensure all model classes use consistent from_config() pattern.

- **Files**:
  - `src/admet/model/chemprop/adapter.py` - Update from_config()
  - `src/admet/model/chemeleon/model.py` - Update from_config()
  - `src/admet/model/classical/xgboost_model.py` - Update from_config()
  - `src/admet/model/classical/lightgbm_model.py` - Update from_config()
  - `src/admet/model/classical/catboost_model.py` - Update from_config()

- **Implementation**:

All from_config() methods should follow this pattern:

```python
@classmethod
def from_config(cls, config: DictConfig) -> "ModelClass":
    """Create model from unified configuration.

    Parameters
    ----------
    config : DictConfig
        Unified configuration. Can be partial; will be merged
        with structured defaults.

    Returns
    -------
    ModelClass
        Configured model instance.
    """
    from admet.model.config import get_structured_config_for_model_type

    # Get base config with defaults
    base = get_structured_config_for_model_type(cls.model_type)

    # Merge with user config
    merged = OmegaConf.merge(base, config)
    OmegaConf.resolve(merged)

    return cls(merged)
```

- **Success**:
  - All models use same from_config() pattern
  - Config merging with proper defaults
  - Validation happens automatically

- **Research References**:
  - Current from_config() implementations in each model

- **Dependencies**:
  - Task 4.2: get_structured_config_for_model_type() must exist

---
## Phase 5: Configuration Migration

### Task 5.1: Create config migration script

Create a script to migrate existing YAML configs to the new unified schema.

- **Files**:
  - `scripts/lib/migrate_to_unified_config.py` - NEW FILE

- **Implementation**:

```python
#!/usr/bin/env python3
"""Migrate YAML configs to unified schema.

This script converts existing config files to the new unified structure:
- Moves model params under model.<type>
- Standardizes mlflow.enabled (from mlflow.tracking)
- Moves joint_sampling to root level if nested under chemprop

Usage:
    python scripts/lib/migrate_to_unified_config.py --config-file config.yaml
    python scripts/lib/migrate_to_unified_config.py --config-dir configs/ --dry-run
"""

import argparse
from pathlib import Path

import yaml


def migrate_config(config: dict) -> dict:
    """Migrate config dict to unified schema."""
    migrated = {}

    # Handle model section
    model_section = config.get("model", {})
    model_type = model_section.get("type", "chemprop")

    migrated["model"] = {
        "type": model_type,
    }

    # Move model params under model.<type>
    if model_type in model_section:
        migrated["model"][model_type] = model_section[model_type]
    else:
        # Legacy: params directly in model section
        params = {k: v for k, v in model_section.items() if k not in ("type", "fingerprint")}
        if params:
            migrated["model"][model_type] = params

    # Handle fingerprint for classical models
    if "fingerprint" in model_section:
        migrated["model"]["fingerprint"] = model_section["fingerprint"]

    # Standardize mlflow
    mlflow = config.get("mlflow", {})
    if "tracking" in mlflow and "enabled" not in mlflow:
        mlflow["enabled"] = mlflow.pop("tracking")
    migrated["mlflow"] = mlflow

    # Move joint_sampling to root
    if "joint_sampling" in config:
        migrated["joint_sampling"] = config["joint_sampling"]

    # Copy other sections
    for key in ("data", "optimization", "ray", "task_affinity", "inter_task_affinity"):
        if key in config:
            migrated[key] = config[key]

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate configs to unified schema")
    parser.add_argument("--config-file", type=Path, help="Single config file to migrate")
    parser.add_argument("--config-dir", type=Path, help="Directory of configs to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    # ... implementation ...


if __name__ == "__main__":
    main()
```

- **Success**:
  - Script handles all existing config formats
  - Dry-run mode for safety
  - Preserves comments where possible
  - Reports changes made

- **Research References**:
  - [scripts/lib/migrate_sampling_configs.py](scripts/lib/migrate_sampling_configs.py) - Existing migration script

- **Dependencies**:
  - Phase 1 complete: Know target schema

---

### Task 5.2: Migrate example configs in configs/0-experiment/

Migrate all example configs to the new unified schema.

- **Files**:
  - `configs/0-experiment/chemprop.yaml` - Migrate
  - `configs/0-experiment/chemeleon.yaml` - Migrate
  - `configs/0-experiment/ensemble_chemprop_production.yaml` - Migrate
  - `configs/0-experiment/ensemble_joint_sampling_example.yaml` - Migrate

- **Implementation**:

Example unified config structure:

```yaml
# configs/0-experiment/unified_example.yaml
# Unified config example for any model type

model:
  type: chemprop  # or chemeleon, xgboost, lightgbm, catboost
  chemprop:
    depth: 4
    message_hidden_dim: 1100
    aggregation: norm
    ffn_type: regression
    num_layers: 2
    hidden_dim: 500
    dropout: 0.25
    batch_norm: true
  # fingerprint section for classical models:
  # fingerprint:
  #   type: morgan
  #   morgan:
  #     radius: 2
  #     n_bits: 2048

data:
  data_dir: assets/dataset/split_train_val/...
  smiles_col: SMILES
  target_cols:
    - LogD
    - Log KSOL
  target_weights: [1.0, 1.0]
  quality_col: Quality  # For curriculum learning
  splits: null
  folds: null

optimization:
  criterion: MAE
  init_lr: 1.04e-05
  max_lr: 0.000104
  final_lr: 2.09e-06
  warmup_epochs: 5
  max_epochs: 150
  patience: 15
  batch_size: 128
  seed: 42

mlflow:
  enabled: true  # Canonical field name (not 'tracking')
  tracking_uri: http://127.0.0.1:8084
  experiment_name: unified_example

joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.1
  curriculum:
    enabled: false
    quality_col: Quality

ray:
  max_parallel: 5
```

- **Success**:
  - All example configs migrated
  - Configs load without errors
  - Documentation comments added
  - Tests pass with migrated configs

- **Research References**:
  - Existing configs in `configs/0-experiment/`

- **Dependencies**:
  - Task 5.1: Migration script complete

---

### Task 5.3: Update documentation with new schema

Update documentation to reflect the unified config schema.

- **Files**:
  - `docs/guide/configuration.rst` - Update or create
  - `README.md` - Update config examples
  - `MODEL_CARD.md` - Update if needed

- **Implementation**:

Document the unified config schema:

```markdown
## Configuration Schema

All models use a unified configuration schema. The `model.type` field
determines which model is trained.

### Model Types

- `chemprop` - Chemprop MPNN (supports curriculum, task affinity)
- `chemeleon` - Chemeleon foundation model (supports curriculum)
- `xgboost` - XGBoost with fingerprints
- `lightgbm` - LightGBM with fingerprints
- `catboost` - CatBoost with fingerprints

### Training Strategies

**Note**: Curriculum learning and task oversampling require PyTorch DataLoader.
Only neural models (chemprop, chemeleon) support these features.

Attempting to enable curriculum for classical models will raise:
```
ConfigValidationError: Curriculum learning is not supported for xgboost models.
```

### Example Configs

See `configs/0-experiment/` for examples of each model type.
```

- **Success**:
  - Documentation covers all model types
  - Clear explanation of feature support
  - Error messages documented
  - Examples provided

- **Research References**:
  - Existing documentation

- **Dependencies**:
  - Phase 1-4 complete

---

## Phase 6: Cleanup and Final Testing

### Task 6.1: Remove old ensemble.py and chemprop/ensemble.py

After the unified ModelEnsemble is working, remove the old implementations.

- **Files**:
  - `src/admet/model/ensemble.py` - KEEP (now unified)
  - `src/admet/model/chemprop/ensemble.py` - DELETE
  - Update imports across codebase

- **Implementation**:

1. Verify unified ModelEnsemble passes all tests
2. Delete `src/admet/model/chemprop/ensemble.py`
3. Update any imports:
   ```python
   # Old:
   from admet.model.chemprop.ensemble import ModelEnsemble

   # New:
   from admet.model.ensemble import ModelEnsemble
   ```
4. Update `src/admet/model/chemprop/__init__.py` to remove export

- **Success**:
  - Old chemprop/ensemble.py deleted
  - All imports updated
  - No broken imports in codebase
  - Tests pass

- **Research References**:
  - `grep -r "from admet.model.chemprop.ensemble" src/ tests/`

- **Dependencies**:
  - Phase 2 complete and tested

---

### Task 6.2: Remove deprecated config classes

Remove old config classes that are no longer needed.

- **Files**:
  - `src/admet/model/chemprop/config.py` - Clean up

- **Implementation**:

After migration, remove:
- Duplicate `DataConfig` (use `UnifiedDataConfig`)
- Duplicate `MlflowConfig` (use `UnifiedMlflowConfig`)
- `ChempropConfig` (replaced by `UnifiedModelConfig`)
- `EnsembleConfig` (replaced by `UnifiedModelConfig`)
- `EnsembleDataConfig` (merged into `UnifiedDataConfig`)

Keep:
- `ModelConfig` (Chemprop architecture params, now `ChempropModelParams`)
- `OptimizationConfig` (now `UnifiedOptimizationConfig`)
- Training strategy configs (moved to `model/config.py`)

- **Success**:
  - No duplicate config classes
  - All imports updated
  - Backward compatibility removed (per user request)
  - Tests pass

- **Research References**:
  - `grep -r "ChempropConfig\|EnsembleConfig" src/ tests/`

- **Dependencies**:
  - Phase 5 complete (configs migrated)

---

### Task 6.3: Run full test suite and fix regressions

Run complete test suite and fix any regressions.

- **Files**:
  - `tests/` - All test files

- **Implementation**:

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run specific test categories
pytest tests/test_unified_config.py -v
pytest tests/test_unified_ensemble.py -v
pytest tests/test_chemprop*.py -v
pytest tests/test_chemeleon*.py -v
pytest tests/test_classical*.py -v
```

Fix any failures:
1. Update test imports
2. Update test configs to new schema
3. Update assertions for new API
4. Remove tests for deleted classes

- **Success**:
  - All tests pass
  - No regressions
  - Coverage maintained
  - CI passes

- **Research References**:
  - Existing test files

- **Dependencies**:
  - All previous phases complete

---

### Task 6.4: Update imports across codebase

Ensure all imports use the new locations.

- **Files**:
  - All Python files that import config or ensemble classes

- **Implementation**:

Search and update:
```bash
# Find old imports
grep -r "from admet.model.chemprop.config import" src/ tests/
grep -r "from admet.model.chemprop.ensemble import" src/ tests/
```

Update to:
```python
# Config classes
from admet.model.config import (
    UnifiedModelConfig,
    UnifiedDataConfig,
    UnifiedMlflowConfig,
    UnifiedOptimizationConfig,
    JointSamplingConfig,
    CurriculumConfig,
    validate_model_config,
    get_structured_config_for_model_type,
)

# Ensemble
from admet.model.ensemble import ModelEnsemble
```

- **Success**:
  - No imports from deleted modules
  - All imports resolve correctly
  - No circular imports
  - linting passes

- **Research References**:
  - `grep -r "from admet.model" src/`

- **Dependencies**:
  - Task 6.1-6.2 complete
