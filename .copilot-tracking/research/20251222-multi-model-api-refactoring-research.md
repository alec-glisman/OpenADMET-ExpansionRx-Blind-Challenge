<!-- markdownlint-disable-file -->

# Task Research Notes: Multi-Model API Refactoring

## Research Executed

### File Analysis

- [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py)
  - 695 lines defining comprehensive OmegaConf dataclass configs
  - `DataConfig`, `ModelConfig`, `OptimizationConfig`, `MlflowConfig` are reusable across model types
  - `ChempropConfig` is top-level config combining all sub-configs
  - Model-specific params currently in flat `ModelConfig` (depth, message_hidden_dim, ffn_type, etc.)

- [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py)
  - 2484 lines, `ChempropModel` class handles full training workflow
  - Key methods: `fit()`, `predict()`, `from_config()`, `to_config()`
  - Uses PyTorch Lightning for training orchestration
  - Heavy MLflow integration for tracking
  - `ChempropHyperparams` dataclass duplicates some config information

- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py)
  - 1355 lines, `ChempropEnsemble` orchestrates multi-model training
  - Uses Ray for parallelization
  - `from_config()` factory pattern

- [src/admet/model/classical/models.py](src/admet/model/classical/models.py)
  - Simple wrapper functions: `build_model()`, `fit_model()`, `predict_model()`
  - Supports xgboost, lightgbm via `MultiOutputRegressor`
  - No class-based abstraction, no config system

- [src/admet/cli/model.py](src/admet/cli/model.py)
  - CLI commands hardcoded to chemprop modules
  - `train`, `ensemble`, `hpo` commands call chemprop-specific modules

### Code Search Results

- `from_config|from_yaml`
  - 18 matches - pattern already established in chemprop modules
  - Consistent factory pattern: `Model.from_config(config)`

- `class.*Model|def train|def fit`
  - 15 matches across model modules
  - `ChempropModel`, `ChempropHyperparams`, `ModelConfig` are key classes

### External Research

- #fetch:https://scikit-learn.org/stable/developers/develop.html
  - **Estimator API**: `fit()`, `predict()`, `score()`, `get_params()`, `set_params()`
  - **Mixins**: `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`
  - **Conventions**: init stores params, fit does training, trailing underscore for fitted attributes
  - **Key insight**: Validation in `fit()`, not `__init__()`

- #githubRepo:"autogluon/autogluon" multi-model registry abstract base class
  - **AbstractModel base class** in `core/models/abstract/abstract_model.py`
  - **ModelRegistry** for dynamic model type registration
  - **Key methods**: `fit()`, `predict()`, `predict_proba()`, `score()`, `save()`, `load()`
  - **Model discovery**: Registry pattern allows `MODEL_REGISTRY.register()` decorator
  - **Configuration**: Hyperparameters passed via `hyperparameters` dict in constructor

### Project Conventions

- Standards referenced: OmegaConf dataclass configs, type hints, docstrings (PEP 257)
- Instructions followed: `.github/instructions/python.instructions.md`

## Key Discoveries

### Current Architecture Issues

1. **Tight Coupling**: `ModelConfig` contains chemprop-specific params (depth, message_hidden_dim, etc.)
2. **Duplicate Definitions**: `ChempropHyperparams` and `ModelConfig` overlap
3. **No Abstraction Layer**: No common interface for different model types
4. **CLI Hardcoding**: CLI directly imports chemprop modules
5. **Config Not Model-Aware**: YAML has no `model_type` discriminator

### Implementation Patterns from AutoGluon

```python
# AutoGluon AbstractModel pattern
class AbstractModel(ModelBase, Tunable):
    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        problem_type: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ):
        ...

    def fit(self, X, y, X_val=None, y_val=None, **kwargs) -> Self:
        # Template method - calls _fit()
        self._fit(X, y, X_val, y_val, **kwargs)
        return self

    @abstractmethod
    def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """Subclasses implement actual training logic"""
        pass

    def predict(self, X, **kwargs) -> np.ndarray:
        """Returns predictions"""
        ...

    @classmethod
    def load(cls, path: str) -> Self:
        """Load model from disk"""
        ...

    def save(self, path: str) -> str:
        """Save model to disk"""
        ...
```

### sklearn Estimator API Conventions

```python
from sklearn.base import BaseEstimator, RegressorMixin

class MyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, param1=1, param2=2):
        # Store params exactly as passed (no validation)
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y, sample_weight=None):
        # Validation happens here, not in __init__
        X, y = check_X_y(X, y)
        # Training logic
        self.coef_ = ...  # Trailing underscore for fitted attributes
        return self

    def predict(self, X):
        check_is_fitted(self, ['coef_'])
        return ...

    def score(self, X, y):
        # Default R² for regressors via RegressorMixin
        return super().score(X, y)
```

### Model Registry Pattern

```python
# From AutoGluon multimodal/tests/unittests/others/test_registry.py
class ModelRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str = None):
        def decorator(model_cls):
            key = name or model_cls.__name__
            cls._registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._registry.keys())
```

### Proposed Config Structure

```yaml
# New YAML structure with model_type discriminator
data:
  data_dir: assets/dataset/...
  smiles_col: SMILES
  target_cols: [LogD, "Log KSOL", ...]
  target_weights: [1.0, 1.0, ...]

model:
  type: chemprop  # <-- Discriminator field
  # Common model params
  seed: 12345

  # Model-specific params nested under model type
  chemprop:
    depth: 5
    message_hidden_dim: 600
    num_layers: 2
    hidden_dim: 600
    dropout: 0.1
    batch_norm: true
    ffn_type: regression
    aggregation: mean

optimization:
  criterion: MAE
  init_lr: 0.00113
  max_lr: 0.000227
  final_lr: 0.000113
  warmup_epochs: 5
  max_epochs: 150
  patience: 15
  batch_size: 128

mlflow:
  tracking: true
  experiment_name: production_ensemble
```

### Alternative Config Structures

**Option A: Nested model-specific params (Recommended)**
```yaml
model:
  type: chemprop
  chemprop:
    depth: 5
    ...
```

**Option B: Flat discriminated union**
```yaml
model:
  type: chemprop
  depth: 5  # Only valid when type=chemprop
  n_estimators: 100  # Only valid when type=xgboost
```

**Option C: Separate config files per model type**
```yaml
# Uses YAML inheritance
defaults:
  - model/chemprop  # Pulls in model-specific defaults
```

### Technical Requirements

1. **Abstract Base Class**: Define common interface
2. **Config Dataclasses**: Shared + model-specific configs
3. **Factory Pattern**: `ModelFactory.create(config)` dispatches by type
4. **Registry**: Dynamic model registration
5. **Backward Compatibility**: Existing configs should continue working

## Recommended Approach

### Phase 1: Define Abstract Model Interface

Create `src/admet/model/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


@dataclass
class BaseModelConfig:
    """Common configuration shared across all model types."""
    seed: int = 12345


@dataclass
class BaseOptimizationConfig:
    """Optimization settings applicable to all models."""
    criterion: str = "MAE"
    max_epochs: int = 150
    patience: int = 15
    batch_size: int = 128


class BaseModel(ABC):
    """Abstract base class for all ADMET prediction models.

    Follows sklearn estimator conventions with additions for
    molecular property prediction.
    """

    # Class-level attributes
    model_type: str  # e.g., "chemprop", "xgboost", "lightgbm"
    supports_multi_task: bool = True
    supports_sample_weights: bool = False
    requires_smiles: bool = True  # vs. pre-computed features

    def __init__(
        self,
        config: Union[DictConfig, Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self._is_fitted = False

    @classmethod
    @abstractmethod
    def from_config(cls, config: DictConfig) -> "BaseModel":
        """Factory method to create model from config."""
        pass

    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        df_validation: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        target_cols: list[str] = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
    ) -> np.ndarray:
        """Generate predictions for input data."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk."""
        pass

    def score(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        target_cols: list[str] = None,
        metric: str = "mae",
    ) -> float:
        """Evaluate model on data."""
        # Default implementation
        pass
```

### Phase 2: Model Registry

```python
# src/admet/model/registry.py
from typing import Dict, Type
from admet.model.base import BaseModel


class ModelRegistry:
    """Central registry for all model types."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str):
        """Decorator to register a model class."""
        def decorator(model_cls: Type[BaseModel]):
            cls._models[model_type] = model_cls
            model_cls.model_type = model_type
            return model_cls
        return decorator

    @classmethod
    def get(cls, model_type: str) -> Type[BaseModel]:
        """Get model class by type name."""
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._models.keys())}"
            )
        return cls._models[model_type]

    @classmethod
    def create(cls, config: DictConfig) -> BaseModel:
        """Factory method to create model from config."""
        model_type = config.model.type
        model_cls = cls.get(model_type)
        return model_cls.from_config(config)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(cls._models.keys())
```

### Phase 3: Refactored Config Structure

```python
# src/admet/model/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import MISSING


@dataclass
class DataConfig:
    """Data configuration - shared across all model types."""
    data_dir: str = MISSING
    test_file: Optional[str] = None
    blind_file: Optional[str] = None
    smiles_col: str = "SMILES"
    target_cols: List[str] = field(default_factory=list)
    target_weights: List[float] = field(default_factory=list)
    output_dir: Optional[str] = None


@dataclass
class ChempropModelParams:
    """Chemprop-specific model parameters."""
    depth: int = 5
    message_hidden_dim: int = 600
    num_layers: int = 2
    hidden_dim: int = 600
    dropout: float = 0.1
    batch_norm: bool = True
    ffn_type: str = "regression"
    aggregation: str = "mean"
    trunk_n_layers: int = 2
    trunk_hidden_dim: int = 600
    n_experts: int = 4


@dataclass
class XGBoostModelParams:
    """XGBoost-specific model parameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


@dataclass
class LightGBMModelParams:
    """LightGBM-specific model parameters."""
    n_estimators: int = 100
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


@dataclass
class CatBoostModelParams:
    """CatBoost-specific model parameters."""
    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0


@dataclass
class ModelConfig:
    """Unified model configuration with type discriminator."""
    type: str = "chemprop"  # Model type discriminator
    seed: int = 12345

    # Model-specific params (only one should be populated based on type)
    chemprop: Optional[ChempropModelParams] = None
    xgboost: Optional[XGBoostModelParams] = None
    lightgbm: Optional[LightGBMModelParams] = None
    catboost: Optional[CatBoostModelParams] = None


@dataclass
class TrainingConfig:
    """Training configuration shared across model types."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
```

### Phase 4: Adapt ChempropModel

```python
# src/admet/model/chemprop/model.py
from admet.model.base import BaseModel
from admet.model.registry import ModelRegistry


@ModelRegistry.register("chemprop")
class ChempropModel(BaseModel):
    """Chemprop MPNN model for molecular property prediction."""

    model_type = "chemprop"
    supports_multi_task = True
    supports_sample_weights = False
    requires_smiles = True

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChempropModel":
        """Create ChempropModel from unified config."""
        # Extract chemprop-specific params
        model_params = config.model.chemprop or ChempropModelParams()
        # ... rest of existing from_config logic
```

### Phase 5: Implement Classical Models

```python
# src/admet/model/classical/xgboost_model.py
from admet.model.base import BaseModel
from admet.model.registry import ModelRegistry


@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost model for molecular property prediction."""

    model_type = "xgboost"
    supports_multi_task = True
    supports_sample_weights = True
    requires_smiles = False  # Uses pre-computed features

    def __init__(self, config, output_dir=None):
        super().__init__(config, output_dir)
        self.model = None
        self.feature_generator = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "XGBoostModel":
        model_params = config.model.xgboost or XGBoostModelParams()
        return cls(config)

    def fit(self, df_train, df_validation=None, **kwargs):
        # Generate molecular fingerprints
        X_train = self._generate_features(df_train)
        y_train = df_train[self.target_cols].values

        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = XGBRegressor(**self._get_xgb_params())
        self.model = MultiOutputRegressor(base)
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, df, smiles_col="SMILES"):
        X = self._generate_features(df)
        return self.model.predict(X)
```

## Implementation Guidance

### Objectives
1. Create unified model interface enabling seamless model swapping
2. Maintain backward compatibility with existing chemprop configs
3. Enable easy addition of new model types (chemeleon, tabpfn, catboost)
4. Preserve all existing functionality (MLflow, curriculum learning, etc.)

### Key Tasks

**Phase 1: Foundation (Low Risk)**
- [ ] Create `src/admet/model/base.py` with `BaseModel` abstract class
- [ ] Create `src/admet/model/registry.py` with `ModelRegistry`
- [ ] Create `src/admet/model/config.py` with unified config dataclasses
- [ ] Add tests for base classes

**Phase 2: Chemprop Refactor (Medium Risk)**
- [ ] Update `ChempropModel` to inherit from `BaseModel`
- [ ] Register `ChempropModel` with registry
- [ ] Update config to support both old (flat) and new (nested) format
- [ ] Deprecate `ChempropHyperparams` in favor of `ChempropModelParams`
- [ ] Update ensemble to use factory pattern

**Phase 3: Classical Models (Low Risk)**
- [ ] Implement `XGBoostModel` with `BaseModel` interface
- [ ] Implement `LightGBMModel` with `BaseModel` interface
- [ ] Add feature generation (fingerprints, descriptors)
- [ ] Add tests for classical models

**Phase 4: CLI & Integration (Medium Risk)**
- [ ] Update CLI to use `ModelRegistry.create(config)`
- [ ] Support `model.type` in YAML configs
- [ ] Add config migration utility for existing YAMLs
- [ ] Update documentation

**Phase 5: Advanced Models (Future)**
- [ ] Add `ChemeleonModel` (pre-trained chemprop)
- [ ] Add `TabPFNModel`
- [ ] Add `CatBoostModel`

### Dependencies

- OmegaConf (already used)
- Abstract Base Classes (stdlib)
- No new external dependencies required for Phase 1-3

### Success Criteria

1. `ModelRegistry.create(config)` returns correct model type
2. `model.fit()` and `model.predict()` work uniformly
3. Existing chemprop configs work without modification
4. New model types can be added with <100 lines of code
5. All existing tests pass

## Additional Research: Feature Generation

### RDKit Fingerprint Types (from official documentation)

**Morgan Fingerprints (Circular Fingerprints)**
- Equivalent to ECFP/FCFP fingerprints
- Key parameters: `radius` (2 for ECFP4), `fpSize` (2048 default)
- Feature-based variant: `GetMorganFeatureAtomInvGen()` for FCFP-style
```python
from rdkit.Chem import AllChem
fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
fp = fpgen.GetFingerprint(mol)  # bit vector
fp_count = fpgen.GetSparseCountFingerprint(mol)  # count vector
```

**RDKit (Topological) Fingerprints**
- Daylight-like fingerprint based on hashing molecular subgraphs
- Key parameters: `minPath=1`, `maxPath=7`, `fpSize=2048`, `numBitsPerFeature=2`
```python
fpgen = AllChem.GetRDKitFPGenerator(maxPath=7, fpSize=2048)
fp = fpgen.GetFingerprint(mol)
```

**Atom Pairs and Topological Torsions**
- Atom pairs: Distance-based descriptors
- Topological torsions: 4-atom chains
```python
fpgen_ap = AllChem.GetAtomPairGenerator()
fpgen_tt = AllChem.GetTopologicalTorsionGenerator()
```

**MACCS Keys**
- 166 public keys implemented as SMARTS
```python
from rdkit.Chem import MACCSkeys
fp = MACCSkeys.GenMACCSKeys(mol)
```

### Descriptor Calculation
- `Descriptors.CalcMolDescriptors(mol)` returns dictionary of all 200+ descriptors
- Easy DataFrame integration: `df = pandas.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols])`

### Proposed Feature Generation Module

```python
# src/admet/features/fingerprints.py
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors


@dataclass
class FingerprintConfig:
    """Configuration for molecular fingerprint generation."""
    type: Literal["morgan", "rdkit", "maccs", "atom_pairs", "topological_torsion", "mordred"] = "morgan"

    # Morgan-specific params
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    morgan_use_features: bool = False  # FCFP-style
    morgan_use_chirality: bool = False

    # RDKit-specific params
    rdkit_min_path: int = 1
    rdkit_max_path: int = 7
    rdkit_n_bits: int = 2048

    # Mordred-specific params
    mordred_2d_only: bool = True
    mordred_ignore_3d: bool = True


class FingerprintGenerator:
    """Generate molecular fingerprints from SMILES."""

    SUPPORTED_TYPES = ["morgan", "rdkit", "maccs", "atom_pairs", "topological_torsion", "mordred"]

    def __init__(self, config: FingerprintConfig):
        self.config = config
        self._fp_generator = self._build_generator()

    def _build_generator(self):
        cfg = self.config
        if cfg.type == "morgan":
            inv_gen = AllChem.GetMorganFeatureAtomInvGen() if cfg.morgan_use_features else None
            return AllChem.GetMorganGenerator(
                radius=cfg.morgan_radius,
                fpSize=cfg.morgan_n_bits,
                atomInvariantsGenerator=inv_gen,
                includeChirality=cfg.morgan_use_chirality,
            )
        elif cfg.type == "rdkit":
            return AllChem.GetRDKitFPGenerator(
                minPath=cfg.rdkit_min_path,
                maxPath=cfg.rdkit_max_path,
                fpSize=cfg.rdkit_n_bits,
            )
        elif cfg.type == "atom_pairs":
            return AllChem.GetAtomPairGenerator()
        elif cfg.type == "topological_torsion":
            return AllChem.GetTopologicalTorsionGenerator()
        else:
            return None  # MACCS and mordred handled separately

    def generate(self, smiles_list: List[str]) -> np.ndarray:
        """Generate fingerprints for list of SMILES."""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(self._get_fp_size()))
            else:
                fps.append(self._get_fp(mol))
        return np.vstack(fps)

    def _get_fp(self, mol) -> np.ndarray:
        cfg = self.config
        if cfg.type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        elif cfg.type == "mordred":
            return self._calc_mordred(mol)
        else:
            fp = self._fp_generator.GetFingerprint(mol)
            return np.array(fp)

    def _get_fp_size(self) -> int:
        cfg = self.config
        if cfg.type == "maccs":
            return 167
        elif cfg.type == "mordred":
            return 1613 if cfg.mordred_2d_only else 1826
        elif cfg.type in ["morgan", "rdkit"]:
            return cfg.morgan_n_bits if cfg.type == "morgan" else cfg.rdkit_n_bits
        else:
            return 2048  # Default for atom pairs, torsions
```

## Additional Research: MLflow Integration

### Current MLflow Usage in model.py (lines 1100-1150)

```python
def _init_mlflow(self) -> None:
    if self.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
    mlflow.set_experiment(self.mlflow_experiment_name)
    mlflow.enable_system_metrics_logging()

    # Three modes:
    # Mode 1: Attach to existing run (for ensemble nested runs)
    if self.mlflow_run_id is not None:
        self._mlflow_run = mlflow.start_run(run_id=self.mlflow_run_id)
    # Mode 2: Create nested run under parent
    elif self.mlflow_nested and self.mlflow_parent_run_id is not None:
        self._mlflow_run = mlflow.start_run(
            run_name=self.mlflow_run_name,
            nested=True,
            parent_run_id=self.mlflow_parent_run_id,
        )
    # Mode 3: Start a new standalone run
    else:
        self._mlflow_run = mlflow.start_run(run_name=self.mlflow_run_name)
```

### Key MLflow Integration Points

1. **Run Management**: `_init_mlflow()`, `close()` - handles run lifecycle
2. **Parameter Logging**: `mlflow.log_params()` - logs hyperparameters
3. **Metric Logging**: `mlflow.log_metrics()` - logs training/validation metrics
4. **Artifact Logging**: `mlflow.log_artifact()` - logs files (checkpoints, predictions)
5. **Model Registration**: `MLflowModelCheckpoint` callback for PyTorch Lightning
6. **Nested Runs**: Parent-child relationship for ensemble training

### MLflow Mixin for BaseModel

```python
# src/admet/model/mlflow_mixin.py
from typing import Optional, Dict, Any
import mlflow
from mlflow import MlflowClient


class MLflowMixin:
    """Mixin providing MLflow tracking capabilities to models."""

    mlflow_tracking: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "default"
    mlflow_run_name: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_parent_run_id: Optional[str] = None
    mlflow_nested: bool = False

    _mlflow_run: Optional[Any] = None
    _mlflow_client: Optional[MlflowClient] = None

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        if not self.mlflow_tracking:
            return

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.enable_system_metrics_logging()

        if self.mlflow_run_id is not None:
            self._mlflow_run = mlflow.start_run(run_id=self.mlflow_run_id)
        elif self.mlflow_nested and self.mlflow_parent_run_id is not None:
            self._mlflow_run = mlflow.start_run(
                run_name=self.mlflow_run_name,
                nested=True,
                parent_run_id=self.mlflow_parent_run_id,
            )
            self.mlflow_run_id = self._mlflow_run.info.run_id
        else:
            self._mlflow_run = mlflow.start_run(run_name=self.mlflow_run_name)
            self.mlflow_run_id = self._mlflow_run.info.run_id

        self._mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)

    def _close_mlflow(self) -> None:
        """End MLflow run."""
        if self._mlflow_run is not None:
            mlflow.end_run()
            self._mlflow_run = None

    def _log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if self.mlflow_tracking and self._mlflow_run:
            mlflow.log_params(params)

    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if self.mlflow_tracking and self._mlflow_run:
            mlflow.log_metrics(metrics, step=step)

    def _log_artifact(self, local_path: str) -> None:
        """Log artifact to MLflow."""
        if self.mlflow_tracking and self._mlflow_run:
            mlflow.log_artifact(local_path)
```

## Additional Research: HPO Configuration

### Current HPO Structure (hpo_config.py)

```python
@dataclass
class ParameterSpace:
    """Defines the search range for a hyperparameter."""
    low: float | int
    high: float | int
    log_scale: bool = False
    parameter_type: Literal["float", "int", "choice"] = "float"
    choices: List[Any] = field(default_factory=list)

@dataclass
class SearchSpaceConfig:
    """Chemprop-specific HPO search space."""
    depth: Optional[ParameterSpace] = None
    message_hidden_dim: Optional[ParameterSpace] = None
    ffn_type: Optional[ParameterSpace] = None
    n_experts: Optional[ParameterSpace] = None
    # ... many more chemprop-specific params
```

### Per-Model HPO Search Space Design

```python
# src/admet/hpo/search_spaces.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal


@dataclass
class ParameterSpace:
    """Defines the search range for a hyperparameter."""
    low: float | int = 0
    high: float | int = 1
    log_scale: bool = False
    parameter_type: Literal["float", "int", "choice"] = "float"
    choices: List[Any] = field(default_factory=list)

    def to_ray_tune(self):
        """Convert to Ray Tune search space."""
        from ray import tune
        if self.parameter_type == "choice":
            return tune.choice(self.choices)
        elif self.parameter_type == "int":
            if self.log_scale:
                return tune.lograndint(int(self.low), int(self.high))
            return tune.randint(int(self.low), int(self.high))
        else:
            if self.log_scale:
                return tune.loguniform(self.low, self.high)
            return tune.uniform(self.low, self.high)


@dataclass
class ChempropSearchSpace:
    """Chemprop-specific hyperparameter search space."""
    depth: Optional[ParameterSpace] = None
    message_hidden_dim: Optional[ParameterSpace] = None
    num_layers: Optional[ParameterSpace] = None
    hidden_dim: Optional[ParameterSpace] = None
    dropout: Optional[ParameterSpace] = None
    ffn_type: Optional[ParameterSpace] = None
    n_experts: Optional[ParameterSpace] = None
    init_lr: Optional[ParameterSpace] = None
    max_lr: Optional[ParameterSpace] = None


@dataclass
class XGBoostSearchSpace:
    """XGBoost-specific hyperparameter search space."""
    n_estimators: Optional[ParameterSpace] = None
    max_depth: Optional[ParameterSpace] = None
    learning_rate: Optional[ParameterSpace] = None
    subsample: Optional[ParameterSpace] = None
    colsample_bytree: Optional[ParameterSpace] = None
    reg_alpha: Optional[ParameterSpace] = None
    reg_lambda: Optional[ParameterSpace] = None


@dataclass
class LightGBMSearchSpace:
    """LightGBM-specific hyperparameter search space."""
    n_estimators: Optional[ParameterSpace] = None
    num_leaves: Optional[ParameterSpace] = None
    max_depth: Optional[ParameterSpace] = None
    learning_rate: Optional[ParameterSpace] = None
    subsample: Optional[ParameterSpace] = None


@dataclass
class HPOSearchSpace:
    """Unified HPO search space with model type discriminator."""
    model_type: str = "chemprop"
    chemprop: Optional[ChempropSearchSpace] = None
    xgboost: Optional[XGBoostSearchSpace] = None
    lightgbm: Optional[LightGBMSearchSpace] = None
```

## Updated Implementation Guidance

Based on user clarifications:

### Requirement 1: Config Migration Script
- Create Python script to migrate all 100+ YAML configs in-place
- No deprecation shims - direct migration
- Output: updated YAML files with `model.type` and nested params
- **User Choice**: Option A - modify files in-place

### Requirement 2: Feature Generation Module
- Support Morgan, RDKit, Mordred fingerprints
- Allow kwargs for each via YAML config
- Place in `src/admet/features/fingerprints.py`
- **Mordred**: Use `mordred-community` package from https://github.com/JacksonBurns/mordred-community
- Add as project dependency in pyproject.toml

### Requirement 3: Consistent MLflow Behavior
- Extract MLflow logic into `MLflowMixin`
- All models inherit mixin for consistent tracking
- Log models, register datasets, track params/metrics/artifacts

### Requirement 4: Per-Model Ensemble Support
- `BaseModel.fit()` at per-model level
- `Ensemble` class wraps multiple `BaseModel` instances
- Ensemble parallelization via Ray (existing pattern)
- **User Choice**: Enable mixed model type ensembles (e.g., 3 chemprop + 2 xgboost)

### Requirement 5: Per-Model HPO
- Separate search space dataclass per model type
- HPO config includes `model_type` discriminator
- HPO trainable uses `ModelRegistry.create()`

### Requirement 6: Classical Model Defaults
- Default fingerprint: Morgan with radius=2, n_bits=2048
- All params exposed to YAML config (type, radius, n_bits, etc.)

### Requirement 7: Test Coverage Strategy
- Focus on all test types
- Order: easy → hard (interface compliance → config migration → e2e flows)

## Potential Issues Identified

1. **Config Migration**: Existing 100+ YAML configs need migration strategy
2. **Feature Generation**: Classical models need fingerprint/descriptor generation
3. **MLflow Integration**: Need to ensure MLflow tracking works across all model types
4. **Ensemble Handling**: `ChempropEnsemble` needs to support multiple model types
5. **HPO Compatibility**: Ray Tune search spaces vary per model type
6. **PyTorch Lightning**: Only chemprop uses Lightning; classical models don't need it

## Chemeleon Foundation Model Research

### Overview (from chemprop docs)
- **Model File**: Stored on Zenodo at https://zenodo.org/records/15460715/files/chemeleon_mp.pt
- **Architecture**: Pre-trained BondMessagePassing with 8.7M params, output_dim=2048
- **Usage**: Replace standard `mp` (message passing) in chemprop with pre-trained weights
- **Training Mode**: Can freeze encoder (`mp.eval()`, `mp.requires_grad_(False)`) or continue training

### Chemeleon Initialization Pattern
```python
import torch
from chemprop import featurizers, nn

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
agg = nn.MeanAggregation()
chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
mp.load_state_dict(chemeleon_mp['state_dict'])

# Important: set input_dim=mp.output_dim for FFN
ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=mp.output_dim)
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)
```

### Freezing Strategy
```python
# Freeze encoder (message passing)
mp.eval()
mp.apply(lambda module: module.requires_grad_(False))
```

### Proposed ChemeleonModel Configuration
```yaml
model:
  type: chemeleon
  chemeleon:
    checkpoint_path: "chemeleon_mp.pt"  # or auto-download from Zenodo
    freeze_encoder: true  # Default: frozen
    unfreeze_encoder_epoch: null  # Optional: epoch to unfreeze
    freeze_decoder_initially: false  # Start with decoder trainable
    unfreeze_decoder_epoch: null  # Optional: epoch to start training decoder
    num_layers: 2
    hidden_dim: 300
    dropout: 0.0
```

### Gradual Unfreezing Scheduler Design
```python
@dataclass
class UnfreezeScheduleConfig:
    freeze_encoder: bool = True
    unfreeze_encoder_epoch: Optional[int] = None
    unfreeze_encoder_lr_multiplier: float = 0.1
    freeze_decoder_initially: bool = False
    unfreeze_decoder_epoch: Optional[int] = None


class GradualUnfreezeCallback(pl.Callback):
    """PyTorch Lightning callback for gradual unfreezing of encoder/decoder."""

    def __init__(self, config: UnfreezeScheduleConfig):
        self.config = config
        self._encoder_unfrozen = not config.freeze_encoder
        self._decoder_unfrozen = not config.freeze_decoder_initially

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Unfreeze encoder at specified epoch
        if not self._encoder_unfrozen and self.config.unfreeze_encoder_epoch:
            if epoch >= self.config.unfreeze_encoder_epoch:
                for param in pl_module.message_passing.parameters():
                    param.requires_grad = True
                pl_module.message_passing.train()
                self._encoder_unfrozen = True
                logger.info(f"Unfroze encoder at epoch {epoch}")

        # Unfreeze decoder at specified epoch
        if not self._decoder_unfrozen and self.config.unfreeze_decoder_epoch:
            if epoch >= self.config.unfreeze_decoder_epoch:
                for param in pl_module.predictor.parameters():
                    param.requires_grad = True
                self._decoder_unfrozen = True
                logger.info(f"Unfroze decoder at epoch {epoch}")
```

### ChemeleonModel Implementation Sketch
```python
@ModelRegistry.register("chemeleon")
class ChemeleonModel(BaseModel):
    """Chemeleon foundation model wrapper with freezing support."""

    def __init__(self, config: ChemeleonConfig):
        super().__init__(config)
        self.mp = self._load_pretrained_mp(config.checkpoint_path)
        if config.freeze_encoder:
            self._freeze_encoder()
        self.ffn = self._build_ffn(input_dim=self.mp.output_dim)
        self.unfreeze_callback = GradualUnfreezeCallback(config.unfreeze_schedule)

    def _load_pretrained_mp(self, path: str) -> nn.BondMessagePassing:
        if path == "auto":
            path = self._download_from_zenodo()
        checkpoint = torch.load(path, weights_only=True)
        mp = nn.BondMessagePassing(**checkpoint['hyper_parameters'])
        mp.load_state_dict(checkpoint['state_dict'])
        return mp

    def _freeze_encoder(self):
        self.mp.eval()
        self.mp.apply(lambda m: m.requires_grad_(False))

    def get_trainer_callbacks(self) -> list:
        return [self.unfreeze_callback]
```

## Updated Potential Issues

1. **Config Migration**: Existing 100+ YAML configs need migration strategy
2. **Feature Generation**: Classical models need fingerprint/descriptor generation
3. **MLflow Integration**: Need to ensure MLflow tracking works across all model types
4. **Ensemble Handling**: `ChempropEnsemble` needs to support multiple model types
5. **HPO Compatibility**: Ray Tune search spaces vary per model type
6. **PyTorch Lightning**: Only chemprop uses Lightning; classical models don't need it
7. **Chemeleon Checkpoint**: Need auto-download logic for Zenodo-hosted weights
8. **Gradual Unfreezing**: Scheduler needs to integrate with existing curriculum callbacks
