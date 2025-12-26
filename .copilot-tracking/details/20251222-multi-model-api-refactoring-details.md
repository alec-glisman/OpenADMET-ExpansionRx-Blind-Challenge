<!-- markdownlint-disable-file -->

# Task Details: Multi-Model API Refactoring

## Research Reference

**Source Research**: #file:../research/20251222-multi-model-api-refactoring-research.md

---

## Phase 1: Foundation - Base Classes and Registry

### Task 1.1: Create BaseModel abstract class

Create an abstract base class that all model types will inherit from.

- **Files**:
  - `src/admet/model/base.py` - New file with BaseModel ABC
- **Success**:
  - BaseModel class is importable from `admet.model.base`
  - Abstract methods `fit()`, `predict()`, `from_config()` defined
  - Property `model_type` returns string identifier
  - Sklearn compatibility via `get_params()`, `set_params()`
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 200-300) - sklearn BaseEstimator pattern
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 400-500) - AutoGluon AbstractModel
- **Dependencies**: None

**Implementation Specification:**
```python
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
import numpy as np
import pandas as pd
from omegaconf import DictConfig

T = TypeVar("T", bound="BaseModel")

class BaseModel(ABC):
    """Abstract base class for all ADMET models."""

    model_type: str  # Class attribute identifying model type

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: Optional[list[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, smiles: list[str]) -> np.ndarray:
        """Make predictions."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls: type[T], config: DictConfig) -> T:
        """Create model from OmegaConf config."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters (sklearn compatibility)."""
        return {"config": self.config}

    def set_params(self, **params: Any) -> "BaseModel":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

---

### Task 1.2: Create ModelRegistry with decorator-based registration

Implement a registry pattern for dynamic model creation.

- **Files**:
  - `src/admet/model/registry.py` - New file with ModelRegistry
- **Success**:
  - `@ModelRegistry.register("model_type")` decorator works
  - `ModelRegistry.create(config)` returns correct model instance
  - `ModelRegistry.list_models()` returns all registered types
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 500-600) - AutoGluon registry pattern
- **Dependencies**:
  - Task 1.1 (BaseModel class)

**Implementation Specification:**
```python
from typing import Callable, Type
from omegaconf import DictConfig
from admet.model.base import BaseModel

class ModelRegistry:
    """Registry for model types with factory pattern."""

    _registry: dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a model class."""
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            if model_type in cls._registry:
                raise ValueError(f"Model type '{model_type}' already registered")
            cls._registry[model_type] = model_cls
            model_cls.model_type = model_type
            return model_cls
        return decorator

    @classmethod
    def create(cls, config: DictConfig) -> BaseModel:
        """Create model instance from config."""
        model_type = config.model.type
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._registry.keys())}")
        model_cls = cls._registry[model_type]
        return model_cls.from_config(config)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(cls._registry.keys())

    @classmethod
    def get(cls, model_type: str) -> Type[BaseModel]:
        """Get model class by type."""
        return cls._registry[model_type]
```

---

### Task 1.3: Create MLflowMixin for consistent tracking

Extract MLflow tracking logic into a reusable mixin.

- **Files**:
  - `src/admet/model/mlflow_mixin.py` - New file with MLflowMixin
- **Success**:
  - `MLflowMixin` provides `log_params()`, `log_metrics()`, `log_model()`
  - Consistent run naming and artifact logging
  - Works with nested runs for ensemble members
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 700-800) - MLflow integration patterns
  - `src/admet/model/chemprop/model.py` (Lines 200-350) - Current MLflow implementation
- **Dependencies**:
  - Task 1.1 (BaseModel class)

**Implementation Specification:**
```python
import mlflow
from omegaconf import OmegaConf, DictConfig
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowMixin:
    """Mixin providing consistent MLflow tracking."""

    config: DictConfig  # Expected from BaseModel
    _mlflow_run_id: Optional[str] = None

    def init_mlflow(self, run_name: Optional[str] = None, nested: bool = False) -> None:
        """Initialize MLflow run."""
        mlflow_config = self.config.get("mlflow", {})
        if not mlflow_config.get("enabled", True):
            return

        experiment_name = mlflow_config.get("experiment_name", "admet")
        mlflow.set_experiment(experiment_name)

        run_name = run_name or f"{self.model_type}_{mlflow_config.get('run_name', 'run')}"
        self._mlflow_run = mlflow.start_run(run_name=run_name, nested=nested)
        self._mlflow_run_id = self._mlflow_run.info.run_id

    def log_params_from_config(self) -> None:
        """Log all config parameters."""
        if not self._mlflow_run_id:
            return
        flat_config = OmegaConf.to_container(self.config, resolve=True)
        mlflow.log_params(self._flatten_dict(flat_config))

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self._mlflow_run_id:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log model artifact."""
        if not self._mlflow_run_id:
            return
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model)

    def end_mlflow(self) -> None:
        """End MLflow run."""
        if self._mlflow_run_id:
            mlflow.end_run()

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dict for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowMixin._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

---

### Task 1.4: Create unified config dataclasses

Create base config classes that all model configs inherit from.

- **Files**:
  - `src/admet/model/config.py` - New file with BaseModelConfig and unified structure
- **Success**:
  - `BaseModelConfig` with common fields (type, mlflow, data)
  - Per-model config dataclasses inherit from base
  - OmegaConf structured config support
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 600-700) - Config structure proposal
  - `src/admet/model/chemprop/config.py` - Current config patterns
- **Dependencies**:
  - Task 1.1

**Implementation Specification:**
```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from omegaconf import MISSING

@dataclass
class DataConfig:
    """Common data configuration."""
    train_path: str = MISSING
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    smiles_column: str = "smiles"
    target_columns: list[str] = field(default_factory=list)

@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    enabled: bool = True
    experiment_name: str = "admet"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    log_model: bool = True

@dataclass
class BaseModelConfig:
    """Base configuration for all models."""
    type: str = MISSING  # Model type discriminator
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

@dataclass
class ChempropModelConfig:
    """Chemprop-specific model configuration."""
    message_passing_layers: int = 3
    hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    aggregation: str = "mean"
    # ... other chemprop-specific fields

@dataclass
class ChempropConfig(BaseModelConfig):
    """Full chemprop configuration."""
    type: str = "chemprop"
    model: ChempropModelConfig = field(default_factory=ChempropModelConfig)
    # ... training config, etc.
```

---

### Task 1.5: Add unit tests for base classes

Create comprehensive tests for BaseModel, ModelRegistry, and MLflowMixin.

- **Files**:
  - `tests/test_model_base.py` - New test file
- **Success**:
  - Tests verify BaseModel interface requirements
  - Tests verify ModelRegistry registration and creation
  - Tests verify MLflowMixin tracking behavior
- **Research References**:
  - `tests/test_chemprop_model_utils.py` - Existing test patterns
- **Dependencies**:
  - Tasks 1.1, 1.2, 1.3, 1.4

---

## Phase 2: Feature Generation Module

### Task 2.1: Create FingerprintConfig dataclass

Define configuration for all supported fingerprint types.

- **Files**:
  - `src/admet/features/config.py` - New file
- **Success**:
  - `FingerprintConfig` supports morgan, rdkit, maccs, mordred
  - All fingerprint-specific params configurable (radius, bits, etc.)
  - Default: Morgan with radius=2, n_bits=2048
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 850-950) - RDKit fingerprint options
- **Dependencies**: None

**Implementation Specification:**
```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class MorganFingerprintConfig:
    """Morgan fingerprint configuration."""
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = False
    use_bond_types: bool = True
    use_features: bool = False

@dataclass
class RDKitFingerprintConfig:
    """RDKit fingerprint configuration."""
    min_path: int = 1
    max_path: int = 7
    n_bits: int = 2048
    branched_paths: bool = True

@dataclass
class MACCSConfig:
    """MACCS keys configuration."""
    pass  # MACCS has fixed 167 keys

@dataclass
class MordredConfig:
    """Mordred descriptor configuration."""
    ignore_3d: bool = True
    normalize: bool = True

@dataclass
class FingerprintConfig:
    """Feature generation configuration."""
    type: Literal["morgan", "rdkit", "maccs", "mordred"] = "morgan"
    morgan: MorganFingerprintConfig = field(default_factory=MorganFingerprintConfig)
    rdkit: RDKitFingerprintConfig = field(default_factory=RDKitFingerprintConfig)
    maccs: MACCSConfig = field(default_factory=MACCSConfig)
    mordred: MordredConfig = field(default_factory=MordredConfig)
```

---

### Task 2.2: Implement FingerprintGenerator class

Create unified fingerprint generation supporting all types.

- **Files**:
  - `src/admet/features/fingerprints.py` - New file
- **Success**:
  - `FingerprintGenerator.generate(smiles)` returns feature array
  - Supports batch processing
  - Handles invalid SMILES gracefully
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 850-950) - RDKit examples
  - mordred-community documentation
- **Dependencies**:
  - Task 2.1 (FingerprintConfig)

**Implementation Specification:**
```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, MACCSkeys
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FingerprintGenerator:
    """Unified fingerprint/descriptor generator."""

    def __init__(self, config: FingerprintConfig):
        self.config = config
        self._fp_func = self._get_fingerprint_function()

    def _get_fingerprint_function(self):
        """Get appropriate fingerprint function based on config."""
        fp_type = self.config.type
        if fp_type == "morgan":
            return self._morgan_fp
        elif fp_type == "rdkit":
            return self._rdkit_fp
        elif fp_type == "maccs":
            return self._maccs_fp
        elif fp_type == "mordred":
            return self._mordred_fp
        raise ValueError(f"Unknown fingerprint type: {fp_type}")

    def generate(self, smiles: list[str]) -> np.ndarray:
        """Generate fingerprints for list of SMILES."""
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smi}")
                fps.append(self._get_null_fp())
            else:
                fps.append(self._fp_func(mol))
        return np.array(fps)

    def _morgan_fp(self, mol) -> np.ndarray:
        cfg = self.config.morgan
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=cfg.radius, nBits=cfg.n_bits,
            useChirality=cfg.use_chirality,
            useBondTypes=cfg.use_bond_types,
            useFeatures=cfg.use_features
        )
        return np.array(fp)

    def _rdkit_fp(self, mol) -> np.ndarray:
        cfg = self.config.rdkit
        fp = Chem.RDKFingerprint(
            mol, minPath=cfg.min_path, maxPath=cfg.max_path,
            fpSize=cfg.n_bits, branchedPaths=cfg.branched_paths
        )
        return np.array(fp)

    def _maccs_fp(self, mol) -> np.ndarray:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)

    def _mordred_fp(self, mol) -> np.ndarray:
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=self.config.mordred.ignore_3d)
        result = calc(mol)
        arr = np.array([float(x) if not isinstance(x, mordred.error.Error) else np.nan for x in result])
        if self.config.mordred.normalize:
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _get_null_fp(self) -> np.ndarray:
        """Return null fingerprint for invalid molecules."""
        if self.config.type == "morgan":
            return np.zeros(self.config.morgan.n_bits)
        elif self.config.type == "rdkit":
            return np.zeros(self.config.rdkit.n_bits)
        elif self.config.type == "maccs":
            return np.zeros(167)
        elif self.config.type == "mordred":
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            return np.zeros(len(calc.descriptors))
```

---

### Task 2.3: Add mordred-community dependency

Add mordred-community package to project dependencies.

- **Files**:
  - `pyproject.toml` - Add mordred-community to dependencies
- **Success**:
  - `mordred-community` listed in dependencies
  - Package installable via `pip install -e .`
- **Research References**:
  - https://github.com/JacksonBurns/mordred-community
- **Dependencies**: None

**Implementation Specification:**
```toml
# In pyproject.toml [project.dependencies] section, add:
"mordred-community>=2.0.0",
```

---

### Task 2.4: Add unit tests for fingerprint generation

Test all fingerprint types with various SMILES inputs.

- **Files**:
  - `tests/test_fingerprints.py` - New test file
- **Success**:
  - Tests verify each fingerprint type generates correct shape
  - Tests verify invalid SMILES handling
  - Tests verify config options work correctly
- **Dependencies**:
  - Tasks 2.1, 2.2, 2.3

---

## Phase 3: Config Migration

### Task 3.1: Create config migration script

Create Python script to migrate all YAML configs to new structure.

- **Files**:
  - `scripts/migrate_configs.py` - New migration script
- **Success**:
  - Script reads existing YAML configs
  - Transforms to new nested structure with `model.type`
  - Writes back to same file (in-place)
  - Preserves comments where possible
- **Research References**:
  - `configs/` - Existing config structure
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 950-1000) - Migration requirements
- **Dependencies**:
  - Task 1.4 (unified config structure)

**Implementation Specification:**
```python
#!/usr/bin/env python
"""Migrate YAML configs to new multi-model structure."""
import argparse
from pathlib import Path
import ruamel.yaml

def migrate_config(config_path: Path) -> None:
    """Migrate a single config file."""
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True

    with open(config_path) as f:
        config = yaml.load(f)

    # Skip if already migrated
    if config.get("model", {}).get("type"):
        return

    # Transform structure
    model_section = config.pop("model", {})
    new_config = {
        "model": {
            "type": "chemprop",  # Default for existing configs
            "chemprop": model_section,
        },
        **config
    }

    # Write back
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for yaml_file in args.config_dir.rglob("*.yaml"):
        if args.dry_run:
            print(f"Would migrate: {yaml_file}")
        else:
            migrate_config(yaml_file)
            print(f"Migrated: {yaml_file}")

if __name__ == "__main__":
    main()
```

---

### Task 3.2: Run migration on all existing configs

Execute migration script on configs directory.

- **Files**:
  - `configs/**/*.yaml` - All YAML files migrated
- **Success**:
  - All configs have `model.type` field
  - Existing model params nested under `model.chemprop`
  - No data loss
- **Dependencies**:
  - Task 3.1

---

### Task 3.3: Validate migrated configs

Verify all migrated configs are valid.

- **Files**:
  - `scripts/validate_configs.py` - New validation script
- **Success**:
  - All configs loadable with OmegaConf
  - All configs pass schema validation
  - Test loading each with `ModelRegistry.create()`
- **Dependencies**:
  - Task 3.2

---

## Phase 4: Refactor ChempropModel

### Task 4.1: Update ChempropModel to inherit from BaseModel

Modify ChempropModel to implement BaseModel interface.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Modify existing
- **Success**:
  - ChempropModel inherits from BaseModel
  - `fit()`, `predict()`, `from_config()` match interface
  - Existing functionality preserved
- **Research References**:
  - `src/admet/model/chemprop/model.py` (current implementation)
- **Dependencies**:
  - Task 1.1 (BaseModel)

**Key Changes:**
```python
from admet.model.base import BaseModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

@ModelRegistry.register("chemprop")
class ChempropModel(BaseModel, MLflowMixin):
    """Chemprop model with BaseModel interface."""

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: Optional[list[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "ChempropModel":
        # Existing fit logic adapted to new interface
        ...

    def predict(self, smiles: list[str]) -> np.ndarray:
        # Existing predict logic
        ...

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChempropModel":
        # Extract chemprop-specific config from nested structure
        chemprop_config = config.model.chemprop
        return cls(config)
```

---

### Task 4.2: Register ChempropModel with ModelRegistry

Add registration decorator to ChempropModel.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Add decorator
- **Success**:
  - `@ModelRegistry.register("chemprop")` decorator applied
  - `ModelRegistry.create(config)` returns ChempropModel for type="chemprop"
- **Dependencies**:
  - Tasks 1.2, 4.1

---

### Task 4.3: Update ChempropModel.from_config for new config structure

Update from_config to read nested config structure.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Modify from_config
- **Success**:
  - Reads config from `config.model.chemprop` section
  - Backward compatible with existing tests
- **Dependencies**:
  - Task 4.1

---

### Task 4.4: Add tests for refactored ChempropModel

Verify ChempropModel works with new structure.

- **Files**:
  - `tests/test_chemprop_refactor.py` - New test file
- **Success**:
  - Tests verify fit/predict with new config structure
  - Tests verify registry registration
  - Existing tests still pass
- **Dependencies**:
  - Tasks 4.1, 4.2, 4.3

---

## Phase 5: Implement ChemeleonModel

### Task 5.1: Create ChemeleonConfig with freezing options

Define configuration for Chemeleon foundation model.

- **Files**:
  - `src/admet/model/chemeleon/config.py` - New file
- **Success**:
  - `ChemeleonConfig` with checkpoint_path, freeze_encoder, unfreeze_schedule
  - `UnfreezeScheduleConfig` for gradual unfreezing
  - OmegaConf structured config support
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 1005-1100) - Chemeleon research
- **Dependencies**:
  - Task 1.4 (BaseModelConfig)

**Implementation Specification:**
```python
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from admet.model.config import BaseModelConfig

@dataclass
class UnfreezeScheduleConfig:
    """Configuration for gradual unfreezing of encoder/decoder."""
    freeze_encoder: bool = True
    unfreeze_encoder_epoch: Optional[int] = None
    unfreeze_encoder_lr_multiplier: float = 0.1
    freeze_decoder_initially: bool = False
    unfreeze_decoder_epoch: Optional[int] = None

@dataclass
class ChemeleonModelConfig:
    """Chemeleon-specific model configuration."""
    checkpoint_path: str = "auto"  # "auto" downloads from Zenodo
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0

@dataclass
class ChemeleonConfig(BaseModelConfig):
    """Full Chemeleon configuration."""
    type: str = "chemeleon"
    model: ChemeleonModelConfig = field(default_factory=ChemeleonModelConfig)
```

---

### Task 5.2: Implement GradualUnfreezeCallback for encoder/decoder

Create PyTorch Lightning callback for scheduled unfreezing.

- **Files**:
  - `src/admet/model/chemeleon/callbacks.py` - New file
- **Success**:
  - `GradualUnfreezeCallback` unfreezes at specified epochs
  - Supports separate encoder/decoder schedules
  - Logs unfreezing events
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 1050-1080) - Callback design
  - `src/admet/model/chemprop/curriculum.py` - Existing callback patterns
- **Dependencies**:
  - Task 5.1 (UnfreezeScheduleConfig)

**Implementation Specification:**
```python
import logging
import pytorch_lightning as pl
from admet.model.chemeleon.config import UnfreezeScheduleConfig

logger = logging.getLogger(__name__)

class GradualUnfreezeCallback(pl.Callback):
    """Callback for gradual unfreezing of encoder and decoder."""

    def __init__(self, config: UnfreezeScheduleConfig):
        super().__init__()
        self.config = config
        self._encoder_unfrozen = not config.freeze_encoder
        self._decoder_unfrozen = not config.freeze_decoder_initially

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch

        # Unfreeze encoder at specified epoch
        if not self._encoder_unfrozen and self.config.unfreeze_encoder_epoch is not None:
            if epoch >= self.config.unfreeze_encoder_epoch:
                self._unfreeze_encoder(pl_module)
                self._encoder_unfrozen = True
                logger.info(f"Unfroze encoder at epoch {epoch}")

        # Unfreeze decoder at specified epoch
        if not self._decoder_unfrozen and self.config.unfreeze_decoder_epoch is not None:
            if epoch >= self.config.unfreeze_decoder_epoch:
                self._unfreeze_decoder(pl_module)
                self._decoder_unfrozen = True
                logger.info(f"Unfroze decoder at epoch {epoch}")

    def _unfreeze_encoder(self, pl_module: pl.LightningModule) -> None:
        """Unfreeze message passing encoder."""
        if hasattr(pl_module, "message_passing"):
            for param in pl_module.message_passing.parameters():
                param.requires_grad = True
            pl_module.message_passing.train()

    def _unfreeze_decoder(self, pl_module: pl.LightningModule) -> None:
        """Unfreeze predictor/FFN decoder."""
        if hasattr(pl_module, "predictor"):
            for param in pl_module.predictor.parameters():
                param.requires_grad = True
```

---

### Task 5.3: Implement ChemeleonModel with pretrained loading

Create ChemeleonModel class inheriting from BaseModel.

- **Files**:
  - `src/admet/model/chemeleon/model.py` - New file
- **Success**:
  - `ChemeleonModel` loads pretrained BondMessagePassing
  - Freezes encoder by default
  - Integrates GradualUnfreezeCallback
  - Implements fit/predict interface
- **Research References**:
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 1015-1040) - Chemeleon init pattern
  - chemprop.readthedocs.io Chemeleon documentation
- **Dependencies**:
  - Tasks 5.1, 5.2

**Implementation Specification:**
```python
import torch
import logging
from typing import Optional
import numpy as np
from omegaconf import DictConfig
from chemprop import featurizers, nn, models
from admet.model.base import BaseModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry
from admet.model.chemeleon.config import ChemeleonConfig
from admet.model.chemeleon.callbacks import GradualUnfreezeCallback

logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

@ModelRegistry.register("chemeleon")
class ChemeleonModel(BaseModel, MLflowMixin):
    """Chemeleon foundation model wrapper."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        model_cfg = config.model.chemeleon

        self.mp = self._load_pretrained_mp(model_cfg.checkpoint_path)
        if model_cfg.unfreeze_schedule.freeze_encoder:
            self._freeze_encoder()

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.agg = nn.MeanAggregation()
        self.ffn = nn.RegressionFFN(
            input_dim=self.mp.output_dim,
            hidden_dim=model_cfg.ffn_hidden_dim,
            n_layers=model_cfg.ffn_num_layers,
            dropout=model_cfg.dropout,
        )

        self.model = models.MPNN(
            self.mp, self.agg, self.ffn, batch_norm=False
        )

        self.unfreeze_callback = GradualUnfreezeCallback(model_cfg.unfreeze_schedule)

    def _load_pretrained_mp(self, path: str) -> nn.BondMessagePassing:
        """Load pretrained message passing from checkpoint."""
        if path == "auto":
            path = self._download_from_zenodo()

        checkpoint = torch.load(path, weights_only=True)
        mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
        mp.load_state_dict(checkpoint["state_dict"])
        return mp

    def _freeze_encoder(self) -> None:
        """Freeze message passing encoder."""
        self.mp.eval()
        self.mp.apply(lambda m: m.requires_grad_(False))
        logger.info("Froze Chemeleon encoder")

    def _download_from_zenodo(self) -> str:
        """Download checkpoint from Zenodo."""
        import urllib.request
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "admet" / "chemeleon"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / "chemeleon_mp.pt"

        if not checkpoint_path.exists():
            logger.info(f"Downloading Chemeleon checkpoint from {ZENODO_URL}")
            urllib.request.urlretrieve(ZENODO_URL, checkpoint_path)

        return str(checkpoint_path)

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: Optional[list[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "ChemeleonModel":
        """Train the model."""
        # Implementation similar to ChempropModel
        # Uses PyTorch Lightning trainer with GradualUnfreezeCallback
        ...
        self._fitted = True
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Make predictions."""
        ...

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChemeleonModel":
        """Create model from config."""
        return cls(config)

    def get_trainer_callbacks(self) -> list:
        """Get PyTorch Lightning callbacks."""
        return [self.unfreeze_callback]
```

---

### Task 5.4: Add Zenodo auto-download for chemeleon_mp.pt

Implement automatic download of Chemeleon weights.

- **Files**:
  - `src/admet/model/chemeleon/model.py` - Add download logic (included in 5.3)
- **Success**:
  - `checkpoint_path="auto"` downloads from Zenodo
  - Caches to `~/.cache/admet/chemeleon/`
  - Shows progress for large download
- **Research References**:
  - https://zenodo.org/records/15460715/files/chemeleon_mp.pt
- **Dependencies**:
  - Task 5.3

---

### Task 5.5: Add tests for ChemeleonModel

Test ChemeleonModel with frozen/unfrozen encoder scenarios.

- **Files**:
  - `tests/test_chemeleon_model.py` - New test file
- **Success**:
  - Tests verify pretrained loading
  - Tests verify encoder freezing
  - Tests verify gradual unfreezing callback
  - Tests verify fit/predict interface
- **Dependencies**:
  - Tasks 5.1-5.4

---

## Phase 6: Implement Classical Models

### Task 6.1: Implement XGBoostModel with BaseModel interface

Create XGBoostModel inheriting from BaseModel.

- **Files**:
  - `src/admet/model/classical/xgboost_model.py` - New file
- **Success**:
  - `XGBoostModel` implements fit/predict/from_config
  - Uses FingerprintGenerator for feature extraction
  - Registered with ModelRegistry as "xgboost"
- **Research References**:
  - `src/admet/model/classical/models.py` - Existing XGBoost wrapper
  - #file:../research/20251222-multi-model-api-refactoring-research.md (Lines 300-400) - sklearn patterns
- **Dependencies**:
  - Tasks 1.1, 1.2, 2.2

**Implementation Specification:**
```python
import numpy as np
from typing import Optional
import xgboost as xgb
from omegaconf import DictConfig
from admet.model.base import BaseModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry
from admet.features.fingerprints import FingerprintGenerator

@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel, MLflowMixin):
    """XGBoost model with BaseModel interface."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        model_cfg = config.model.xgboost

        self.fp_generator = FingerprintGenerator(config.model.fingerprint)
        self.model = xgb.XGBRegressor(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.max_depth,
            learning_rate=model_cfg.learning_rate,
            subsample=model_cfg.subsample,
            colsample_bytree=model_cfg.colsample_bytree,
            random_state=model_cfg.get("seed", 42),
        )

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: Optional[list[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        X = self.fp_generator.generate(smiles)

        eval_set = None
        if val_smiles is not None and val_y is not None:
            X_val = self.fp_generator.generate(val_smiles)
            eval_set = [(X_val, val_y)]

        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self._fitted = True
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        X = self.fp_generator.generate(smiles)
        return self.model.predict(X)

    @classmethod
    def from_config(cls, config: DictConfig) -> "XGBoostModel":
        return cls(config)
```

---

### Task 6.2: Implement LightGBMModel with BaseModel interface

Create LightGBMModel inheriting from BaseModel.

- **Files**:
  - `src/admet/model/classical/lightgbm_model.py` - New file
- **Success**:
  - `LightGBMModel` implements fit/predict/from_config
  - Uses FingerprintGenerator for feature extraction
  - Registered with ModelRegistry as "lightgbm"
- **Dependencies**:
  - Tasks 1.1, 1.2, 2.2

---

### Task 6.3: Implement CatBoostModel with BaseModel interface

Create CatBoostModel inheriting from BaseModel.

- **Files**:
  - `src/admet/model/classical/catboost_model.py` - New file
- **Success**:
  - `CatBoostModel` implements fit/predict/from_config
  - Uses FingerprintGenerator for feature extraction
  - Registered with ModelRegistry as "catboost"
- **Dependencies**:
  - Tasks 1.1, 1.2, 2.2

---

### Task 6.4: Add tests for classical models

Test all classical models with fingerprint generation.

- **Files**:
  - `tests/test_classical_models.py` - New test file
- **Success**:
  - Tests verify each model type implements interface
  - Tests verify fingerprint generation integration
  - Tests verify MLflow tracking
- **Dependencies**:
  - Tasks 6.1-6.3

---

## Phase 7: Update Ensemble and CLI

### Task 7.1: Create generic Ensemble class supporting mixed model types

Refactor ensemble to support any model type via ModelRegistry.

- **Files**:
  - `src/admet/model/ensemble.py` - New file or refactor existing
- **Success**:
  - `Ensemble` class works with any BaseModel subclass
  - Supports mixed model types (e.g., 3 chemprop + 2 xgboost)
  - Maintains Ray parallelization for training
- **Research References**:
  - `src/admet/model/chemprop/ensemble.py` - Current implementation
- **Dependencies**:
  - Phase 1-6 complete

**Implementation Specification:**
```python
from typing import Optional
import numpy as np
from omegaconf import DictConfig
from admet.model.base import BaseModel
from admet.model.registry import ModelRegistry
import ray

class Ensemble:
    """Generic ensemble supporting any model type."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.models: list[BaseModel] = []

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: Optional[list[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "Ensemble":
        """Train all ensemble members."""
        ensemble_config = self.config.ensemble

        # Create model configs for each member
        model_configs = self._create_member_configs(ensemble_config)

        # Train in parallel with Ray
        if ensemble_config.get("parallel", True):
            self.models = self._train_parallel(model_configs, smiles, y, val_smiles, val_y)
        else:
            self.models = self._train_sequential(model_configs, smiles, y, val_smiles, val_y)

        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Aggregate predictions from all members."""
        predictions = [model.predict(smiles) for model in self.models]
        return np.mean(predictions, axis=0)

    def _create_member_configs(self, ensemble_config) -> list[DictConfig]:
        """Create config for each ensemble member."""
        configs = []
        for member in ensemble_config.members:
            # Member can specify model type or inherit from parent
            configs.append(member)
        return configs
```

---

### Task 7.2: Update CLI to use ModelRegistry.create()

Modify CLI commands to use ModelRegistry for model creation.

- **Files**:
  - `src/admet/cli/model.py` - Modify existing
- **Success**:
  - `admet train --config config.yaml` uses ModelRegistry
  - Supports all registered model types
  - Backward compatible with existing commands
- **Dependencies**:
  - Task 7.1

---

### Task 7.3: Add integration tests for ensemble and CLI

Test ensemble training with mixed model types.

- **Files**:
  - `tests/test_ensemble_mixed.py` - New test file
- **Success**:
  - Tests verify mixed model ensembles work
  - Tests verify CLI commands
- **Dependencies**:
  - Tasks 7.1, 7.2

---

## Phase 8: Per-Model HPO Configuration

### Task 8.1: Create per-model HPO search space dataclasses

Define search spaces for each model type.

- **Files**:
  - `src/admet/model/hpo/search_spaces.py` - New file
- **Success**:
  - `ChempropSearchSpace`, `XGBoostSearchSpace`, etc.
  - Ray Tune compatible tune.* syntax
  - Model-specific hyperparameter ranges
- **Research References**:
  - `src/admet/model/chemprop/hpo_config.py` - Current HPO config
- **Dependencies**:
  - Phase 1-7 complete

**Implementation Specification:**
```python
from dataclasses import dataclass
from ray import tune

@dataclass
class ChempropSearchSpace:
    """Chemprop HPO search space."""
    hidden_dim: any = tune.choice([100, 200, 300, 500])
    ffn_hidden_dim: any = tune.choice([100, 200, 300, 500])
    ffn_num_layers: any = tune.choice([1, 2, 3])
    dropout: any = tune.uniform(0.0, 0.5)
    learning_rate: any = tune.loguniform(1e-5, 1e-2)

@dataclass
class XGBoostSearchSpace:
    """XGBoost HPO search space."""
    n_estimators: any = tune.choice([100, 200, 500, 1000])
    max_depth: any = tune.choice([3, 5, 7, 9])
    learning_rate: any = tune.loguniform(0.01, 0.3)
    subsample: any = tune.uniform(0.6, 1.0)
    colsample_bytree: any = tune.uniform(0.6, 1.0)

@dataclass
class ChemeleonSearchSpace:
    """Chemeleon HPO search space."""
    ffn_hidden_dim: any = tune.choice([100, 200, 300, 500])
    ffn_num_layers: any = tune.choice([1, 2, 3])
    dropout: any = tune.uniform(0.0, 0.5)
    unfreeze_encoder_epoch: any = tune.choice([None, 5, 10, 20])
```

---

### Task 8.2: Update HPO trainable to use ModelRegistry

Modify HPO trainable function to create models via registry.

- **Files**:
  - `src/admet/model/hpo/trainable.py` - Modify existing
- **Success**:
  - Trainable uses `ModelRegistry.create()`
  - Search space selected based on model type
  - Works with all registered models
- **Dependencies**:
  - Task 8.1

---

### Task 8.3: Add tests for HPO configuration

Test HPO with different model types.

- **Files**:
  - `tests/test_hpo_multi_model.py` - New test file
- **Success**:
  - Tests verify each model type has valid search space
  - Tests verify HPO trainable works with registry
- **Dependencies**:
  - Tasks 8.1, 8.2

---

## Phase 9: End-to-End Testing and Documentation

### Task 9.1: Add end-to-end training tests for all model types

Create comprehensive e2e tests.

- **Files**:
  - `tests/test_e2e_training.py` - New test file
- **Success**:
  - E2E test for each model type: chemprop, chemeleon, xgboost, lightgbm, catboost
  - Tests verify complete workflow: config → train → predict
  - Tests run on small dataset for CI speed
- **Dependencies**:
  - Phase 1-8 complete

---

### Task 9.2: Update documentation and README

Update project documentation for multi-model support.

- **Files**:
  - `README.md` - Update usage examples
  - `docs/guide/models.rst` - New documentation page
- **Success**:
  - Documentation covers all model types
  - Examples show config structure for each type
  - Migration guide for existing users
- **Dependencies**:
  - All phases complete

---

## Dependencies

- OmegaConf (existing)
- mordred-community (new - for Mordred descriptors)
- RDKit (existing - for fingerprints)
- XGBoost, LightGBM, CatBoost (existing)
- Ray (existing - for ensemble parallelization)
- MLflow (existing)
- PyTorch Lightning (existing - for chemprop/chemeleon)
- ruamel.yaml (for config migration script)

## Success Criteria

- All model types implement identical fit(), predict(), from_config() interface
- ModelRegistry.create(config) returns correct model type based on config.model.type
- All 100+ existing YAML configs migrated to new nested structure
- FingerprintGenerator supports Morgan, RDKit, and Mordred with configurable params
- MLflow tracking works consistently for all model types
- ChemeleonModel loads pretrained weights with frozen encoder by default
- GradualUnfreezeCallback correctly unfreezes encoder/decoder at specified epochs
- Mixed model type ensembles train and predict successfully
- Per-model HPO search spaces work with Ray Tune
- All existing tests continue to pass
- New tests cover interface compliance → config migration → e2e flows
