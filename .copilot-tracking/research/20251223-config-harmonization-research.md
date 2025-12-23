# Task Research Notes: Config Harmonization Strategy

## Research Executed

### File Analysis

- [src/admet/model/config.py](src/admet/model/config.py) - Unified base configs (BaseDataConfig, BaseMlflowConfig, BaseModelConfig)
- [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py) - Chemprop-specific configs (695 lines, includes advanced features)
- [src/admet/model/base.py](src/admet/model/base.py) - BaseModel abstract class
- [src/admet/model/registry.py](src/admet/model/registry.py) - ModelRegistry pattern for model creation
- [src/admet/model/classical/base.py](src/admet/model/classical/base.py) - ClassicalModelBase with fingerprint support
- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) - ModelEnsemble with config merge issues

### Current Architecture Analysis

#### Two Parallel Config Systems

1. **Unified Base Configs** ([src/admet/model/config.py](src/admet/model/config.py)):
   - `BaseDataConfig`, `BaseMlflowConfig`, `BaseModelConfig`
   - `FingerprintConfig` for classical models
   - Model-specific params: `XGBoostModelParams`, `LightGBMModelParams`, `CatBoostModelParams`, `ChemeleonModelParams`

2. **Chemprop-Specific Configs** ([src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py)):
   - `DataConfig`, `ModelConfig`, `OptimizationConfig`, `MlflowConfig`
   - Advanced features: `JointSamplingConfig`, `TaskAffinityConfig`, `InterTaskAffinityConfig`
   - `ChempropConfig` - complete training config
   - `EnsembleConfig` - ensemble training config

#### Problem: Ensemble uses ChempropConfig for ALL models

```python
# Line 577-580 in ensemble.py
base_config = OmegaConf.structured(ChempropConfig)  # <-- ALWAYS ChempropConfig
override_config = OmegaConf.create(config_dict)
config = OmegaConf.merge(base_config, override_config)
```

This causes validation errors for non-Chemprop models.

## Key Discoveries

### Features by Generalizability

| Feature | Chemprop | Chemeleon | Classical (XGB/LGB/CB) | Notes |
|---------|----------|-----------|------------------------|-------|
| **DataConfig** | ✅ | ✅ | ✅ | Universal |
| **MlflowConfig** | ✅ | ✅ | ✅ | Universal |
| **RayConfig** | ✅ | ✅ | ✅ | Universal |
| **JointSampling** | ✅ | ✅ | ❌ | Needs PyTorch DataLoader |
| **TaskOversampling** | ✅ | ⚠️ | ❌ | Needs custom DataLoader |
| **Curriculum** | ✅ | ⚠️ | ❌ | Needs training callbacks |
| **InterTaskAffinity** | ✅ | ⚠️ | ❌ | Needs gradient access |
| **TaskAffinity** | ✅ | ❌ | ❌ | Pre-training grouping |
| **OptimizationConfig** | ✅ | ⚠️ | ❌ | LR schedulers differ |
| **FingerprintConfig** | ❌ | ❌ | ✅ | Classical only |
| **UnfreezeSchedule** | ❌ | ✅ | ❌ | Chemeleon only |

### Feature Compatibility Matrix

1. **Universal (all models)**:
   - Data paths, SMILES/target columns
   - MLflow tracking
   - Ray parallelization
   - Seed, batch size (conceptually)

2. **PyTorch-based only (Chemprop, Chemeleon)**:
   - Joint sampling (custom DataLoader)
   - Curriculum learning (training callbacks)
   - Inter-task affinity (gradient computation)
   - Learning rate schedulers

3. **Classical only**:
   - Fingerprint configuration
   - n_estimators, max_depth, etc.

## Recommended Approach: Layered Config Architecture

### Proposed Config Hierarchy

```
BaseConfig (universal)
├── data: BaseDataConfig
├── mlflow: BaseMlflowConfig
├── ray: RayConfig
└── seed: int

PyTorchConfig(BaseConfig)
├── optimization: OptimizationConfig
├── joint_sampling: JointSamplingConfig
├── inter_task_affinity: InterTaskAffinityConfig
└── training: batch_size, epochs, patience

ChempropConfig(PyTorchConfig)
├── model: ChempropModelParams
└── task_affinity: TaskAffinityConfig

ChemeleonConfig(PyTorchConfig)
├── model: ChemeleonModelParams
└── unfreeze_schedule: UnfreezeScheduleConfig

ClassicalConfig(BaseConfig)
├── fingerprint: FingerprintConfig
├── model: XGBoostParams | LightGBMParams | CatBoostParams
└── n_jobs, verbose
```

### Implementation Plan

#### Phase 1: Create Unified Base Config

```python
# src/admet/model/config.py - EXTEND existing

@dataclass
class UnifiedConfig:
    """Universal config for all model types."""
    model_type: str = MISSING
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
    mlflow: BaseMlflowConfig = field(default_factory=BaseMlflowConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    seed: int = 42
```

#### Phase 2: Create PyTorch Training Config

```python
@dataclass
class PyTorchTrainingConfig:
    """Training config for PyTorch-based models."""
    max_epochs: int = 150
    patience: int = 15
    batch_size: int = 128
    num_workers: int = 0
    progress_bar: bool = False

@dataclass
class PyTorchModelConfig(UnifiedConfig):
    """Config for PyTorch-based models (Chemprop, Chemeleon)."""
    training: PyTorchTrainingConfig = field(default_factory=PyTorchTrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)
    inter_task_affinity: InterTaskAffinityConfig = field(default_factory=InterTaskAffinityConfig)
```

#### Phase 3: Fix Ensemble to use model-type-specific configs

```python
# In ensemble.py train_single_model()
if model_type == "chemprop":
    base_config = OmegaConf.structured(ChempropConfig)
elif model_type == "chemeleon":
    base_config = OmegaConf.structured(ChemeleonConfig)
else:
    base_config = OmegaConf.structured(ClassicalConfig)
```

#### Phase 4: Migrate Advanced Features to PyTorch Base

Move from `chemprop/config.py` to shared location:

- `JointSamplingConfig` → `src/admet/model/config.py`
- `InterTaskAffinityConfig` → `src/admet/model/config.py`
- `CurriculumConfig` → `src/admet/model/config.py`

## Implementation Guidance

### Objectives

1. Unify config frontend so all models use consistent YAML structure
2. Keep model-specific params in nested sections (model.chemprop.*, model.xgboost.*)
3. Enable JointSampling and InterTaskAffinity for Chemeleon
4. Fix ensemble to use correct config type per model

### Key Tasks

1. **Create `UnifiedEnsembleConfig`** in `src/admet/model/config.py`
   - Replace `EnsembleConfig` in `chemprop/config.py`
   - Includes all universal sections + model_type discriminator

2. **Create config factory function**

   ```python
   def get_model_config_class(model_type: str) -> type:
       """Return appropriate config class for model type."""
       if model_type == "chemprop":
           return ChempropConfig
       elif model_type == "chemeleon":
           return ChemeleonConfig
       else:
           return ClassicalConfig
   ```

3. **Update `ModelEnsemble.train_all()`**
   - Detect model type early
   - Use correct structured config for merge

4. **Add JointSampling support to Chemeleon**
   - Chemeleon uses PyTorch DataLoader → can use custom sampler
   - Need to implement `JointSampler` integration in Chemeleon training

### Dependencies

- OmegaConf structured configs with dataclass inheritance
- PyTorch DataLoader/Sampler for JointSampling
- MLflow for tracking
- Ray for parallelization

### Success Criteria

1. All 5 configs (chemprop, chemeleon, xgboost, lightgbm, catboost) parse without errors
2. JointSampling works for Chemprop and Chemeleon
3. Classical models train without config validation errors
4. Ensemble training completes for all model types
5. Tests pass for all model types

## File Changes Required

| File | Action | Description |
|------|--------|-------------|
| `src/admet/model/config.py` | EXTEND | Add UnifiedConfig, PyTorchModelConfig, ClassicalConfig |
| `src/admet/model/chemprop/config.py` | REFACTOR | Move JointSampling, InterTaskAffinity to shared config |
| `src/admet/model/chemprop/ensemble.py` | FIX | Use correct config class per model type |
| `src/admet/model/chemeleon/model.py` | EXTEND | Add JointSampling support |
| `configs/4-more-models/*.yaml` | UPDATE | Ensure consistent structure |
