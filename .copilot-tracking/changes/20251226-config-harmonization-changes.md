# Changes: Unified Configuration and Ensemble Harmonization

**Date**: December 26, 2025
**Branch**: fix/joint_sampling
**Plan**: .copilot-tracking/plans/20251226-config-harmonization-plan.instructions.md
**Status**: ✅ COMPLETE

## Summary

This document tracks all changes for the Unified Configuration and Ensemble Harmonization implementation. The goal is to unify the YAML configuration schema across all model types (Chemprop, Chemeleon, XGBoost, LightGBM, CatBoost) and enable JointSampling for PyTorch-based models while providing clear errors for classical models.

**Final Test Results**: 636 passed, 5 skipped, 5 deselected

---

## Phase 1: Unified Configuration Schema ✅

### Added

- **[src/admet/model/config.py](src/admet/model/config.py)**: Added unified configuration dataclasses
  - `UnifiedModelConfig`: Master configuration containing all model types
  - `UnifiedDataConfig`: Universal data configuration with quality_col for curriculum
  - `UnifiedOptimizationConfig`: Training optimization settings with scheduler support
  - `UnifiedMlflowConfig`: MLflow tracking configuration with `tracking` alias
  - `ModelSection`: Model type discriminator with nested model-specific params
  - `EnsembleSection`: Ensemble configuration (enabled, n_models, aggregation, splits, folds)
  - `RayConfig`: Ray parallelization settings
  - `SchedulerConfig`: Learning rate scheduler configuration
  - Model-specific parameter classes:
    - `ChempropModelParams`, `ChempropOptimizationParams`
    - `ChemeleonModelParams`, `ChemeleonHeadConfig`, `UnfreezeScheduleConfig`
    - `XGBoostModelParams`, `LightGBMModelParams`, `CatBoostModelParams`
  - `FingerprintConfig` and related configs (`MorganFingerprintConfig`, etc.)
  - `validate_model_config()`: Model-type-aware validation function
  - `get_structured_config_for_model_type()`: Factory function for structured configs
  - Constants: `MODEL_TYPES`, `PYTORCH_MODEL_TYPES`, `CLASSICAL_MODEL_TYPES`, `FINGERPRINT_TYPES`

- **[tests/test_unified_config.py](tests/test_unified_config.py)**: Comprehensive test suite (38 tests)
  - `TestUnifiedModelConfig`: Tests for OmegaConf.structured, defaults, YAML merge
  - `TestTrainingStrategyConfigs`: Tests for JointSamplingConfig, CurriculumConfig, etc.
  - `TestConfigValidation`: Tests for classical model restrictions
  - `TestGetStructuredConfigForModelType`: Factory function tests
  - `TestUnifiedDataConfig`, `TestUnifiedMlflowConfig`, `TestUnifiedOptimizationConfig`
  - `TestConfigMergePreservesUserOverrides`: Merge behavior tests
  - `TestEnsembleSection`, `TestRayConfig`: New ensemble/ray config tests
  - `TestModelTypeConstants`: Constant consistency tests

### Modified

- **[src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py)**: Cleaned up and refactored
  - Removed duplicate training strategy config classes
  - Now imports `JointSamplingConfig`, `TaskOversamplingConfig`, `CurriculumConfig`, `TaskAffinityConfig`, `InterTaskAffinityConfig` from `admet.model.config`
  - Reduced from 648 to ~360 lines

### Technical Notes

- OmegaConf does not support Python's `Literal` type annotation in dataclass fields
- Used `str` type with documented valid values and runtime validation constants
- `validate_model_config()` raises `ConfigValidationError` for invalid combinations

---

## Phase 2: Ensemble Configuration ✅

### Added

- **[src/admet/model/config.py](src/admet/model/config.py)**:
  - `EnsembleSection`: Configuration for ensemble training mode
    - `enabled`: Whether to use ensemble training mode
    - `n_models`: Number of models to train
    - `aggregation`: Method to aggregate predictions ("mean" or "median")
    - `use_splits`: Whether to use split/fold directory structure
    - `splits`: Specific split indices to use
    - `folds`: Specific fold indices to use
  - `RayConfig`: Configuration for Ray parallelization
    - `max_parallel`: Maximum models to train in parallel
    - `num_cpus`: CPUs to allocate to Ray
    - `num_gpus`: GPUs to allocate to Ray

### Test Results

- 38 unified config tests pass
- 15 ensemble tests pass (backward compatible)

---

## Phase 3: Adapter Updates (Chemeleon JointSampling) ✅

### Modified

- **[src/admet/model/chemeleon/model.py](src/admet/model/chemeleon/model.py)**:
  - Added imports for `CurriculumState` and `JointSampler`
  - Added instance attributes: `_joint_sampler`, `_curriculum_state`, `_quality_col`
  - Extended `fit()` method signature to accept `quality_labels: list[str] | None`
  - Added `_create_train_dataloader()` method with JointSampler support
    - Creates JointSampler when `joint_sampling.enabled=True`
    - Handles task oversampling alpha
    - Handles curriculum state transitions (using `qualities` and `patience` params)
    - Falls back to standard DataLoader when disabled

### Test Results

- 20 Chemeleon tests pass
- 82 model/sampler tests pass

---

## Phase 4: Config Migration (YAML Compatibility) ✅

### Modified

- **[src/admet/model/config.py](src/admet/model/config.py)**: Extended dataclasses for YAML compatibility
  - `ChemeleonModelParams`: Added `zenodo_id`, `zenodo_filename`, `model_cache_dir`, `unfreeze_encoder`, `head`
  - `ChemeleonHeadConfig`: New dataclass for head configuration (hidden_dims, dropout, activation)
  - `UnfreezeScheduleConfig`: Added `unfreeze_at_epoch` alias
  - `XGBoostModelParams`: Renamed `seed` → `random_state`, added `n_jobs`
  - `LightGBMModelParams`: Renamed `seed` → `random_state`, added `n_jobs`, `verbose`
  - `CatBoostModelParams`: Added `random_strength`, `thread_count`
  - `UnifiedMlflowConfig`: Added `tracking` alias for `enabled`
  - `UnifiedOptimizationConfig`: Added `learning_rate`, `weight_decay`, `scheduler`
  - `SchedulerConfig`: New dataclass for LR scheduler (type, warmup_epochs, step_size, gamma)

### Bug Fix

- Fixed `CurriculumState` constructor call in `_create_train_dataloader()` to use correct parameters (`qualities`, `patience`) instead of invalid (`max_epochs`, `warmup_epochs`)

### Verified Configs

All YAML configs in `configs/4-more-models/` now load successfully:

- `chemprop.yaml` ✅
- `chemeleon.yaml` ✅
- `xgboost.yaml` ✅
- `lightgbm.yaml` ✅
- `catboost.yaml` ✅

---

## Phase 5: Documentation ✅

### Modified

- **[README.md](README.md)**:
  - Updated "Planned" section → "Classical Models" with XGBoost, LightGBM, CatBoost as "Implemented"
  - Added "Configuration System" section explaining unified config schema
  - Added example YAML config snippet
  - Added link to `configs/4-more-models/` for examples

---

## Phase 6: Integration Tests ✅

### Test Results

**Final Count**: 636 passed, 5 skipped, 5 deselected, 29 warnings

Key test suites verified:

- 38 unified config tests ✅
- 78 config + model tests ✅
- 15 ensemble tests ✅
- 141 curriculum/sampler/classical tests ✅
- All other tests ✅

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/admet/model/config.py` | Modified | Added unified config schema, ensemble/ray configs, YAML compatibility |
| `src/admet/model/chemprop/config.py` | Modified | Cleaned up, imports from model.config |
| `src/admet/model/chemeleon/model.py` | Modified | Added JointSampler support |
| `tests/test_unified_config.py` | Added | 38 tests for unified config |
| `README.md` | Modified | Added classical models and config system docs |

---

## API Changes

### New Public API

```python
from admet.model.config import (
    # Master config
    UnifiedModelConfig,
    UnifiedDataConfig,
    UnifiedMlflowConfig,
    UnifiedOptimizationConfig,

    # Model-specific
    ChemeleonModelParams,
    ChemeleonHeadConfig,
    XGBoostModelParams,
    LightGBMModelParams,
    CatBoostModelParams,

    # Ensemble/Ray
    EnsembleSection,
    RayConfig,
    SchedulerConfig,

    # Factory/validation
    get_structured_config_for_model_type,
    validate_model_config,

    # Constants
    MODEL_TYPES,
    PYTORCH_MODEL_TYPES,
    CLASSICAL_MODEL_TYPES,
    FINGERPRINT_TYPES,
)
```

### Breaking Changes

None - all existing APIs preserved with backward compatibility.
