<!-- markdownlint-disable-file -->

# Release Changes: Multi-Model API Refactoring

**Related Plan**: 20251222-multi-model-api-refactoring-plan.instructions.md
**Implementation Date**: 2025-12-22

## Summary

Refactoring the config and API to support multiple model types (chemprop, chemeleon, xgboost, catboost, lightgbm) with a unified BaseModel interface, ModelRegistry, feature generation module, and consistent MLflow integration.

## Changes

### Phase 1: Core Architecture ✅

#### Added

- `src/admet/model/base.py` - BaseModel abstract class with unified fit/predict interface, sklearn compatibility (get_params/set_params), and is_fitted property
- `src/admet/model/registry.py` - ModelRegistry with @register decorator pattern, create() factory, list_models(), get(), is_registered(), clear() methods
- `src/admet/model/mlflow_mixin.py` - MLflowMixin for consistent MLflow tracking across all model types: init_mlflow(), log_params_from_config(), log_metrics(), log_metric(), log_artifact(), log_model(), end_mlflow()
- `src/admet/model/config.py` - Unified configuration dataclasses:
  - BaseDataConfig, BaseMlflowConfig, BaseModelConfig (base classes)
  - FingerprintConfig with MorganConfig, RDKitConfig, MACCSConfig, MordredConfig
  - ChempropModelParams, ChempropOptimizationParams
  - ChemeleonModelParams, UnfreezeScheduleConfig
  - XGBoostModelParams, LightGBMModelParams, CatBoostModelParams
- `tests/test_model_base.py` - 25 unit tests for BaseModel, ModelRegistry, MLflowMixin, and integration tests

#### Modified

- `src/admet/model/__init__.py` - Added exports for BaseModel, BaseDataConfig, BaseModelConfig, BaseMlflowConfig, FingerprintConfig, MLflowMixin, ModelRegistry

#### Removed

(none)

### Phase 2: Feature Generation Module ✅

#### Added

- `src/admet/features/__init__.py` - Feature generation module init with FingerprintGenerator export
- `src/admet/features/fingerprints.py` - FingerprintGenerator class supporting:
  - Morgan fingerprints (circular, configurable radius/bits/chirality)
  - RDKit fingerprints (path-based, configurable path length/bits)
  - MACCS keys (fixed 167 bits)
  - Mordred descriptors (~1800 molecular descriptors)
  - Invalid SMILES handling with zero vectors
  - Batch and single molecule generation
- `tests/test_fingerprints.py` - 22 unit tests for fingerprint generation

#### Modified

- `pyproject.toml` - Added `mordred-community>=2.0.0` and `catboost>=1.2` dependencies

#### Removed

(none)

### Phase 3: Config Migration ✅

#### Added

- `scripts/lib/__init__.py` - Library utilities module init
- `scripts/lib/config_migration.py` - Config migration utilities:
  - `migrate_to_new_structure()` - Migrate old config to new model.type structure
  - `migrate_config_file()` - Migrate single YAML file in place
  - `migrate_config_directory()` - Batch migrate directory
  - `ensure_model_type()` - Add model.type with backward compatibility
  - `get_model_type()` / `get_model_params()` - Config accessor utilities
- `scripts/lib/config_validation.py` - Config validation utilities:
  - `validate_config()` - Validate config against model-specific schema
  - Model-specific validators for chemprop, chemeleon, xgboost, lightgbm, catboost

#### Modified

(none - migration is opt-in, existing configs preserved)

#### Removed

(none)

### Phase 4: Refactor ChempropModel ✅

#### Added

- `src/admet/model/chemprop/adapter.py` - ChempropModelAdapter:
  - Implements BaseModel interface wrapping existing ChempropModel
  - Registered with ModelRegistry as "chemprop"
  - Supports both new (model.type + model.chemprop) and legacy config structures
  - Handles fit/predict interface translation
- `tests/test_chemprop_adapter.py` - 7 unit tests for adapter

#### Modified

- `src/admet/model/chemprop/__init__.py` - Added ChempropModelAdapter to exports and lazy imports

#### Removed

(none)

### Phase 5: Implement ChemeleonModel ✅

#### Added

- `src/admet/model/chemeleon/__init__.py` - Module init with ChemeleonModel, GradualUnfreezeCallback exports
- `src/admet/model/chemeleon/callbacks.py` - GradualUnfreezeCallback for transfer learning:
  - Scheduled unfreezing of encoder/decoder layers
  - Learning rate multiplier support for unfrozen layers
  - Integrates with PyTorch Lightning Trainer
- `src/admet/model/chemeleon/model.py` - ChemeleonModel implementation:
  - Auto-downloads pre-trained weights from Zenodo (zenodo.org/records/15460715)
  - Frozen encoder by default for transfer learning
  - GradualUnfreezeCallback integration via get_trainer_callbacks()
  - Full fit/predict implementation using PyTorch Lightning
  - Registered with @ModelRegistry.register("chemeleon")
- `tests/test_chemeleon_model.py` - 10 unit tests for ChemeleonModel and callbacks

#### Modified

- `src/admet/model/chemeleon/model.py` - Fixed field name: encoder_lr_multiplier → unfreeze_encoder_lr_multiplier to match UnfreezeScheduleConfig

#### Removed

(none)

### Phase 6: Implement Classical Models ✅

#### Added

- `src/admet/model/classical/base.py` - ClassicalModelBase class:
  - Shared functionality for gradient boosting models
  - FingerprintGenerator integration for feature generation
  - MLflowMixin for experiment tracking
  - Save/load functionality via joblib
  - sklearn-compatible get_params/set_params interface
- `src/admet/model/classical/xgboost_model.py` - XGBoostModel:
  - Registered with @ModelRegistry.register("xgboost")
  - Configurable via model.xgboost section
  - Default params: n_estimators=100, max_depth=6, learning_rate=0.1
- `src/admet/model/classical/lightgbm_model.py` - LightGBMModel:
  - Registered with @ModelRegistry.register("lightgbm")
  - Configurable via model.lightgbm section
  - Default params: n_estimators=100, num_leaves=31, learning_rate=0.1
- `src/admet/model/classical/catboost_model.py` - CatBoostModel:
  - Registered with @ModelRegistry.register("catboost")
  - Configurable via model.catboost section
  - Default params: iterations=100, depth=6, learning_rate=0.1
- `tests/test_classical_models.py` - 24 unit tests for classical models

#### Modified

- `src/admet/model/classical/__init__.py` - Updated exports to include new model classes while preserving legacy models module

#### Removed

(none)

### Phase 7: Update Ensemble and CLI ✅

#### Added

- `src/admet/model/ensemble.py` - Generic Ensemble class:
  - Model-agnostic ensemble that works with any registered model type
  - Configurable number of models and aggregation method (mean/median)
  - Uncertainty estimation via ensemble variance
  - Save/load functionality with joblib
  - MLflow integration via MLflowMixin
- `tests/test_ensemble_generic.py` - 9 unit tests for generic Ensemble

#### Modified

- `src/admet/cli/model.py` - Updated CLI for multi-model support:
  - `train` command now auto-detects model type from config
  - Added `--model-type` flag to override config model type
  - Added `train-chemprop` legacy command for backward compatibility
  - Added `list` command to show all registered model types
  - Added `_register_all_models()` helper for lazy model registration
  - Added `_train_model_from_config()` for registry-based training

#### Removed

(none)

### Phase 8: Per-Model HPO Configuration ✅

#### Added

- `src/admet/model/hpo/__init__.py` - Model-agnostic HPO search space definitions:
  - `ParameterSpace` dataclass for distribution configuration
  - `XGBoostSearchSpace`, `LightGBMSearchSpace`, `CatBoostSearchSpace`, `ChemeleonSearchSpace`
  - `FingerprintSearchSpace` for fingerprint configuration tuning
  - `HPOSearchSpaceConfig` unified search space configuration
  - Default search space generators for each model type
  - `get_search_space_for_model()` factory function
- `src/admet/model/hpo/search_space.py` - Search space builder:
  - `_build_parameter_space()` converts ParameterSpace to Ray Tune format
  - `build_search_space()` builds unified search space from config
  - Model-specific builders: `build_xgboost_search_space()`, `build_lightgbm_search_space()`, etc.
- `tests/test_hpo_search_space_generic.py` - 21 unit tests for HPO search space

#### Modified

(none)

#### Removed

(none)

### Phase 9: End-to-End Testing and Documentation ✅

#### Added

(none - testing done via comprehensive unit test suite)

#### Modified

- `src/admet/model/__init__.py` - Updated exports:
  - Added all model parameter config classes (XGBoostModelParams, LightGBMModelParams, etc.)
  - Added Ensemble class export
  - Added chemeleon and hpo sub-modules to __all__

#### Removed

(none)

---

## Test Summary

All **118 tests** pass across the new implementation:

| Test File | Tests | Status |
|-----------|-------|--------|
| test_model_base.py | 25 | ✅ |
| test_fingerprints.py | 22 | ✅ |
| test_chemprop_adapter.py | 7 | ✅ |
| test_chemeleon_model.py | 10 | ✅ |
| test_classical_models.py | 24 | ✅ |
| test_ensemble_generic.py | 9 | ✅ |
| test_hpo_search_space_generic.py | 21 | ✅ |
| **Total** | **118** | ✅ |

## Breaking Changes

None - all changes are backward compatible:
- Existing configs continue to work via adapter patterns
- Legacy ChempropModel usage preserved
- Old CLI commands remain functional
