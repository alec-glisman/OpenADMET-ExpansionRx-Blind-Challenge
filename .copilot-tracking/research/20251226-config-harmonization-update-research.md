<!-- markdownlint-disable-file -->

# Task Research Notes: Config Harmonization Plan Update Assessment

## Research Executed

### File Analysis

- [src/admet/model/config.py](src/admet/model/config.py)
  - Contains comprehensive base configs: `BaseDataConfig`, `BaseMlflowConfig`, `FingerprintConfig`
  - Already has model-specific param classes: `ChempropModelParams`, `ChemeleonModelParams`, `XGBoostModelParams`, `LightGBMModelParams`, `CatBoostModelParams`
  - **Key finding**: Significant progress has been made since original plan was created

- [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py)
  - Contains `JointSamplingConfig`, `CurriculumConfig`, `TaskOversamplingConfig`, `TaskAffinityConfig`, `InterTaskAffinityConfig`
  - These are Chemprop-specific but **should be model-agnostic**
  - `ChempropConfig` and `EnsembleConfig` already fully functional

- [src/admet/model/ensemble.py](src/admet/model/ensemble.py)
  - **New generic ensemble** that uses `ModelRegistry.create()`
  - Works with any model type via config `model.type` discriminator
  - Does NOT use structured dataclass configs - uses raw DictConfig

- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py)
  - **Problem identified**: Line 618 uses `OmegaConf.structured(ChempropConfig)` for ALL models
  - This forces Chemprop schema on non-Chemprop models, causing validation errors

- [src/admet/model/registry.py](src/admet/model/registry.py)
  - ModelRegistry pattern working well
  - Registered models: `chemprop`, `chemeleon`, `xgboost`, `lightgbm`, `catboost`
  - All models implement `from_config()` class method

### Code Search Results

- `@ModelRegistry.register` decorator used consistently across all model types
- Each model handles its own config structure in `from_config()` method
- No unified config validation - each model parses config differently

### Project Conventions

- Standards referenced: `.github/instructions/python.instructions.md`
- OmegaConf used throughout for config management
- Dataclass-based configs with MISSING defaults for required fields

## Key Discoveries

### What Has Changed Since Original Plan

1. **Generic Ensemble Added**: New `src/admet/model/ensemble.py` provides model-agnostic ensemble functionality
2. **JointSampler Implemented**: `JointSamplingConfig` with task oversampling + curriculum already exists
3. **ModelRegistry Pattern Mature**: All 5 model types registered and working
4. **Base Configs Exist**: `BaseDataConfig`, `BaseMlflowConfig` already in `config.py`
5. **Model Params Structured**: `XGBoostModelParams`, `LightGBMModelParams`, etc. defined

### Remaining Problems (Original Plan Still Valid)

| Problem | Status | Notes |
|---------|--------|-------|
| Duplicated DataConfig | ⚠️ Partial | `BaseDataConfig` exists but `DataConfig` still duplicated in chemprop |
| Chemprop-only features | ❌ Not started | `JointSamplingConfig`, `CurriculumConfig` still in chemprop/config.py |
| ChempropConfig forced on all models | ❌ Not fixed | ensemble.py Line 618 still uses `ChempropConfig` for all |
| Inconsistent config structure | ⚠️ Partial | Model params nested under `model.<type>.*` |
| Missing unified optimization | ⚠️ Partial | Classical models ignore neural-specific fields |

### Current YAML Config Structure Analysis

**Chemprop config structure** (configs/0-experiment/ensemble_chemprop_production.yaml):
```yaml
data:
  data_dir: ...
  smiles_col: SMILES
  target_cols: [...]
model:
  type: chemprop
  chemprop:
    depth: 4
    message_hidden_dim: 1100
    # ...
optimization:
  criterion: MAE
  max_lr: 0.000104
  # ...
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.1
  curriculum:
    enabled: false
mlflow:
  tracking: true  # Note: uses "tracking" not "enabled"
```

**Chemeleon config structure** (configs/0-experiment/chemeleon.yaml):
```yaml
model:
  type: chemeleon
  chemeleon:
    checkpoint_path: auto
    ffn_type: regression
    # ...
data:
  data_dir: ...
  smiles_col: SMILES
  target_cols: [...]
optimization:
  max_epochs: 100
  learning_rate: 1.0e-4  # Note: different field name than Chemprop
mlflow:
  enabled: true  # Note: uses "enabled" not "tracking"
```

### Inconsistencies Found

1. **MLflow field name**: Chemprop uses `tracking`, others use `enabled`
2. **Learning rate fields**: Chemprop uses `init_lr/max_lr/final_lr`, others use `learning_rate`
3. **JointSampling**: Only available for Chemprop, not exposed for Chemeleon
4. **No unified model section**: `model.type` + `model.<type>` pattern but not all models follow it

## Recommended Approach

### Simplify the Original Plan

The original plan has **6 complex phases** with significant refactoring. Given that:
- Core infrastructure exists (ModelRegistry, base configs)
- JointSampling already implemented for Chemprop
- Two ensemble implementations exist (generic + Chemprop-specific)

**Recommend a simpler 3-phase approach:**

### Phase 1: Unify Config Field Names (Quick Wins)

**Goal**: Make all configs use consistent field names without breaking changes.

1. **Standardize MLflow config**: Use `enabled` everywhere (add alias for `tracking`)
2. **Standardize optimization fields**: Support both `learning_rate` and `max_lr`
3. **Add quality_col to BaseDataConfig**: Allow curriculum without full JointSamplingConfig

**Effort**: 1 day

### Phase 2: Fix Ensemble Config Merging

**Goal**: Make chemprop/ensemble.py handle different model types correctly.

1. Create `get_base_config_for_model_type()` function:
   ```python
   def get_base_config_for_model_type(model_type: str) -> DictConfig:
       """Return appropriate base config for model type."""
       if model_type == "chemprop":
           return OmegaConf.structured(ChempropConfig)
       elif model_type == "chemeleon":
           return OmegaConf.structured(ChemeleonConfig)  # Need to create
       else:
           return OmegaConf.structured(ClassicalModelConfig)  # Need to create
   ```

2. Update `train_single_model()` to use this function

**Effort**: 1-2 days

### Phase 3: Extract Training Strategies (Optional Enhancement)

**Goal**: Make JointSampling, Curriculum available for Chemeleon.

1. Move `JointSamplingConfig`, `CurriculumConfig` to `src/admet/model/config.py`
2. Create `JointSampler` protocol that any DataLoader-based model can use
3. Update Chemeleon to optionally use JointSampler

**Effort**: 2-3 days (if desired)

## Updated Implementation Checklist

### [ ] Phase 1: Config Field Harmonization

- [ ] Task 1.1: Add `enabled` alias to MlflowConfig (backward compat with `tracking`)
- [ ] Task 1.2: Add `learning_rate` alias to OptimizationConfig (maps to `max_lr`)
- [ ] Task 1.3: Add `quality_col` field to BaseDataConfig
- [ ] Task 1.4: Update config loading to handle both old and new field names
- [ ] Task 1.5: Write tests for backward compatibility

### [ ] Phase 2: Model-Aware Config Merging

- [ ] Task 2.1: Create `ChemeleonConfig` structured dataclass
- [ ] Task 2.2: Create `ClassicalModelConfig` structured dataclass
- [ ] Task 2.3: Create `get_base_config_for_model_type()` factory function
- [ ] Task 2.4: Update `chemprop/ensemble.py` to use factory
- [ ] Task 2.5: Write integration tests for multi-model ensemble

### [ ] Phase 3: Unified Training Strategies (Optional)

- [ ] Task 3.1: Move sampling configs to `src/admet/model/config.py`
- [ ] Task 3.2: Create `SamplerProtocol` for model-agnostic sampling
- [ ] Task 3.3: Update Chemeleon to accept JointSampler
- [ ] Task 3.4: Update HPO search space for unified params
- [ ] Task 3.5: Migrate example configs

## Dependencies

- Python 3.10+
- OmegaConf for config management
- Existing ModelRegistry infrastructure
- pytest for testing

## Success Criteria

- Single YAML schema works for all model types (with model-specific sections)
- MLflow config uses consistent `enabled` field
- Chemprop ensemble can train mixed model types without validation errors
- All existing configs continue to work (full backward compatibility)
- Test coverage maintained

## Files to Modify

### Priority 1 (Phase 1)
- `src/admet/model/chemprop/config.py` - Add field aliases
- `src/admet/model/config.py` - Add quality_col to BaseDataConfig
- `tests/test_chemprop_config.py` - Add backward compat tests

### Priority 2 (Phase 2)
- `src/admet/model/config.py` - Add ChemeleonConfig, ClassicalModelConfig
- `src/admet/model/chemprop/ensemble.py` - Use model-aware config factory
- `tests/test_ensemble_chemprop.py` - Add multi-model tests

### Priority 3 (Phase 3 - Optional)
- `src/admet/model/config.py` - Move sampling configs
- `src/admet/model/chemeleon/model.py` - Add JointSampler support
- `configs/` - Update example configs

## Comparison: Original Plan vs Updated Plan

| Aspect | Original Plan (Dec 23) | Updated Plan (Dec 26) |
|--------|------------------------|------------------------|
| Phases | 6 complex phases | 3 focused phases |
| New modules | Create `src/admet/training/` | Extend existing modules |
| Config refactor | Full rewrite of configs | Incremental additions |
| Risk | High (major refactor) | Low (backward compat) |
| Effort | 10-15 days | 4-6 days |
| Curriculum for classical | Yes (complex) | No (classical doesn't need DataLoader) |
| JointSampling for Chemeleon | Yes | Yes (Phase 3) |

## Questions for User

1. **Do you need curriculum learning for classical models (XGBoost, LightGBM, CatBoost)?**
   - Classical models don't use PyTorch DataLoader, so JointSampler would need significant adaptation
   - Recommendation: Skip this unless specifically needed

2. **Is the Chemprop-specific ensemble the primary ensemble, or the generic one?**
   - `src/admet/model/ensemble.py` (generic) vs `src/admet/model/chemprop/ensemble.py`
   - The generic one is simpler but has fewer features
   - The Chemprop one has Ray parallelization, MLflow nesting, etc.

3. **Which config structure do you prefer for model params?**
   - Option A: `model.type` + `model.<type>.*` (current Chemprop pattern)
   - Option B: `model.type` + `<type>.*` at root level
   - Option C: Single `model.*` section with discriminator
