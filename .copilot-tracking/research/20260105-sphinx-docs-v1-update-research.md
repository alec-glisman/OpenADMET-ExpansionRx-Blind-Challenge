<!-- markdownlint-disable-file -->

# Task Research Notes: Sphinx Documentation Update for v1 API

## Research Executed

### File Analysis

- `docs/index.rst`
  - Main toctree includes 31 pages across API Reference and Guides sections
  - Version referenced as `"0.0.1"` but `src/admet/__init__.py` shows `__version__ = "1.2.0"`

- `docs/api/admet.model.rst`
  - References `ChempropConfig`, `EnsembleConfig`, `HPOConfig`, `DataConfig`, `ModelConfig`, `OptimizationConfig`, `MlflowConfig`
  - These are **legacy config names** - actual API uses `UnifiedModelConfig` pattern from `admet.model.config`
  - Missing documentation for new unified config system

- `docs/api/admet.rst`
  - Version example shows `"0.0.1"` - needs update to `"1.2.0"`
  - Package structure list is accurate but incomplete (missing `admet.features` subpackage)

- `docs/api/admet.util.rst`
  - References `parse_data_dir_params()` function
  - Missing: `profiling.py`, `ray_logging.py` modules not documented

- `docs/api/leaderboard.rst`
  - Structure matches actual `admet.leaderboard` module
  - API documentation appears current

- `docs/guide/cli.rst`
  - CLI examples mostly accurate for `admet leaderboard scrape`
  - Model training examples use **legacy programmatic API** (pre-UnifiedModelConfig)
  - `admet model ensemble --config` syntax is current

- `docs/guide/configuration.rst`
  - Shows both old-style (`ChempropConfig`) and new-style (`model.type: chemprop`) configs
  - Missing comprehensive `UnifiedModelConfig` documentation
  - Config loading example at bottom references `UnifiedModelConfig` correctly

### Code Search Results

- `src/admet/__init__.py`
  - `__version__ = "1.2.0"` (docs show `0.0.1`)
  - Only exports `leaderboard` module at top level

- `src/admet/model/__init__.py` exports:
  - Base classes: `BaseModel`, `BaseDataConfig`, `BaseModelConfig`, `BaseMlflowConfig`
  - Configuration: `FingerprintConfig`, `XGBoostModelParams`, `LightGBMModelParams`, `CatBoostModelParams`, `ChemeleonModelParams`, `UnfreezeScheduleConfig`
  - Utilities: `MLflowMixin`, `ModelRegistry`, `Ensemble`
  - Sub-modules: `classical`, `chemprop`, `chemeleon`, `hpo`

- `src/admet/model/chemprop/__init__.py` exports (163 lines):
  - Config classes: `ChempropConfig`, `DataConfig`, `ModelConfig`, `OptimizationConfig`, `MlflowConfig`, `EnsembleConfig`, `EnsembleDataConfig`, `InterTaskAffinityConfig`
  - HPO config classes: `HPOConfig`, `SearchSpaceConfig`, `ParameterSpace`, `ASHAConfig`, `ResourceConfig`, `TransferLearningConfig`
  - Model classes: `ChempropModel`, `ChempropModelAdapter`, `ChempropHyperparams`
  - Ensemble classes: `ModelEnsemble`, `ChempropEnsemble` (backward compat alias)
  - HPO classes: `ChempropHPO`, `RayTuneReportCallback`
  - Task affinity: `TaskAffinityConfig`, `TaskAffinityComputer`, `TaskGrouper`, `InterTaskAffinityCallback`, `InterTaskAffinityComputer`

- `src/admet/features/__init__.py`:
  - Exports `FingerprintGenerator` class
  - **Not documented in API reference**

- `src/admet/util/__init__.py`:
  - Empty file - modules must be imported directly
  - Actual modules: `logging.py`, `profiling.py`, `ray_logging.py`, `utils.py`

### Documentation Build Analysis

- Clean build (`rm -rf docs/_build && make -C docs html`) shows:
  - 7 warnings total
  - `guide/logging.rst:514` - Title level inconsistent (`Test Coverage ~~~`)
  - `guide/logging.rst:527-529` - Undefined labels: `configuration`, `cli`, `hpo`
  - `guide/debugging_per_quality_metrics.md` - Not in any toctree
  - `guide/logging.rst` - Not in any toctree

### Project Conventions

- Standards referenced: Python docstring style (NumPy), OmegaConf dataclasses
- Instructions followed: `.github/copilot-instructions.md`

## Key Discoveries

### Version Mismatch

- Package version: `1.2.0` (src/admet/__init__.py)
- Documentation version example: `0.0.1` (docs/api/admet.rst)

### Missing API Documentation

1. **`admet.features` subpackage** - Contains `FingerprintGenerator` class, no rst file exists
2. **`admet.util.profiling`** - Profiling utilities not documented
3. **`admet.util.ray_logging`** - Ray logging utilities not documented
4. **`admet.model.chemeleon`** - Limited documentation in `admet.model.rst`

### Outdated API Documentation

1. **Config System**
   - Docs reference old `ChempropConfig` loading pattern
   - Current API uses `UnifiedModelConfig` with `model.type` discriminator
   - `admet.model.config.py` (1000 lines) defines unified config but not fully documented

2. **Model Training Examples**
   - CLI: `admet model train -c configs/...` (current)
   - Programmatic: Old pattern `ChempropModel.from_config(cfg)` vs new `ModelRegistry.create(cfg)`

3. **FFN Types**
   - Docs mention `regression`, `mixture_of_experts`, `branched`
   - Source also uses shorthand: `mlp`, `moe`, `branched` (with auto-mapping)

### Documentation Structure Issues

1. **Orphaned Documents**
   - `guide/logging.rst` - Not in index.rst toctree
   - `guide/debugging_per_quality_metrics.md` - Not in index.rst toctree

2. **Internal Link Errors**
   - `logging.rst` references `:ref:`configuration``, `:ref:`cli``, `:ref:`hpo`` - labels don't exist
   - Should use `:doc:` directive instead

3. **RST Formatting**
   - `logging.rst:514` - Title level inconsistent (uses `~~~` where `^^^` expected)

### Scripts Documentation

- `scripts/README.md` (424 lines) - Comprehensive script documentation
- Training scripts use current CLI: `admet model ensemble --config ...`
- HPO script: `python -m admet.model.chemprop.hpo --config ...`

## Recommended Approach

**Priority 1: Fix Build Errors**

1. Add `guide/logging` and `guide/debugging_per_quality_metrics` to `index.rst` toctree
2. Fix RST title level in `logging.rst` line 514
3. Replace `:ref:` with `:doc:` in `logging.rst` lines 527-529

**Priority 2: Version and Core Updates**

1. Update version in `docs/api/admet.rst` from `0.0.1` to `1.2.0`
2. Create `docs/api/admet.features.rst` for `FingerprintGenerator`
3. Update `docs/api/admet.util.rst` to include `profiling` and `ray_logging` modules

**Priority 3: Configuration Documentation**

1. Expand `docs/guide/configuration.rst` with `UnifiedModelConfig` schema documentation
2. Document the `model.type` discriminator pattern
3. Add model-type-specific parameter sections (chemprop, chemeleon, xgboost, etc.)

**Priority 4: Model API Updates**

1. Update `docs/api/admet.model.rst`:
   - Document `ModelRegistry` class and `create()` factory pattern
   - Add `ChempropModelAdapter` documentation
   - Update config class documentation to match unified schema
   - Add `Ensemble` base class documentation

2. Update programmatic examples in `docs/guide/cli.rst` and `docs/guide/modeling.rst`:
   - Replace `ChempropModel.from_config()` with `ModelRegistry.create()`
   - Update config loading pattern

**Priority 5: Classical Models and Features**

1. Document `admet.model.classical` subpackage (XGBoost, LightGBM, CatBoost wrappers)
2. Document `FingerprintConfig` and fingerprint types

## Implementation Guidance

- **Objectives**: Align Sphinx documentation with v1.2.0 API, fix build warnings, document new unified config system
- **Key Tasks**:
  1. Fix immediate build errors (toctree, RST formatting, broken refs)
  2. Update version references
  3. Create missing API documentation files
  4. Update configuration guide with UnifiedModelConfig
  5. Review and update code examples in all guides
- **Dependencies**: None - documentation-only changes
- **Success Criteria**:
  - `make -C docs html` completes with 0 warnings
  - All API modules have corresponding rst documentation
  - Code examples use current `ModelRegistry.create()` pattern
  - Config examples show `model.type` discriminator pattern

## Files Requiring Changes

| File | Change Type | Priority |
|------|-------------|----------|
| `docs/index.rst` | Add logging/debugging to toctree | P1 |
| `docs/guide/logging.rst` | Fix RST formatting, fix refs | P1 |
| `docs/api/admet.rst` | Update version | P2 |
| `docs/api/admet.features.rst` | New file | P2 |
| `docs/api/admet.util.rst` | Add modules | P2 |
| `docs/guide/configuration.rst` | Major expansion | P3 |
| `docs/guide/config_reference.rst` | Review/update | P3 |
| `docs/api/admet.model.rst` | Update classes | P4 |
| `docs/guide/cli.rst` | Update examples | P4 |
| `docs/guide/modeling.rst` | Update examples | P4 |
