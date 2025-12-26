---
applyTo: ".copilot-tracking/changes/20251226-config-harmonization-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Unified Configuration and Ensemble Harmonization

**Status**: ✅ COMPLETE (December 26, 2025)
**Final Test Results**: 636 passed, 5 skipped, 5 deselected

## Overview

Unify the YAML configuration schema across all model types (Chemprop, Chemeleon, XGBoost, LightGBM, CatBoost) and merge the two ensemble implementations into a single, feature-rich ensemble class.

## Objectives

- ✅ Create a unified configuration schema that works across all model types
- ✅ Add EnsembleSection and RayConfig to unified config
- ✅ Move training strategy configs (JointSampling, Curriculum, TaskAffinity) to model-agnostic location
- ✅ Enable JointSampling/Curriculum for Chemeleon (PyTorch-based)
- ✅ Error with clear message if curriculum enabled for classical models (no DataLoader)
- ✅ Use consistent field names across all configs (with backward-compatible aliases)
- ✅ All YAML configs load with unified schema

## Research Summary

### Project Files

- [src/admet/model/config.py](src/admet/model/config.py) - Base configs (BaseDataConfig, BaseMlflowConfig, FingerprintConfig, model params)
- [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py) - JointSamplingConfig, CurriculumConfig, ChempropConfig, EnsembleConfig
- [src/admet/model/ensemble.py](src/admet/model/ensemble.py) - Generic ensemble (simple, no Ray)
- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) - Feature-rich ensemble (Ray, MLflow nesting, split/fold)
- [src/admet/model/registry.py](src/admet/model/registry.py) - ModelRegistry pattern
- [src/admet/model/chemeleon/model.py](src/admet/model/chemeleon/model.py) - Chemeleon with PyTorch DataLoader

### External References

- #file:.copilot-tracking/research/20251226-config-harmonization-update-research.md - Updated research findings

### Standards References

- #file:.github/instructions/python.instructions.md - Python coding conventions
- #file:.github/instructions/performance-optimization.instructions.md - Performance best practices

## Implementation Checklist

### [x] Phase 1: Unified Configuration Schema ✅

- [x] Task 1.1: Create UnifiedModelConfig master dataclass
- [x] Task 1.2: Move training strategy configs to model-agnostic location
- [x] Task 1.3: Create model-type-aware config validation
- [x] Task 1.4: Write config schema tests (38 tests pass)

### [x] Phase 2: Ensemble Configuration ✅

- [x] Task 2.1: Add EnsembleSection to UnifiedModelConfig
- [x] Task 2.2: Add RayConfig for parallelization settings
- [x] Task 2.3: Write ensemble config tests (15 tests pass)

### [x] Phase 3: Model Adapter Updates ✅

- [x] Task 3.1: Update ChemeleonModel for JointSampling support
- [x] Task 3.2: Add _create_train_dataloader() method with JointSampler
- [x] Task 3.3: Add curriculum validation error for classical models
- [x] Task 3.4: Verify Chemeleon tests pass (20 tests)

### [x] Phase 4: Configuration Migration ✅

- [x] Task 4.1: Extend ChemeleonModelParams for YAML compatibility
- [x] Task 4.2: Extend classical model params (XGBoost, LightGBM, CatBoost)
- [x] Task 4.3: Add UnifiedMlflowConfig tracking alias
- [x] Task 4.4: Add SchedulerConfig and optimization fields
- [x] Task 4.5: Verify all configs/4-more-models/*.yaml load successfully

### [x] Phase 5: Documentation ✅

- [x] Task 5.1: Update README.md with classical models status
- [x] Task 5.2: Add Configuration System section to README
- [x] Task 5.3: Update tracking documents

### [x] Phase 6: Integration Tests ✅

- [x] Task 6.1: Run full test suite (636 passed)
- [x] Task 6.2: Verify backward compatibility with existing tests
- [x] Task 6.3: Verify all config imports work correctly

## Dependencies

- Python 3.10+
- OmegaConf for config management
- Ray for parallel ensemble training
- PyTorch for DataLoader-based sampling
- MLflow for experiment tracking
- pytest for testing

## Success Criteria ✅

- ✅ Single YAML schema works for all model types
- ✅ EnsembleSection and RayConfig in unified config
- ✅ JointSampling/Curriculum works for Chemprop and Chemeleon
- ✅ Classical models error clearly if curriculum enabled
- ✅ Backward-compatible aliases for existing configs
- ✅ All existing tests pass (636 passed)
- ✅ New tests cover unified config and ensemble (38 tests)
- ✅ All example configs load with unified schema
- ✅ Documentation updated with new API
