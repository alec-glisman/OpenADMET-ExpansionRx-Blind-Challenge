---
applyTo: ".copilot-tracking/changes/20251222-multi-model-api-refactoring-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Multi-Model API Refactoring

## Overview

Refactor the project's config and API to seamlessly integrate multiple model types (chemprop, chemeleon, xgboost, lightgbm, catboost, tabpfn) with a common base class, unified config structure, feature generation module, and consistent MLflow tracking.

## Objectives

- Create a unified model interface enabling seamless model swapping
- Implement feature generation module supporting Morgan, RDKit, and Mordred fingerprints
- Migrate all existing YAML configs to new nested structure with model type discriminator
- Ensure consistent MLflow behavior across all model types
- Implement ChemeleonModel with frozen encoder by default and gradual unfreezing scheduler
- Support mixed model type ensembles
- Implement per-model HPO search spaces

## Research Summary

### Project Files

- src/admet/model/chemprop/config.py - Current OmegaConf dataclass configs (695 lines)
- src/admet/model/chemprop/model.py - ChempropModel class (2484 lines) with fit/predict/from_config
- src/admet/model/chemprop/ensemble.py - ChempropEnsemble with Ray parallelization (1355 lines)
- src/admet/model/classical/models.py - Simple XGBoost/LightGBM wrappers (no class abstraction)
- src/admet/model/chemprop/hpo_config.py - HPO configuration with SearchSpaceConfig
- configs/ - 100+ YAML config files requiring migration

### External References

- #file:../research/20251222-multi-model-api-refactoring-research.md - Comprehensive research document
- sklearn BaseEstimator API conventions (fit, predict, get_params, set_params)
- AutoGluon AbstractModel and ModelRegistry patterns
- RDKit fingerprint generation (Morgan, RDKit, MACCS, AtomPairs)
- mordred-community package for Mordred descriptors

### Standards References

- #file:../../.github/instructions/python.instructions.md - Python coding conventions
- #file:../../.github/instructions/code-review-generic.instructions.md - Code review standards

## Implementation Checklist

### [ ] Phase 1: Foundation - Base Classes and Registry

- [ ] Task 1.1: Create BaseModel abstract class
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1-60)

- [ ] Task 1.2: Create ModelRegistry with decorator-based registration
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 61-100)

- [ ] Task 1.3: Create MLflowMixin for consistent tracking
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 101-150)

- [ ] Task 1.4: Create unified config dataclasses
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 151-250)

- [ ] Task 1.5: Add unit tests for base classes
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 251-290)

### [ ] Phase 2: Feature Generation Module

- [ ] Task 2.1: Create FingerprintConfig dataclass
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 291-340)

- [ ] Task 2.2: Implement FingerprintGenerator class
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 341-420)

- [ ] Task 2.3: Add mordred-community dependency
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 421-450)

- [ ] Task 2.4: Add unit tests for fingerprint generation
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 451-490)

### [ ] Phase 3: Config Migration

- [ ] Task 3.1: Create config migration script
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 491-580)

- [ ] Task 3.2: Run migration on all existing configs
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 581-610)

- [ ] Task 3.3: Validate migrated configs
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 611-640)

### [ ] Phase 4: Refactor ChempropModel

- [ ] Task 4.1: Update ChempropModel to inherit from BaseModel
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 641-720)

- [ ] Task 4.2: Register ChempropModel with ModelRegistry
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 721-750)

- [ ] Task 4.3: Update ChempropModel.from_config for new config structure
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 751-800)

- [ ] Task 4.4: Add tests for refactored ChempropModel
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 801-840)

### [ ] Phase 5: Implement ChemeleonModel

- [ ] Task 5.1: Create ChemeleonConfig with freezing options
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 841-900)

- [ ] Task 5.2: Implement GradualUnfreezeCallback for encoder/decoder
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 901-970)

- [ ] Task 5.3: Implement ChemeleonModel with pretrained loading
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 971-1050)

- [ ] Task 5.4: Add Zenodo auto-download for chemeleon_mp.pt
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1051-1100)

- [ ] Task 5.5: Add tests for ChemeleonModel
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1101-1150)

### [ ] Phase 6: Implement Classical Models

- [ ] Task 6.1: Implement XGBoostModel with BaseModel interface
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1151-1230)

- [ ] Task 6.2: Implement LightGBMModel with BaseModel interface
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1231-1310)

- [ ] Task 6.3: Implement CatBoostModel with BaseModel interface
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1311-1390)

- [ ] Task 6.4: Add tests for classical models
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1391-1440)

### [ ] Phase 7: Update Ensemble and CLI

- [ ] Task 7.1: Create generic Ensemble class supporting mixed model types
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1441-1530)

- [ ] Task 7.2: Update CLI to use ModelRegistry.create()
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1531-1590)

- [ ] Task 7.3: Add integration tests for ensemble and CLI
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1591-1630)

### [ ] Phase 8: Per-Model HPO Configuration

- [ ] Task 8.1: Create per-model HPO search space dataclasses
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1631-1710)

- [ ] Task 8.2: Update HPO trainable to use ModelRegistry
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1711-1770)

- [ ] Task 8.3: Add tests for HPO configuration
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1771-1810)

### [ ] Phase 9: End-to-End Testing and Documentation

- [ ] Task 9.1: Add end-to-end training tests for all model types
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1811-1870)

- [ ] Task 9.2: Update documentation and README
  - Details: .copilot-tracking/details/20251222-multi-model-api-refactoring-details.md (Lines 1871-1910)

## Dependencies

- OmegaConf (existing)
- mordred-community (new - for Mordred descriptors)
- RDKit (existing - for fingerprints)
- XGBoost, LightGBM, CatBoost (existing)
- Ray (existing - for ensemble parallelization)
- MLflow (existing)
- PyTorch Lightning (existing - for chemprop)

## Success Criteria

- ModelRegistry.create(config) returns correct model type based on config.model.type
- model.fit() and model.predict() work uniformly across all model types
- All 100+ existing YAML configs migrated to new nested structure
- FingerprintGenerator supports Morgan, RDKit, and Mordred with configurable params
- MLflow tracking works consistently for all model types
- ChemeleonModel loads pretrained weights with frozen encoder by default
- GradualUnfreezeCallback correctly unfreezes encoder/decoder at specified epochs
- Mixed model type ensembles train and predict successfully
- Per-model HPO search spaces work with Ray Tune
- All existing tests continue to pass
- New tests cover base classes, models, and integration scenarios
