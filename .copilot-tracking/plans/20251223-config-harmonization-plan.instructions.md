---
applyTo: ".copilot-tracking/changes/20251223-config-harmonization-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Configuration Harmonization for Multi-Model Support

## Overview

Harmonize frontend configuration and backend merging to enable key features (featurizers, task affinity, curriculum learning) across all models, ensembles, and HPO.

## Objectives

- Unify configuration schema across ChemProp, classical models (XGBoost, LightGBM, CatBoost), and Chemeleon
- Extract curriculum learning, task affinity, and sampling strategies into model-agnostic modules
- Create consistent featurization abstraction for graphs, fingerprints, and foundation models
- Enable HPO to search across training strategy parameters for any model type
- Maintain backward compatibility with existing configs during migration

## Research Summary

### Project Files

- [src/admet/model/config.py](src/admet/model/config.py) - Partial base configs needing completion
- [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py) - ChemProp-specific configs with features to extract
- [src/admet/model/classical/base.py](src/admet/model/classical/base.py) - Classical model base missing curriculum support
- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py) - Ensemble logic tightly coupled to ChemProp
- [src/admet/model/hpo/__init__.py](src/admet/model/hpo/__init__.py) - HPO search spaces needing unification

### External References

- #file:../research/config-harmonization-plan.md - Complete research and architecture design

### Standards References

- #file:../../.github/instructions/python.instructions.md - Python coding conventions
- #file:../../.github/instructions/performance-optimization.instructions.md - Performance best practices

## Implementation Checklist

### [ ] Phase 1: Unified Base Configs

- [ ] Task 1.1: Create UnifiedDataConfig dataclass
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 15-60)

- [ ] Task 1.2: Create UnifiedMlflowConfig dataclass
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 62-100)

- [ ] Task 1.3: Create UnifiedOptimizationConfig dataclass
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 102-160)

- [ ] Task 1.4: Add config conversion helpers to ChempropConfig
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 162-220)

- [ ] Task 1.5: Create unit tests for base config conversions
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 222-280)

### [ ] Phase 2: Model-Agnostic Training Strategies

- [ ] Task 2.1: Create training module structure
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 282-330)

- [ ] Task 2.2: Extract SamplingConfig and implement model-agnostic sampler
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 332-400)

- [ ] Task 2.3: Extract CurriculumLearningConfig and implement scheduler
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 402-480)

- [ ] Task 2.4: Extract TaskAffinityConfig and implement affinity computer
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 482-560)

- [ ] Task 2.5: Create CurriculumAwareModel protocol
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 562-610)

- [ ] Task 2.6: Update ClassicalModelBase for curriculum support
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 612-680)

### [ ] Phase 3: Unified Featurization

- [ ] Task 3.1: Create FeaturizationConfig dataclass hierarchy
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 682-750)

- [ ] Task 3.2: Create featurizer factory
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 752-820)

- [ ] Task 3.3: Create graph featurizer wrapper
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 822-880)

- [ ] Task 3.4: Add featurization tests
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 882-940)

### [ ] Phase 4: Unified Model Factory

- [ ] Task 4.1: Create UnifiedModelConfig master dataclass
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 942-1020)

- [ ] Task 4.2: Update ModelRegistry with unified config support
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1022-1090)

- [ ] Task 4.3: Add from_unified_config to ChempropAdapter
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1092-1160)

- [ ] Task 4.4: Add from_unified_config to classical models
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1162-1230)

- [ ] Task 4.5: Add from_unified_config to ChemeleonModel
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1232-1290)

- [ ] Task 4.6: Update ModelEnsemble for unified configs
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1292-1360)

### [ ] Phase 5: HPO Integration

- [ ] Task 5.1: Create UnifiedSearchSpace config
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1362-1440)

- [ ] Task 5.2: Add training strategy parameters to search space
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1442-1510)

- [ ] Task 5.3: Update HPO trainable for unified configs
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1512-1580)

- [ ] Task 5.4: Create HPO integration tests
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1582-1640)

### [ ] Phase 6: Migration and Documentation

- [ ] Task 6.1: Create config migration script
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1642-1720)

- [ ] Task 6.2: Migrate example configs
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1722-1780)

- [ ] Task 6.3: Add config validation with helpful errors
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1782-1840)

- [ ] Task 6.4: Update documentation
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1842-1910)

- [ ] Task 6.5: Create integration test suite
  - Details: .copilot-tracking/details/20251223-config-harmonization-details.md (Lines 1912-1980)

## Dependencies

- Python 3.10+
- OmegaConf for config management
- dataclasses for structured configs
- pytest for testing
- Existing model implementations (ChemProp, XGBoost, LightGBM, CatBoost, Chemeleon)

## Success Criteria

- Single YAML schema works for all model types
- Curriculum learning available for classical models
- Task affinity grouping works across model types
- HPO can search training strategy parameters
- All existing configs continue to work (backward compatibility)
- Test coverage >90% on new config/training modules
- Documentation complete with migration guide
