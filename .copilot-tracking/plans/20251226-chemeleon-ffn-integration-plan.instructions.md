---
applyTo: ".copilot-tracking/changes/20251226-chemeleon-ffn-integration-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: CheMeleon-Chemprop FFN Integration

## Overview

Integrate CheMeleon model with Chemprop's FFN architecture system to support all 3 FFN types (regression, mixture_of_experts, branched) with HPO support.

## Objectives

- Enable CheMeleon to use all 3 FFN architectures via shared factory
- Add HPO support for CheMeleon FFN architecture selection
- Update all documentation, configs, and tests
- Maintain backward compatibility with existing configs

## Research Summary

### Project Files

- `src/admet/model/chemeleon/model.py` - Current hardcoded RegressionFFN at line 187
- `src/admet/model/chemprop/ffn.py` - MoE and BranchedFFN implementations
- `src/admet/model/config.py` - ChemeleonModelParams config class
- `src/admet/model/chemprop/hpo_search_space.py` - Conditional FFN param sampling

### External References

- #file:../research/20251226-chemeleon-ffn-integration-plan.md - Full research and code examples

### Standards References

- #file:../../.github/instructions/python.instructions.md - Python conventions
- #file:../../.github/instructions/self-explanatory-code-commenting.instructions.md - Commenting guidelines

## Implementation Checklist

### [ ] Phase 1: Shared FFN Factory

- [ ] Task 1.1: Create `src/admet/model/ffn_factory.py` with `create_ffn_predictor()` function
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 1-60)

- [ ] Task 1.2: Add unit tests for FFN factory in `tests/test_ffn_factory.py`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 61-90)

### [ ] Phase 2: Config Updates

- [ ] Task 2.1: Update `ChemeleonModelParams` in `src/admet/model/config.py`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 91-130)

### [ ] Phase 3: CheMeleon Model Update

- [ ] Task 3.1: Update `ChemeleonModel._init_model()` to use FFN factory
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 131-180)

- [ ] Task 3.2: Add CheMeleon FFN type tests in `tests/test_chemeleon_model.py`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 181-220)

### [ ] Phase 4: Chemprop Refactor

- [ ] Task 4.1: Refactor `ChempropModel` to use shared FFN factory
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 221-260)

### [ ] Phase 5: HPO Support

- [ ] Task 5.1: Create `src/admet/model/chemeleon/hpo_config.py`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 261-310)

- [ ] Task 5.2: Create `src/admet/model/chemeleon/hpo_search_space.py`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 311-360)

- [ ] Task 5.3: Create `src/admet/model/chemeleon/hpo.py` HPO runner
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 361-420)

### [ ] Phase 6: Config Files

- [ ] Task 6.1: Create `configs/0-experiment/chemeleon.yaml`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 421-460)

- [ ] Task 6.2: Create `configs/1-hpo-single/hpo_chemeleon.yaml`
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 461-500)

### [ ] Phase 7: Documentation

- [ ] Task 7.1: Update `README.md` model table and examples
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 501-530)

- [ ] Task 7.2: Update `docs/guide/modeling.rst` with CheMeleon FFN examples
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 531-570)

- [ ] Task 7.3: Update `MODEL_CARD.md` architecture section
  - Details: .copilot-tracking/details/20251226-chemeleon-ffn-integration-details.md (Lines 571-600)

## Dependencies

- chemprop >= 2.0.0
- PyTorch Lightning
- Ray Tune (for HPO)
- OmegaConf

## Success Criteria

- CheMeleon model accepts `ffn_type` parameter with all 3 options
- Shared FFN factory used by both Chemprop and CheMeleon
- CheMeleon HPO can search over FFN architectures
- All existing tests pass
- New tests for CheMeleon FFN types pass
- Documentation updated with examples
