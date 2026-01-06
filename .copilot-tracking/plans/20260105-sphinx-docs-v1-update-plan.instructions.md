---
applyTo: ".copilot-tracking/changes/20260105-sphinx-docs-v1-update-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Sphinx Documentation Update for v1 API

## Overview

Comprehensive documentation overhaul to align Sphinx docs with v1.2.0 API, including build error fixes, missing API docs, updated examples, and new classical models guide.

## Objectives

- Fix all Sphinx build warnings (target: 0 warnings)
- Update version references from 0.0.1 to 1.2.0
- Document all public API modules including `admet.features` and classical models
- Update all code examples to use `ModelRegistry.create()` pattern
- Create comprehensive classical models guide with usage examples
- Document `UnifiedModelConfig` schema in config_reference.rst

## Research Summary

### Project Files

- #file:docs/index.rst - Main toctree missing logging/debugging pages
- #file:docs/guide/logging.rst - RST formatting errors, broken refs
- #file:docs/api/admet.rst - Version shows 0.0.1, needs update
- #file:docs/api/admet.model.rst - Outdated config class references
- #file:src/admet/model/config.py - UnifiedModelConfig schema (1000 lines)
- #file:src/admet/model/registry.py - ModelRegistry factory pattern

### External References

- #file:.copilot-tracking/research/20260105-sphinx-docs-v1-update-research.md - Full research notes

### Standards References

- #file:.github/instructions/markdown.instructions.md - Markdown documentation standards
- #file:.github/copilot-instructions.md - Project conventions

## Implementation Checklist

### [ ] Phase 1: Fix Build Errors

- [ ] Task 1.1: Add orphaned documents to index.rst toctree
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 15-35)

- [ ] Task 1.2: Fix RST formatting in logging.rst
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 37-60)

- [ ] Task 1.3: Fix broken references in logging.rst
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 62-85)

- [ ] Task 1.4: Verify build completes with 0 warnings
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 87-95)

### [ ] Phase 2: Update Version and Core API Docs

- [ ] Task 2.1: Update version in docs/api/admet.rst
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 100-115)

- [ ] Task 2.2: Create docs/api/admet.features.rst
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 117-160)

- [ ] Task 2.3: Update docs/api/admet.util.rst with missing modules
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 162-210)

- [ ] Task 2.4: Add admet.features to docs/api/admet.rst toctree
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 212-230)

### [ ] Phase 3: Update Model API Documentation

- [ ] Task 3.1: Update docs/api/admet.model.rst with current classes
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 235-320)

- [ ] Task 3.2: Add ModelRegistry documentation
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 322-365)

- [ ] Task 3.3: Document Chemeleon subpackage
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 367-420)

- [ ] Task 3.4: Document classical models API (XGBoost, LightGBM, CatBoost)
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 422-490)

### [ ] Phase 4: Configuration Documentation

- [ ] Task 4.1: Update docs/guide/configuration.rst with UnifiedModelConfig
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 495-580)

- [ ] Task 4.2: Update docs/guide/config_reference.rst with complete schema
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 582-680)

- [ ] Task 4.3: Update config loading examples across all guides
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 682-720)

### [ ] Phase 5: Create Classical Models Guide

- [ ] Task 5.1: Create docs/guide/classical_models.rst
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 725-850)

- [ ] Task 5.2: Add classical_models to index.rst toctree
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 852-870)

### [ ] Phase 6: Update Code Examples in Guides

- [ ] Task 6.1: Update docs/guide/cli.rst examples
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 875-940)

- [ ] Task 6.2: Update docs/guide/modeling.rst examples
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 942-1010)

- [ ] Task 6.3: Update docs/guide/hpo.rst examples
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1012-1070)

- [ ] Task 6.4: Update docs/guide/architecture.rst examples
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1072-1130)

- [ ] Task 6.5: Review and update remaining guide files
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1132-1180)

### [ ] Phase 7: Final Validation

- [ ] Task 7.1: Clean build and verify 0 warnings
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1185-1200)

- [ ] Task 7.2: Verify all internal links resolve
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1202-1215)

- [ ] Task 7.3: Spot-check rendered HTML for code examples
  - Details: .copilot-tracking/details/20260105-sphinx-docs-v1-update-details.md (Lines 1217-1230)

## Dependencies

- Python 3.11 with virtualenv activated
- Sphinx and sphinx-build available
- Source code in src/admet/ for autodoc

## Success Criteria

- `make -C docs html` completes with 0 warnings
- All API modules documented in docs/api/
- All code examples use `ModelRegistry.create()` pattern
- Classical models guide created with XGBoost, LightGBM, CatBoost examples
- UnifiedModelConfig schema fully documented
- Version references updated to 1.2.0
