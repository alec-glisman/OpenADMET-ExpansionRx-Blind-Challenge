---
applyTo: ".copilot-tracking/changes/20260109-sphinx-docs-improvement-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Sphinx Documentation Improvement

## Overview

Systematically restructure and polish Sphinx documentation for ML practitioners and drug discovery experts with clean visual hierarchy and clear user journeys.

## Objectives

- Reorganize documentation into logical sections with progressive disclosure
- Create quickstart tutorial for 5-minute first model training
- Integrate fragmented CLI and scripts documentation into Sphinx
- Remove internal planning docs from public navigation
- Standardize guide styling and remove inconsistencies

## Research Summary

### Project Files

- docs/index.rst - Current landing page with cluttered 3-panel layout
- docs/guide/*.rst - 20 guide files with varying quality
- docs/planning/* - Internal docs incorrectly exposed in navigation
- scripts/README.md - 500 lines of script documentation outside Sphinx

### External References

- #file:../research/20260109-sphinx-docs-improvement-research.md - Comprehensive analysis

### Standards References

- #file:../../.github/instructions/markdown.instructions.md - Documentation standards

## Implementation Checklist

### [ ] Phase 1: Information Architecture Restructure

- [ ] Task 1.1: Create new directory structure for documentation sections

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L13-L27)

- [ ] Task 1.2: Reorganize index.rst with semantic toctree sections

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L29-L85)

- [ ] Task 1.3: Hide planning section from public navigation

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L87-L104)

### [ ] Phase 2: New Content Creation

- [ ] Task 2.1: Create quickstart tutorial (getting-started/quickstart.rst)

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L108-L148)

- [ ] Task 2.2: Create shell scripts reference (reference/scripts.rst)

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L150-L178)

- [ ] Task 2.3: Create getting-started section index

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L180-L195)

### [ ] Phase 3: Content Migration and Conversion

- [ ] Task 3.1: Convert debugging_per_quality_metrics.md to RST

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L199-L222)

- [ ] Task 3.2: Update CLI documentation with complete reference

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L224-L241)

### [ ] Phase 4: Stylistic Standardization

- [ ] Task 4.1: Standardize guide opening lines across all files

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L245-L271)

- [ ] Task 4.2: Remove emoji and symbols from documentation

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L273-L288)

- [ ] Task 4.3: Standardize section headers and structure

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L290-L308)

### [ ] Phase 5: Visual Enhancements

- [ ] Task 5.1: Simplify landing page design

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L312-L356)

- [ ] Task 5.2: Add workflow diagrams to key guides

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L358-L382)

### [ ] Phase 6: Validation and Cleanup

- [ ] Task 6.1: Build documentation and fix all warnings

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L386-L403)

- [ ] Task 6.2: Verify cross-references and links

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L405-L421)

- [ ] Task 6.3: Test user journeys for both audiences

  - Details: [sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md#L423-L445)

## Dependencies

- Sphinx 7.x with Furo theme
- MyST parser for markdown support
- sphinx_panels extension

## Success Criteria

- Documentation builds with zero warnings
- New user can train first model in under 10 minutes following quickstart
- All CLI commands and scripts documented within Sphinx
- No internal planning documents visible in public navigation
- Consistent visual styling across all 20+ guide pages
