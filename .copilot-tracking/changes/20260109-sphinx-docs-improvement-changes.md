# Changes: Sphinx Documentation Improvement

**Date**: 2026-01-09
**Plan**: [20260109-sphinx-docs-improvement-plan.instructions.md](../plans/20260109-sphinx-docs-improvement-plan.instructions.md)
**Details**: [20260109-sphinx-docs-improvement-details.md](../details/20260109-sphinx-docs-improvement-details.md)

## Summary

Comprehensive restructure of Sphinx documentation for improved organization, navigation, and user experience targeting ML practitioners and drug discovery experts.

## Changes Made

### Phase 1: Information Architecture Restructure

✅ **Task 1.1: Created new directory structure**

- Created `docs/getting-started/` for onboarding content
- Created `docs/reference/` for reference documentation
- Created `docs/dev/` for development/internal docs

✅ **Task 1.2: Reorganized index.rst with semantic toctree sections**

- Restructured main landing page toctree into logical groupings:
  - Getting Started (quickstart, architecture)
  - Data Pipeline (data sources, splitting, debugging)
  - Model Training (modeling, classical models, HPO)
  - Advanced Training (curriculum, task affinity)
  - Operations (MLflow, profiling, logging)
  - Reference (CLI, config, scripts, API)
  - Community (overview, leaderboard, development)
  - Development (planning - hidden from main navigation)
- Changed from flat 20-item list to 8 semantic sections

✅ **Task 1.3: Hide planning section from public navigation**

- Created `docs/dev/planning.rst` wrapper for internal planning docs
- Moved planning section to :hidden: Development toctree
- Planning docs still accessible via direct link but not visible in sidebar

### Phase 2: New Content Creation

✅ **Task 2.1: Created quickstart tutorial (getting-started/quickstart.rst)**

- Complete 5-minute end-to-end tutorial from config to MLflow results
- Both CLI and Python API examples with expected output
- Step-by-step instructions with minimal config example
- Troubleshooting section for common issues
- Links to advanced features (ensemble, HPO, curriculum learning)

✅ **Task 2.2: Created shell scripts reference (reference/scripts.rst)**

- Comprehensive documentation of all 14+ shell scripts from scripts/README.md
- Organized by category: training, HPO, data processing, analysis, MLflow, infrastructure
- Usage examples with options tables for every script
- Cross-references to related guides (CLI, modeling, HPO, configuration)
- Expected data directory structure and configuration files
- Total: 700+ lines of RST documentation migrated from markdown

✅ **Task 2.3: Created getting-started section index**

- Clear value proposition for new users
- Tab-based learning paths for 4 user personas:
  - Train a Single Model
  - Run HPO
  - Deploy Ensemble
  - Contribute
- Installation instructions with uv and pip methods
- Common workflow examples for quick reference
- Links to quickstart, CLI reference, scripts reference, architecture

### Phase 3: Content Migration and Conversion

✅ **Task 3.1: Converted debugging_per_quality_metrics.md to RST**

- Converted markdown to RST with proper code blocks and section formatting
- Removed all emoji symbols (✅ ✓) replacing with text equivalents
- Maintained all diagnostic content and code examples
- Deleted markdown source file after conversion

✅ **Task 3.2: Updated CLI documentation with complete reference**

- Expanded from 174 lines to 600+ lines of comprehensive documentation
- Documented all 10 CLI commands across 3 command groups:
  - Model commands: train, ensemble, hpo, hpo-list-studies, list, train-chemprop (deprecated)
  - Data commands: split
  - Leaderboard commands: scrape, report
- Added usage examples with options tables for every command
- Added programmatic usage examples with Typer CliRunner and direct module access
- Cross-referenced to related guides (configuration, modeling, HPO, splitting, leaderboard)
- Added MLflow integration section
- Added testing section with best practices

### Phase 4: Stylistic Standardization

✅ **Task 4.1: Standardized guide opening lines across files**

- Updated 9 guide files to have direct, value-focused opening paragraphs
- Removed generic "This guide explains/covers" patterns
- Each guide now opens with what it does and why it matters:
  - curriculum.rst: "Curriculum learning progressively exposes models to data of increasing noise..."
  - splitting.rst: "Cluster-aware splitting prevents data leakage..."
  - classical_models.rst: "XGBoost, LightGBM, and CatBoost provide fast CPU-based baselines..."
  - hpo.rst: "Hyperparameter optimization systematically explores model configurations..."
  - profiling.rst: "The profiling system identifies performance bottlenecks... (~1%, ~5-10%, ~15-25% overhead)"
  - logging.rst: "Ray Tune logging captures trial output and uploads compressed archives..."
  - modeling.rst: "Model training uses Chemprop MPNNs, classical ML, and foundation models..."
  - task_affinity.rst: "Task Affinity Grouping identifies which tasks benefit from joint training..."
  - mlflow_artifacts.rst: "Ensemble training generates a hierarchical MLflow run structure..."

✅ **Task 4.2: Removed emoji and symbols from documentation**

- Already completed in Task 3.1 (debugging_per_quality_metrics.rst conversion)
- No other emoji found in RST files

✅ **Task 4.3: Standardized section headers and structure**

- Headers already consistent across documentation (= for titles, - for h2, ^ for h3)
- Removed redundant "Overview" section from task_affinity.rst that repeated title
- Consistent section naming throughout guides

### Phase 5: Visual Enhancements

✅ **Task 5.1: Simplified landing page design**

- Replaced cluttered 3-panel layout with clean, focused hero section
- Clear value proposition: "Build and evaluate ADMET prediction models..."
- Three primary user paths instead of mixed content:
  - Train Model → Quickstart Tutorial
  - Run HPO → HPO Guide
  - API Reference → Python API
- Removed installation code block from hero (moved to getting-started)
- Removed contributing section from hero (moved to Community toctree)

✅ **Task 5.2: Added workflow diagrams to key guides**

- Added ASCII workflow diagrams to 3 core guides:
  - quickstart.rst: Config → Data → Train → MLflow
  - modeling.rst: Data Split → Single Model → HPO → Ensemble → Submit
  - hpo.rst: Configure Search Space → Run Trials → Select Top K → Generate Ensemble Configs
- Each diagram includes step descriptions below for clarity
- Simple, readable ASCII format renders in all documentation formats

### Phase 6: Validation and Cleanup

✅ **Task 6.1: Built documentation and fixed all warnings**

- Ran `make -C docs clean && make -C docs html`
- Build completed successfully with zero warnings or errors
- All cross-references resolved correctly
- All toctree entries found and processed

✅ **Task 6.2: Verified cross-references and links**

- Ran `sphinx-build -b linkcheck` to verify links
- Found and documented broken links:
  - `docs/api/leaderboard.rst`: Invalid "https://" URL (minor, doesn't affect build)
  - `http://127.0.0.1:8080`: Local MLflow server (expected, not a real issue)
  - Some redirect URLs (working but redirect to versioned docs)
  - `docs/guide/hpo_warmstart.rst`: Ray Tune Optuna anchor not found
  - `docs/guide/logging.rst`: MLflow artifacts URL returns 403
- All internal :doc: and :ref: references working correctly
- Navigation structure intact across all documentation sections

✅ **Task 6.3: Tested user journeys for both audiences**

- **ML Practitioner Journey**: Home → Getting Started → Quickstart → HPO → Ensemble
  - Clear path from landing page through getting-started/index.rst to quickstart
  - HPO guide accessible from both quickstart and main navigation
  - Ensemble training documented in modeling.rst with cross-references
- **Domain Expert Journey**: Home → Data Pipeline → Data Sources → Quality Debugging
  - Data Pipeline section in navigation includes data sources and splitting
  - Quality debugging guide (debugging_per_quality_metrics.rst) in Data Pipeline section
  - Clear progression through data-focused documentation
- **No dead ends**: All pages have See Also sections with related links
- **No orphaned pages**: All 20+ guides accessible from main toctree

## Summary Statistics

**Files Created:**

- 3 new directories: docs/getting-started/, docs/reference/, docs/dev/
- 5 new content files: quickstart.rst, index.rst (getting-started), scripts.rst, planning.rst (dev), debugging_per_quality_metrics.rst
- 2 tracking files: changes.md, details.md

**Files Modified:**

- docs/index.rst: Reorganized toctree (flat → 8 semantic sections), simplified hero
- docs/guide/cli.rst: Expanded from 174 to 600+ lines (complete CLI reference)
- 9 guide files: Standardized opening lines (curriculum, splitting, classical_models, hpo, profiling, logging, modeling, task_affinity, mlflow_artifacts)
- 3 guide files: Added workflow diagrams (quickstart, modeling, hpo)

**Files Deleted:**

- docs/guide/debugging_per_quality_metrics.md (converted to RST)

**Build Status:**

- Sphinx build: ✓ Zero warnings
- Link check: ✓ Minor issues documented (no blockers)
- Navigation: ✓ All pages accessible
- User journeys: ✓ Both personas supported

**Total Changes:**

- 15 tasks across 6 phases
- 2000+ lines of new documentation
- 10+ files modified for consistency
- Full navigation restructure with semantic grouping
