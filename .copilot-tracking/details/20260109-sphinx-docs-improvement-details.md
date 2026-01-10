<!-- markdownlint-disable-file -->

# Task Details: Sphinx Documentation Improvement

## Research Reference

**Source Research**: #file:../research/20260109-sphinx-docs-improvement-research.md

## Phase 1: Information Architecture Restructure

### Task 1.1: Create new directory structure for documentation sections

Create organized directory structure to support semantic grouping of documentation.

- **Files**:
  - `docs/getting-started/` - New directory for onboarding content
  - `docs/reference/` - New directory for reference documentation
  - `docs/dev/` - New directory for development/internal docs
- **Success**:
  - Directories exist and are recognized by Sphinx
  - Each directory has an index.rst file
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 120-145) - Information architecture proposal
- **Dependencies**:
  - None (first task)

### Task 1.2: Reorganize index.rst with semantic toctree sections

Restructure the main index.rst to use logical groupings instead of flat list.

- **Files**:
  - `docs/index.rst` - Complete rewrite of toctree structure
- **Success**:
  - Toctree organized into: Getting Started, User Guide, Reference, Development
  - Each section has clear caption and logical ordering
  - Progressive disclosure from basics to advanced
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 120-145) - Proposed structure
- **Dependencies**:
  - Task 1.1 completion (directories must exist)

**New toctree structure:**

```rst
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting-started/index
   getting-started/quickstart
   guide/architecture

.. toctree::
   :maxdepth: 1
   :caption: Data Pipeline

   guide/data_sources
   guide/splitting

.. toctree::
   :maxdepth: 1
   :caption: Model Training

   guide/modeling
   guide/classical_models
   guide/hpo
   guide/hpo_warmstart

.. toctree::
   :maxdepth: 1
   :caption: Advanced Training

   guide/curriculum
   guide/task_affinity

.. toctree::
   :maxdepth: 1
   :caption: Operations

   guide/mlflow_artifacts
   guide/profiling
   guide/logging

.. toctree::
   :maxdepth: 1
   :caption: Reference

   guide/cli
   guide/configuration
   guide/config_reference
   reference/scripts
   api/admet

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   guide/development
   dev/planning
```

### Task 1.3: Hide planning section from public navigation

Move internal planning documents out of main navigation.

- **Files**:
  - `docs/dev/planning.rst` - New wrapper for planning content
  - `docs/index.rst` - Update toctree to use :hidden: for dev section
  - `docs/conf.py` - Optional: add exclude patterns
- **Success**:
  - Planning docs not visible in main sidebar navigation
  - Still accessible via direct link for developers
  - No broken internal links
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 108-112) - Planning docs issue
- **Dependencies**:
  - Task 1.1 (dev directory exists)

## Phase 2: New Content Creation

### Task 2.1: Create quickstart tutorial (getting-started/quickstart.rst)

Write a 5-minute end-to-end tutorial for training first model.

- **Files**:
  - `docs/getting-started/quickstart.rst` - New quickstart tutorial
- **Success**:
  - Complete working example from config to MLflow results
  - Both CLI and Python API examples
  - Expected output shown for key commands
  - Estimated reading time under 5 minutes
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 170-175) - Quickstart requirements
- **Dependencies**:
  - Task 1.1 (getting-started directory exists)

**Quickstart structure:**

```rst
Quickstart: Train Your First Model
==================================

Train a Chemprop model on ADMET data in under 5 minutes.

Prerequisites
-------------
- Python 3.11 environment with package installed
- Sample configuration file

Step 1: Prepare Configuration
-----------------------------
[minimal YAML config example]

Step 2: Train Model
-------------------
[CLI command with expected output]
[Python API equivalent]

Step 3: View Results in MLflow
------------------------------
[MLflow UI screenshot or description]

Next Steps
----------
- Explore ensemble training: :doc:`/guide/modeling`
- Run hyperparameter optimization: :doc:`/guide/hpo`
```

### Task 2.2: Create shell scripts reference (reference/scripts.rst)

Migrate scripts/README.md content into Sphinx documentation.

- **Files**:
  - `docs/reference/scripts.rst` - Comprehensive scripts reference
- **Success**:
  - All 14+ shell scripts documented
  - Usage examples with options tables
  - Cross-references to related guides
  - Searchable within Sphinx
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 108-112) - CLI fragmentation issue
  - scripts/README.md - Source content (500 lines)
- **Dependencies**:
  - Task 1.1 (reference directory exists)

**Scripts to document:**

| Script | Purpose |
|--------|---------|
| train_chemprop_ensembles.sh | Ensemble training with Ray |
| train_chemprop_model.sh | Single model training |
| train_chemprop_hpo.sh | HPO with Ray Tune |
| train_production_ensembles.sh | Production deployment |
| run_data_splits.sh | Dataset splitting pipeline |
| generate_ensemble_configs.py | Config generation from HPO |

### Task 2.3: Create getting-started section index

Write index page for getting-started section.

- **Files**:
  - `docs/getting-started/index.rst` - Section index with overview
- **Success**:
  - Clear value proposition for new users
  - Links to quickstart, installation, architecture
  - Audience-specific paths (ML vs domain expert)
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 48-58) - Target audience analysis
- **Dependencies**:
  - Task 1.1 (directory exists)

## Phase 3: Content Migration and Conversion

### Task 3.1: Convert debugging_per_quality_metrics.md to RST

Convert markdown file to RST and remove emoji symbols.

- **Files**:
  - `docs/guide/debugging_per_quality_metrics.rst` - Converted RST file
  - Delete: `docs/guide/debugging_per_quality_metrics.md`
- **Success**:
  - Valid RST syntax with proper code blocks
  - All emoji (✅ ✓) removed
  - Consistent styling with other guides
  - No broken cross-references
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 92-95) - Emoji issue
- **Dependencies**:
  - None

**Emoji removal targets:**

- "✅ Added" → "Added"
- "✓ correct" → "(correct)"
- "1. ✅" → "1."

### Task 3.2: Update CLI documentation with complete reference

Expand CLI documentation to be comprehensive reference.

- **Files**:
  - `docs/guide/cli.rst` - Enhanced CLI documentation
- **Success**:
  - All CLI commands documented (data, model, leaderboard)
  - All subcommands with options tables
  - Examples for common workflows
  - Links to scripts reference
- **Research References**:
  - src/admet/cli/*.py - CLI source modules
- **Dependencies**:
  - Task 2.2 (scripts reference to link to)

## Phase 4: Stylistic Standardization

### Task 4.1: Standardize guide opening lines across all files

Rewrite opening paragraphs to be direct and purpose-focused.

- **Files**:
  - All files in `docs/guide/*.rst` (20 files)
- **Success**:
  - No guides start with "This guide explains..."
  - Each guide opens with purpose statement
  - First paragraph describes value, not structure
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 82-89) - Inconsistent openings
- **Dependencies**:
  - None

**Opening line transformations:**

| File | Current | Proposed |
|------|---------|----------|
| curriculum.rst | "This guide covers quality-aware curriculum learning..." | "Curriculum learning progressively exposes models to data of increasing noise, improving robustness." |
| splitting.rst | "This guide describes how datasets are partitioned..." | "Cluster-aware splitting prevents data leakage by keeping similar molecules together." |
| classical_models.rst | "This guide explains how to use traditional ML models..." | "XGBoost, LightGBM, and CatBoost provide fast CPU-based baselines using molecular fingerprints." |
| hpo.rst | "This guide covers hyperparameter optimization..." | "Hyperparameter optimization systematically explores model configurations using Ray Tune and ASHA." |
| profiling.rst | "This guide explains how to use the comprehensive profiling..." | "The profiling system identifies bottlenecks in ensemble training with minimal overhead." |
| task_affinity.rst | Already good structure | Minor polish only |
| logging.rst | "This guide explains the Ray Tune logging infrastructure..." | "Ray Tune logging captures trial output and uploads compressed archives to MLflow." |
| mlflow_artifacts.rst | Already direct | No change needed |

### Task 4.2: Remove emoji and symbols from documentation

Find and remove all emoji and decorative symbols.

- **Files**:
  - `docs/guide/debugging_per_quality_metrics.rst` - Primary target
  - Any other files with emoji
- **Success**:
  - No ✅ ✓ ❌ or similar symbols in documentation
  - Professional tone maintained
  - Build produces no encoding warnings
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 92-95) - Emoji issue
- **Dependencies**:
  - Task 3.1 (file converted to RST)

### Task 4.3: Standardize section headers and structure

Ensure consistent heading hierarchy and section names.

- **Files**:
  - All files in `docs/guide/*.rst`
- **Success**:
  - Consistent heading underline characters (= for title, - for h2, ^ for h3)
  - No generic "Overview" sections that repeat guide title
  - Descriptive section names (not just "Configuration")
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 86-91) - Redundant headers
- **Dependencies**:
  - None

**Header standardization:**

- Title: `=====` (overline and underline)
- H2: `-----` (underline only)
- H3: `^^^^^` (underline only)
- H4: `"""""` (underline only)

## Phase 5: Visual Enhancements

### Task 5.1: Simplify landing page design

Replace cluttered 3-panel layout with clean user journey design.

- **Files**:
  - `docs/index.rst` - Simplified hero section
  - `docs/_static/css/custom.css` - Optional styling adjustments
- **Success**:
  - Clear value proposition in first viewport
  - Three user paths: Train Model, Run HPO, Deploy Ensemble
  - Reduced visual clutter
  - Mobile-friendly layout
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 150-160) - Landing page issues
- **Dependencies**:
  - Task 1.2 (toctree reorganized)

**Proposed landing page structure:**

```rst
OpenADMET Challenge
===================

Build and evaluate ADMET prediction models with state-of-the-art
graph neural networks, systematic hyperparameter optimization, and
robust ensemble training.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Train Your First Model
      :link: getting-started/quickstart
      :link-type: doc

      Get started in 5 minutes with a complete training example.

   .. grid-item-card:: Run HPO
      :link: guide/hpo
      :link-type: doc

      Systematically optimize hyperparameters with Ray Tune.

   .. grid-item-card:: API Reference
      :link: api/admet
      :link-type: doc

      Explore the complete Python API documentation.
```

### Task 5.2: Add workflow diagrams to key guides

Add Mermaid or ASCII diagrams to illustrate workflows.

- **Files**:
  - `docs/guide/modeling.rst` - Training pipeline diagram
  - `docs/guide/hpo.rst` - HPO workflow diagram
  - `docs/getting-started/quickstart.rst` - Simple pipeline diagram
- **Success**:
  - Key workflows visualized
  - Diagrams render correctly in HTML
  - Accessible alternative text provided
- **Research References**:
  - README.md - Existing Mermaid diagrams to adapt
- **Dependencies**:
  - Phase 2 complete (new content exists)

**Diagram for modeling.rst:**

```
Data → Split → Train → Evaluate → Submit
         ↓
    HPO ─→ Ensemble
```

## Phase 6: Validation and Cleanup

### Task 6.1: Build documentation and fix all warnings

Clean build with zero Sphinx warnings.

- **Files**:
  - All documentation files
  - `docs/conf.py` - Adjust suppress_warnings if needed
- **Success**:
  - `make -C docs html` produces zero warnings
  - All cross-references resolve
  - No missing files or broken includes
- **Research References**:
  - None (validation task)
- **Dependencies**:
  - All previous phases complete

**Build command:**

```bash
source .venv/bin/activate && make -C docs clean && make -C docs html 2>&1 | grep -E "warning|error"
```

### Task 6.2: Verify cross-references and links

Check all internal and external links.

- **Files**:
  - All documentation files
- **Success**:
  - All `:doc:` references resolve
  - All `:ref:` references resolve
  - External URLs accessible
  - No 404 errors in link check
- **Research References**:
  - None (validation task)
- **Dependencies**:
  - Task 6.1 complete

**Link check command:**

```bash
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

### Task 6.3: Test user journeys for both audiences

Manual verification of documentation paths.

- **Files**:
  - None (manual testing)
- **Success**:
  - ML practitioner can navigate: Home → Quickstart → HPO → Ensemble
  - Domain expert can navigate: Home → Data Sources → Endpoints → Quality Tiers
  - All navigation paths logical and complete
  - No dead ends or orphaned pages
- **Research References**:
  - #file:../research/20260109-sphinx-docs-improvement-research.md (Lines 48-58) - Audience analysis
- **Dependencies**:
  - All previous tasks complete

**Test scenarios:**

1. New ML user: Can train model in 10 minutes?
2. Returning user: Can find CLI reference quickly?
3. Domain expert: Can understand quality tiers?
4. Contributor: Can find development guide?

## Dependencies

- Sphinx 7.x with Furo theme
- MyST parser for markdown support
- sphinx_panels extension (or sphinx-design for grids)

## Success Criteria

- Documentation builds with zero warnings
- New user can train first model in under 10 minutes following quickstart
- All CLI commands and scripts documented within Sphinx
- No internal planning documents visible in public navigation
- Consistent visual styling across all 20+ guide pages
