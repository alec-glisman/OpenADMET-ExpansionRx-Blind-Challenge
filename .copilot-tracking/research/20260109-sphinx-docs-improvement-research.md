# Task Research Notes: Sphinx Documentation Improvement

## Research Executed

### File Analysis

- `docs/index.rst` - Main landing page with 3-panel layout, 18 guide pages in toctree
- `docs/conf.py` - Furo theme with Merck-inspired color palette, sphinx_panels enabled
- `docs/guide/*.rst` - 20 guide files with varying quality and consistency
- `docs/_static/css/custom.css` - Custom CSS with professional color variables
- `src/admet/cli/*.py` - 4 CLI modules (main, data, model, leaderboard)
- `scripts/*.sh` - 14+ shell scripts with comprehensive documentation in scripts/README.md
- `configs/` - 4-tier config structure (experiment → HPO → ensemble → production)

### Current Documentation Structure Assessment

**Strengths:**

- Furo theme is clean and professional
- Good API autodoc setup with NumPy docstrings
- Comprehensive config_reference.rst (889 lines)
- Strong technical depth in curriculum, task_affinity guides

**Weaknesses:**

- Inconsistent guide organization (no clear user journey)
- Landing page panels link to incomplete destinations
- CLI documentation fragmented between cli.rst and scripts/README.md
- Planning section exposed in main navigation (internal docs visible)
- No quickstart guide for the target audience (ML/drug discovery practitioners)
- Debugging guide as markdown file breaks visual consistency
- Emoji use in some files (debugging_per_quality_metrics.md uses ✅ ✓ symbols)
- Repeated phrases across guides ("This guide explains...", "Overview", "Quick Start")
- Missing visual hierarchy - all guides appear equal weight
- No workflow diagrams in documentation

### Target Audience Analysis

**ML Practitioners want:**

- Quick path to training their first model
- Understanding of HPO workflow and best practices
- Clear configuration reference
- Troubleshooting guidance

**Drug Discovery Experts want:**

- Understanding of ADMET endpoints
- Interpretation of model predictions
- Quality tier explanations
- Uncertainty quantification

### External Research

- Furo theme best practices: Clean sidebar, minimal customization, semantic sections
- Professional documentation patterns: Jupyter Book, scikit-learn, PyTorch documentation
- Drug discovery documentation: ChEMBL docs, RDKit cookbook style

## Key Discoveries

### Project Structure

The documentation has 18+ guides but lacks:

1. Clear entry point for each audience type
2. Progressive disclosure (basics → intermediate → advanced)
3. Workflow-oriented organization

### Current Guide Categories (Implicit)

| Category | Guides | Notes |
|----------|--------|-------|
| Getting Started | overview, development, cli | Fragmented |
| Data Pipeline | data_sources, splitting | Good but hidden |
| Training | modeling, classical_models, curriculum, task_affinity | Strong content, poor discovery |
| HPO | hpo, hpo_warmstart | Good but disconnected from training |
| Operations | profiling, logging, mlflow_artifacts | Scattered |
| Reference | configuration, config_reference | Overlapping content |
| Internal | debugging_per_quality_metrics, planning/* | Should not be in main nav |

### Stylistic Issues Found

1. **Inconsistent opening lines:**
   - "This guide covers..." (curriculum.rst)
   - "This guide explains..." (splitting.rst)
   - "This page gives..." (architecture.rst)
   - "Welcome to..." (overview.rst)

2. **Redundant section headers:**
   - Nearly every guide has "Overview", "Quick Start", "Configuration"
   - Creates cognitive load when skimming

3. **Mixed formatting:**
   - Some guides use `.. contents::` TOC, others don't
   - Inconsistent heading levels
   - debugging_per_quality_metrics.md uses markdown (inconsistent with RST)

4. **Emoji/symbol usage:**
   - debugging_per_quality_metrics.md: "✅ Added", "✓ correct"
   - Should be removed for professional tone

### Information Architecture Issues

1. **No clear user journey:**
   - New user lands on index.rst
   - Sees 18 guides in alphabetical-ish order
   - No guidance on where to start

2. **CLI docs fragmented:**
   - `guide/cli.rst` has basic usage
   - `scripts/README.md` has 500 lines of detailed script documentation
   - Not integrated into Sphinx

3. **Planning docs exposed:**
   - Internal planning documents visible in main navigation
   - Should be hidden or moved to developer section

## Recommended Approach

### Information Architecture Restructure

Reorganize documentation into clear sections with progressive disclosure:

```
Documentation
├── Getting Started (NEW)
│   ├── Installation
│   ├── Quickstart Tutorial
│   └── Architecture Overview
├── User Guide
│   ├── Data Preparation
│   │   ├── Data Sources
│   │   └── Dataset Splitting
│   ├── Model Training
│   │   ├── Chemprop Models
│   │   ├── Classical Models
│   │   └── Ensemble Training
│   ├── Advanced Training
│   │   ├── Hyperparameter Optimization
│   │   ├── Warmstarting HPO
│   │   ├── Curriculum Learning
│   │   └── Task Affinity Grouping
│   └── Operations
│       ├── MLflow Integration
│       ├── Performance Profiling
│       └── Logging & Monitoring
├── Reference
│   ├── CLI Reference
│   ├── Configuration Reference
│   ├── Shell Scripts Reference (NEW from scripts/README.md)
│   └── API Reference
└── Development
    ├── Contributing
    └── Internal Notes (planning docs, hidden from public)
```

### Visual Design Improvements

1. **Simplify landing page:**
   - Remove sphinx_panels 3-column layout (cluttered)
   - Add hero section with clear value proposition
   - Feature 3 paths: "I want to train a model", "I want to run HPO", "I want to deploy"

2. **Add workflow diagrams:**
   - Data → Split → Train → Evaluate → Submit pipeline
   - HPO → Analyze → Ensemble workflow
   - Use Mermaid diagrams (already in README.md)

3. **Consistent guide structure:**
   - Remove repetitive "Overview" sections
   - Start each guide with a single clear sentence
   - Use admonitions sparingly (tip, warning, note)

### Content Improvements

1. **Create Quickstart Tutorial:**
   - End-to-end example in 5 minutes
   - Train a model, view results in MLflow
   - For both ML practitioners and domain experts

2. **Integrate scripts documentation:**
   - Move scripts/README.md content into Sphinx
   - Create `reference/scripts.rst` with full documentation

3. **Convert markdown to RST:**
   - debugging_per_quality_metrics.md → .rst
   - Remove emoji symbols

4. **Hide planning section:**
   - Move to `dev/` subdirectory
   - Or exclude from toctree entirely

### Stylistic Guidelines

1. **Guide opening lines:**
   - Start with purpose, not meta-description
   - Bad: "This guide explains how to..."
   - Good: "Dataset splitting ensures representative train/validation distributions..."

2. **Section headers:**
   - Use descriptive headers, not generic ones
   - Bad: "Configuration"
   - Good: "YAML Configuration Format"

3. **Code examples:**
   - Always show both CLI and Python API when available
   - Include expected output for key commands

4. **Admonitions:**
   - Use sparingly (max 2-3 per page)
   - Reserve `warning` for breaking changes or data loss
   - Use `tip` for best practices

## Implementation Guidance

### Phase 1: Structure (High Impact)

- Reorganize toctree into logical sections
- Create new index.rst with clear user journeys
- Hide planning section

### Phase 2: Content (Medium Impact)

- Create quickstart tutorial
- Integrate scripts documentation
- Convert markdown files to RST

### Phase 3: Polish (Lower Impact)

- Standardize guide openings
- Remove emoji/symbols from all files
- Add workflow diagrams

### Success Criteria

- New user can start training in &lt;10 minutes
- All CLI commands documented in Sphinx
- No internal documents visible in public navigation
- Consistent visual style across all pages

## Files to Create/Modify

**New Files:**

- `docs/getting-started/index.rst` - Getting started section
- `docs/getting-started/quickstart.rst` - 5-minute tutorial
- `docs/reference/scripts.rst` - Scripts documentation

**Modify:**

- `docs/index.rst` - New landing page structure
- `docs/conf.py` - Exclude patterns for planning
- `docs/guide/debugging_per_quality_metrics.md` → `.rst` conversion

**Hide/Move:**

- `docs/planning/*` - Exclude from toctree or move to dev section
