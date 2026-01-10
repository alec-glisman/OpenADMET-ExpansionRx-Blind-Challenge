<!-- markdownlint-disable-file -->

# Task Research Notes: README.md Currency and Accuracy Analysis

## Status: ✅ COMPLETED

All identified issues have been resolved. The README, CLAUDE.md, and copilot-instructions.md have been updated with:

1. **Correct config paths** - All paths now match actual filesystem structure
2. **Missing CLI command** - `admet model hpo-list-studies` added to CLI Quick Reference
3. **Updated date** - January 2026 with current rank (Top 5.9%)
4. **Task-weighted loss** - Added to Training Strategy section
5. **Warmstart HPO** - Added to HPO section
6. **Updated Project Layout** - Matches actual config directory structure

## Research Executed

### File Analysis

- [README.md](README.md)
  - Comprehensive review of all sections for accuracy against current codebase
  - Identified multiple config path references that don't match actual filesystem structure
  - Found CLI commands missing from quick reference table
  - Identified outdated model card references

- [MODEL_CARD.md](MODEL_CARD.md)
  - Version 1.2.0, well-maintained with recent HPO findings
  - Curriculum learning documented as "not used in production"
  - FFN architecture evaluation documented

- [SUBMISSIONS.md](SUBMISSIONS.md)
  - Current through January 9, 2026 (Submission 0)
  - Documents task-weighted loss strategy and results
  - Best submission: Top 5.9% (rank 18/307) with MA-RAE 0.59

- [src/admet/cli/model.py](src/admet/cli/model.py)
  - Contains `hpo-list-studies` command not in README CLI table
  - Contains `train-chemprop` legacy command
  - All model types properly registered

### Code Search Results

- `configs/**/*.yaml`
  - 239 config files found in total
  - Actual structure differs from README references

- `configs/0-experiment/` actual structure:
  - `0-single-fold/` (chemprop.yaml, chemeleon.yaml)
  - `1-ensemble/` (chemeleon_production_ensemble.yaml, chemprop_ensemble_production.yaml)
  - `2-classical-models-ensemble/` (catboost.yaml, lightgbm.yaml, xgboost.yaml)
  - `curriculum-learning/` (chemprop_curriculum.yaml, chemprop_ensemble_curriculum.yaml)
  - `task-affinity/` (chemprop_single_hpo_task_affinity_groups.yaml)

- `configs/1-hpo-single-fold/` actual structure:
  - hpo_chemeleon.yaml
  - hpo_chemprop.yaml
  - hpo_chemprop_warmstart_example.yaml
  - phases/ (subdirectory)

- `configs/3-hpo-ensemble-production/` actual structure:
  - `0_chemprop_v1/`
  - `1_chemeleon_v1/`
  - `2_task_weighted_ensemble/` (task_weighted_ensemble.yaml)
  - `3_hybrid_ensemble/`

### Project Conventions

- Standards referenced: copilot-instructions.md, CLAUDE.md
- Config directory naming convention: numbered prefixes (0-, 1-, 2-, 3-)
- Production configs in `3-hpo-ensemble-production/`

## Key Discoveries

### 1. Incorrect Config Path References in README

**Current README paths that DON'T exist:**

| README Reference | Actual Location |
|------------------|-----------------|
| `configs/0-experiment/chemprop.yaml` | `configs/0-experiment/0-single-fold/chemprop.yaml` |
| `configs/3-production/ensemble.yaml` | `configs/3-hpo-ensemble-production/` (various) |
| `configs/1-hpo-single/hpo_chemprop.yaml` | `configs/1-hpo-single-fold/hpo_chemprop.yaml` |
| `configs/4-more-models/` | `configs/0-experiment/2-classical-models-ensemble/` |
| `configs/task-affinity/` | `configs/0-experiment/task-affinity/` |

### 2. Missing CLI Commands in Quick Reference

Commands in `src/admet/cli/model.py` missing from README:

| Missing Command | Purpose |
|-----------------|---------|
| `admet model hpo-list-studies` | List available Optuna studies for warmstart |
| `admet model train-chemprop` | Legacy direct Chemprop training |

### 3. Config Directory Structure Mismatch

**README Project Layout shows:**
```
configs/
├── 0-experiment/
├── 1-hpo-single/
├── 2-hpo-ensemble/
├── 3-production/
├── 4-more-models/
├── curriculum/
└── task-affinity/
```

**Actual structure:**
```
configs/
├── 0-experiment/
│   ├── 0-single-fold/
│   ├── 1-ensemble/
│   ├── 2-classical-models-ensemble/
│   ├── curriculum-learning/
│   └── task-affinity/
├── 1-hpo-single-fold/
├── 2-hpo-ensemble/
└── 3-hpo-ensemble-production/
    ├── 0_chemprop_v1/
    ├── 1_chemeleon_v1/
    ├── 2_task_weighted_ensemble/
    └── 3_hybrid_ensemble/
```

### 4. Model Submission Updates

The README currently links to MODEL_CARD.md and SUBMISSIONS.md correctly. The SUBMISSIONS.md is current through January 9, 2026 with:
- Best rank: 18/307 (Top 5.9%)
- MA-RAE: 0.59 ± 0.02
- Task-weighted loss strategy documented

### 5. New Features Not Prominently Documented

#### Task-Weighted Loss

The January 9 submission uses task-weighted loss based on leaderboard analysis:
- Configurable via `target_weights` in data config
- Weights derived from per-task ranking analysis
- Example weights: LogD=1.5, KSOL=0.7, Caco-2 Efflux=1.8

This feature IS documented in MODEL_CARD.md but not prominently in README.

#### Warmstart HPO

The `hpo-list-studies` CLI command supports warmstart:
```yaml
search_algorithm:
  type: optuna
  persist_study: true
  warmstart_from: '<study_name>'
```

This is documented in `docs/guide/hpo_warmstart.rst` but not in README HPO section.

### 6. Outdated Information

- README mentions "configs/4-more-models/" which doesn't exist
- CLI Quick Reference table is incomplete
- Project Layout section doesn't match actual config structure
- Version references may need updates (Chemprop v2.2.1 mentioned)

## Recommended Approach

**Update README.md with the following changes:**

### Priority 1: Fix Config Path References (Breaking)

1. Update CLI Quick Reference examples with correct paths
2. Fix all `configs/` path references throughout document
3. Update Project Layout section to match actual structure

### Priority 2: Add Missing CLI Commands

Add to CLI Quick Reference:
- `admet model hpo-list-studies` - List Optuna studies for warmstart

### Priority 3: Update Project Layout

Replace current config layout with accurate structure.

### Priority 4: Feature Documentation (Optional)

Consider adding brief mention of:
- Task-weighted loss (link to MODEL_CARD.md)
- Warmstart HPO (link to docs)
- Hybrid ensemble capabilities

## Implementation Guidance

- **Objectives**: Ensure README accurately reflects current codebase state
- **Key Tasks**:
  1. Update all config path references
  2. Add missing CLI command to table
  3. Update Project Layout section
  4. Consider adding brief feature highlights for task-weighted loss
- **Dependencies**: None - documentation-only changes
- **Success Criteria**: All config paths in README correspond to existing files

## Summary of Required Changes

### Must Fix (Accuracy Issues)

| Section | Issue | Fix |
|---------|-------|-----|
| CLI Quick Reference | Wrong config paths | Update to correct paths |
| CLI Quick Reference | Missing `hpo-list-studies` | Add new row |
| Model Training | `configs/0-experiment/chemprop.yaml` | `configs/0-experiment/0-single-fold/chemprop.yaml` |
| Ensemble Training | `configs/3-production/ensemble.yaml` | `configs/0-experiment/1-ensemble/chemprop_ensemble_production.yaml` or `configs/3-hpo-ensemble-production/...` |
| HPO | `configs/1-hpo-single/hpo_chemprop.yaml` | `configs/1-hpo-single-fold/hpo_chemprop.yaml` |
| Config Examples | `configs/4-more-models/` | `configs/0-experiment/2-classical-models-ensemble/` |
| Task Affinity | `configs/task-affinity/` | `configs/0-experiment/task-affinity/` |
| Project Layout | Entire configs section | Needs restructure to match actual |

### Nice to Have (Enhancements)

| Enhancement | Benefit |
|-------------|---------|
| Add task-weighted loss mention | Documents recent submission strategy |
| Add warmstart HPO mention | Documents advanced HPO capability |
| Link to latest submission stats | Shows current performance |
