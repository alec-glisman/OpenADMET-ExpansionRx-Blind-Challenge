# Branch Merge Plan: Consolidating Joint Sampling Fix with Backup Features

**Created:** 2025-12-24
**Author:** Claude Code Assistant
**Goal:** Merge `backup-main-20251224` into `fix/joint_sampling`, then merge into `main`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Branch Overview](#branch-overview)
3. [Pre-Merge Checklist](#pre-merge-checklist)
4. [Phase 1: Preparation](#phase-1-preparation)
5. [Phase 2: Merge Execution](#phase-2-merge-execution)
6. [Phase 3: Conflict Resolution Guide](#phase-3-conflict-resolution-guide)
7. [Phase 4: Validation Tiers](#phase-4-validation-tiers)
8. [Phase 5: Final Merge to Main](#phase-5-final-merge-to-main)
9. [Rollback Procedures](#rollback-procedures)
10. [Appendix: File Change Summary](#appendix-file-change-summary)

---

## Executive Summary

### Current Situation

| Branch | Purpose | Commits after merge-base | Lines Changed |
|--------|---------|-------------------------|---------------|
| `fix/joint_sampling` | Critical bug fix for two-stage JointSampler | 6 | +646 / -96 |
| `backup-main-20251224` | Major refactor with MLOps infrastructure | 13 | +5,877 / -343 |
| `main` | Production baseline (v1.1.0) | 0 (ancestor) | — |

### Merge Strategy

**Recommended approach:** Merge `backup-main-20251224` INTO `fix/joint_sampling`

**Rationale:**

- Preserves your JointSampler fix as the authoritative version
- Conflicts default to your fix branch's resolution
- Easier to review what new features are being added vs. protecting your fix

### Time Estimates

| Phase | Duration | Can Skip? |
|-------|----------|-----------|
| Preparation | 5 min | No |
| Merge & Conflict Resolution | 15-45 min | No |
| Tier 1-2 Validation | 5-10 min | No |
| Tier 3 Smoke Test | 10-15 min | If Tier 2 passes completely |
| Tier 4 Full Validation | ~1 hour | Run overnight or when confident |
| Final merge to main | 2 min | No |

---

## Branch Overview

### What `fix/joint_sampling` Contains (YOUR FIX)

Key commits:

```
d633886 Update random seed in configuration files to improve reproducibility
1697b86 Merge commit '7e49f73' into fix/joint_sampling
7e49f73 feat/fix: Implement two-stage sampling in JointSampler and comprehensive tests
9a62eb2 feat: Log full ensemble configuration as YAML artifact in MLflow
dc38735 fix: Ensure reproducibility by seeding random number generators
```

Critical files:

- `src/admet/model/chemprop/joint_sampler.py` (340 lines - **PROTECT THIS**)
- `src/admet/model/chemprop/ensemble.py` (YAML logging improvements)
- `src/admet/model/chemprop/model.py` (seeding improvements)

### What `backup-main-20251224` Adds

New infrastructure:

- `src/admet/model/base.py` - Base model abstraction (148 lines)
- `src/admet/model/config.py` - Unified config system (379 lines)
- `src/admet/model/registry.py` - Model registry (147 lines)
- `src/admet/model/mlflow_mixin.py` - MLflow integration (283 lines)
- `src/admet/model/ensemble.py` - Generic ensemble framework (322 lines)
- `src/admet/model/chemeleon/` - Chemeleon model support (new directory)
- `src/admet/model/classical/` - XGBoost, LightGBM, CatBoost models
- `src/admet/model/hpo/` - Expanded HPO infrastructure

New tests (~3,600 lines):

- `tests/test_chemeleon_model.py`
- `tests/test_classical_models.py`
- `tests/test_determinism.py`
- `tests/test_fingerprints.py`
- `tests/test_model_base.py`
- And more...

---

## Pre-Merge Checklist

Before starting, verify:

- [ ] Working directory is clean (`git status` shows no changes)
- [ ] You are on `fix/joint_sampling` branch
- [ ] All current tests pass (`pytest tests/ -x -q`)
- [ ] You have at least 1GB free disk space
- [ ] MLflow server is accessible (if running validation tier 3+)
- [ ] No other processes are using the repository

---

## Phase 1: Preparation

### Step 1.1: Verify Current State

```bash
# Check you're on the right branch with clean state
git status
git branch --show-current  # Should output: fix/joint_sampling
```

### Step 1.2: Run Baseline Tests

```bash
# Run all tests to establish baseline
pytest tests/ -x -q --tb=short

# If any fail, STOP and fix them before proceeding
```

### Step 1.3: Create Safety Backups

```bash
# Create a timestamped backup branch
git branch backup-fix-joint-sampling-$(date +%Y%m%d-%H%M%S)

# Verify backup was created
git branch | grep backup-fix

# Optional: Push backup to remote for extra safety
git push origin backup-fix-joint-sampling-$(date +%Y%m%d-%H%M%S)
```

### Step 1.4: Document Current Commit

```bash
# Record the current HEAD for potential rollback
echo "Pre-merge HEAD: $(git rev-parse HEAD)" > /tmp/merge-rollback-point.txt
cat /tmp/merge-rollback-point.txt
```

---

## Phase 2: Merge Execution

### Step 2.1: Fetch Latest Remote State

```bash
git fetch origin
```

### Step 2.2: Perform Merge (No Auto-Commit)

```bash
# Merge without auto-committing to inspect changes
git merge backup-main-20251224 --no-commit --no-ff
```

**Expected output:** Either "Automatic merge went well" or conflict notices.

### Step 2.3: Identify Conflicts

```bash
# List files with conflicts
git diff --name-only --diff-filter=U

# Get summary of all changes staged
git diff --cached --stat
```

### Step 2.4: Review Changes Before Committing

```bash
# See what will be committed
git status

# Review specific file changes
git diff --cached -- src/admet/model/chemprop/joint_sampler.py
git diff --cached -- src/admet/model/chemprop/model.py
git diff --cached -- src/admet/model/chemprop/ensemble.py
```

---

## Phase 3: Conflict Resolution Guide

### Priority Order for Conflict Resolution

Resolve conflicts in this order (highest risk first):

#### 3.1 CRITICAL: `joint_sampler.py`

**Strategy:** Keep YOUR version entirely

```bash
# If conflicted, use your version
git checkout --ours src/admet/model/chemprop/joint_sampler.py
git add src/admet/model/chemprop/joint_sampler.py
```

**Why:** This is your bug fix. The backup branch may have a different/broken implementation.

#### 3.2 HIGH RISK: `chemprop/model.py`

**Strategy:** Manual merge required

Both branches have changes:

- Your branch: Seeding improvements, reproducibility fixes
- Backup branch: Enhanced logging, metrics handling

```bash
# Open in editor and manually resolve
code src/admet/model/chemprop/model.py

# After manual resolution
git add src/admet/model/chemprop/model.py
```

**Key sections to protect:**

- Random seed initialization
- Any code referencing `JointSampler`
- Reproducibility-related changes

#### 3.3 HIGH RISK: `chemprop/ensemble.py`

**Strategy:** Keep your YAML logging, accept backup's other improvements

Your changes to protect:

- YAML artifact logging to MLflow
- Seed configuration logging

```bash
# Review diff carefully
git diff HEAD...backup-main-20251224 -- src/admet/model/chemprop/ensemble.py
```

#### 3.4 MEDIUM RISK: `chemprop/curriculum.py`

**Strategy:** Accept backup's improvements, verify compatibility

Backup adds:

- Enhanced sampling logging
- Count normalization
- Phase weight updates

```bash
# If no conflicts, accept backup version
git checkout --theirs src/admet/model/chemprop/curriculum.py
git add src/admet/model/chemprop/curriculum.py
```

#### 3.5 MEDIUM RISK: `chemprop/config.py`

**Strategy:** Manual review needed

Both branches have configuration changes. Ensure:

- JointSampling config options are preserved
- New config options from backup are included

#### 3.6 LOW RISK: New Files from Backup

**Strategy:** Accept all new files

These are additive and won't conflict:

```bash
# These should auto-merge without issues
src/admet/model/base.py
src/admet/model/config.py
src/admet/model/registry.py
src/admet/model/mlflow_mixin.py
src/admet/model/ensemble.py
src/admet/model/chemeleon/*
src/admet/model/classical/*
src/admet/model/hpo/*
```

### Conflict Resolution Commands Reference

```bash
# Keep your version (ours = fix/joint_sampling)
git checkout --ours <file>

# Keep their version (theirs = backup-main-20251224)
git checkout --theirs <file>

# Mark as resolved after manual edit
git add <file>

# See conflict markers in a file
grep -n "<<<<<<" <file>

# Abort merge if things go wrong
git merge --abort
```

---

## Phase 4: Validation Tiers

### Tier 1: Syntax & Import Check (~30 seconds)

```bash
# Test that core modules can be imported
python -c "
from admet.model.chemprop import ensemble, model
from admet.model.chemprop.joint_sampler import JointSampler
print('✓ Core imports successful')
"
```

**If this fails:** You have syntax errors or broken imports. Check the error message and fix.

### Tier 2: Unit Tests (~5-10 minutes)

```bash
# Run critical tests first (your fix)
pytest tests/test_joint_sampler.py -v

# Run curriculum tests (related functionality)
pytest tests/test_curriculum*.py -v

# Run all model tests
pytest tests/test_chemprop*.py tests/test_ensemble*.py -v

# Run full test suite
pytest tests/ -x --tb=short
```

**Pass criteria:** All tests pass (0 failures)

**If tests fail:**

1. Note which tests fail
2. Check if failure is due to merge conflict resolution
3. Review the specific file changes for those tests

### Tier 3: Smoke Test (~10-15 minutes)

Create a minimal test configuration:

```bash
# Create smoke test config
cat > configs/test-merge-validation.yaml << 'EOF'
data:
  data_dir: assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data
  splits: null
  folds: [0]  # Only use first fold
  test_file: assets/dataset/set/local_test.csv
  blind_file: assets/dataset/set/blind_test.csv
  output_dir: null
  smiles_col: SMILES
  target_cols:
    - LogD
    - Log KSOL
  target_weights:
    - 1.0
    - 1.0
model:
  depth: 2
  message_hidden_dim: 100
  aggregation: norm
  ffn_type: regression
  num_layers: 2
  hidden_dim: 50
  dropout: 0.1
  batch_norm: true
optimization:
  criterion: MAE
  init_lr: 0.001
  max_lr: 0.001
  final_lr: 0.0001
  warmup_epochs: 1
  max_epochs: 3
  patience: 2
  batch_size: 64
  num_workers: 0
  seed: 42
  progress_bar: true
mlflow:
  tracking: false
inter_task_affinity:
  enabled: false
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.02
  curriculum:
    enabled: false
  seed: 42
  increment_seed_per_epoch: true
  log_to_mlflow: false
ray:
  max_parallel: 1
EOF
```

Run smoke test:

```bash
# Run with minimal config
python -m admet.model.chemprop.ensemble --config configs/test-merge-validation.yaml

# Expected: Should complete without errors in ~10 minutes
```

**Pass criteria:**

- No Python exceptions
- Training completes for all specified epochs
- Predictions are generated

### Tier 4: Full Validation (~1 hour)

Only run after Tiers 1-3 pass:

```bash
# Run full training with production config
python -m admet.model.chemprop.ensemble \
    --config configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml
```

**Pass criteria:**

- Training completes successfully
- Prediction accuracy is comparable to pre-merge baseline
- No warnings about reproducibility issues

---

## Phase 5: Final Merge to Main

### Step 5.1: Commit the Merge (if not already done)

```bash
# If merge was done with --no-commit
git commit -m "Merge backup-main-20251224 into fix/joint_sampling

Combines:
- JointSampler two-stage sampling fix (critical bug fix)
- MLOps infrastructure improvements from backup
- New model types (Chemeleon, Classical ML)
- Enhanced HPO framework
- Comprehensive test additions

Tested with: pytest (all pass), smoke test (pass)"
```

### Step 5.2: Final Test Run

```bash
# One more test run to be safe
pytest tests/ -x -q
```

### Step 5.3: Merge to Main

```bash
# Switch to main
git checkout main

# Merge the combined branch
git merge fix/joint_sampling --no-ff -m "Merge fix/joint_sampling with consolidated features

This merge includes:
- Critical JointSampler bug fix for two-stage sampling
- Features from backup-main-20251224:
  - Multi-model support infrastructure
  - Classical ML models (XGBoost, LightGBM, CatBoost)
  - Chemeleon model support
  - Enhanced MLflow integration
  - Comprehensive test coverage additions"

# Verify merge
git log --oneline -5
```

### Step 5.4: Push to Remote

```bash
# Push main
git push origin main

# Optional: Push the fix branch for reference
git push origin fix/joint_sampling
```

---

## Rollback Procedures

### If Merge Goes Wrong (Before Commit)

```bash
# Abort the merge entirely
git merge --abort

# Verify you're back to original state
git status
git log --oneline -3
```

### If Merge Goes Wrong (After Commit, Before Push)

```bash
# Reset to pre-merge state
git reset --hard <pre-merge-commit-hash>

# Or use the backup branch
git reset --hard backup-fix-joint-sampling-<timestamp>
```

### If Already Pushed to Remote

```bash
# Create a revert commit (safer than force push)
git revert -m 1 HEAD

# Or if you must force push (DANGER: affects others)
git push origin main --force-with-lease
```

### Full Recovery from Backup

```bash
# List backup branches
git branch | grep backup

# Checkout the backup
git checkout backup-fix-joint-sampling-<timestamp>

# Create new branch from backup
git checkout -b fix/joint_sampling-recovered
```

---

## Appendix: File Change Summary

### Files Modified in Both Branches (Potential Conflicts)

| File | fix/joint_sampling | backup-main-20251224 | Risk |
|------|-------------------|---------------------|------|
| `chemprop/__init__.py` | +44 lines | Changes | Low |
| `chemprop/config.py` | +162 lines | Changes | Medium |
| `chemprop/curriculum.py` | Minor | +significant | Medium |
| `chemprop/ensemble.py` | +87 lines | Changes | High |
| `chemprop/model.py` | +140 lines | +significant | High |
| `chemprop/joint_sampler.py` | +340 (NEW) | Different impl | Critical |

### Files Only in `backup-main-20251224` (Will Be Added)

```
src/admet/model/base.py                    (148 lines)
src/admet/model/config.py                  (379 lines)
src/admet/model/ensemble.py                (322 lines)
src/admet/model/mlflow_mixin.py            (283 lines)
src/admet/model/registry.py                (147 lines)
src/admet/model/chemeleon/__init__.py
src/admet/model/chemeleon/callbacks.py
src/admet/model/chemeleon/model.py
src/admet/model/chemprop/adapter.py        (322 lines)
src/admet/model/classical/__init__.py
src/admet/model/classical/base.py
src/admet/model/classical/catboost_model.py
src/admet/model/classical/lightgbm_model.py
src/admet/model/classical/xgboost_model.py
src/admet/model/hpo/__init__.py            (272 lines)
src/admet/model/hpo/search_space.py        (235 lines)
```

### Test Files Added by Backup

```
tests/test_chemeleon_model.py              (274 lines)
tests/test_chemprop_adapter.py             (159 lines)
tests/test_classical_models.py             (312 lines)
tests/test_determinism.py                  (621 lines)
tests/test_ensemble_generic.py             (175 lines)
tests/test_fingerprints.py                 (338 lines)
tests/test_hpo_search_space_generic.py     (213 lines)
tests/test_joint_sampler.py                (385 lines)
tests/test_model_base.py                   (536 lines)
```

---

## Quick Reference Commands

```bash
# === PHASE 1: PREP ===
git status && git branch --show-current
pytest tests/ -x -q --tb=short
git branch backup-fix-joint-sampling-$(date +%Y%m%d-%H%M%S)

# === PHASE 2: MERGE ===
git merge backup-main-20251224 --no-commit --no-ff
git diff --name-only --diff-filter=U  # See conflicts

# === PHASE 3: RESOLVE ===
git checkout --ours src/admet/model/chemprop/joint_sampler.py
git add src/admet/model/chemprop/joint_sampler.py
# ... resolve other conflicts ...

# === PHASE 4: VALIDATE ===
python -c "from admet.model.chemprop import ensemble, model"
pytest tests/test_joint_sampler.py tests/test_curriculum*.py -v
pytest tests/ -x --tb=short

# === PHASE 5: FINALIZE ===
git commit -m "Merge backup-main-20251224 into fix/joint_sampling"
git checkout main && git merge fix/joint_sampling --no-ff
git push origin main

# === EMERGENCY ===
git merge --abort           # Cancel merge
git reset --hard HEAD~1     # Undo last commit
```

---

**End of Merge Plan**
