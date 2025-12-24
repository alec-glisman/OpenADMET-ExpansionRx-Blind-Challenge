# Regression Bug Fix Plan - ADMET Ensemble Predictions
## Updated: Based on MLflow Data & Single-Commit Regression

---

## Executive Summary

**Problem**: Ensemble prediction performance degraded between commits:
- **Good commit**: `0d41199f07d930062b943681357a7029554961f6` (R² ≈ 0.52)
- **Bad commit**: `1f74f8396f7be56ad573256d4769ed0e96d9d69b` (R² ≈ 0.48)
- **Impact**: ~7.7% decrease in R² (0.52 → 0.48), with similar degradations in MAE, RAE
- **Scope**: Single commit introduced the bug
- **Training time**: ~1 hour per full ensemble run

**MLflow Evidence**:
- Good run ID: `e05f94eb818943e496579d7e36459a47`
- Bad run ID: `e26ac60be8b948bab4482c7589e856bb`
- Experiment ID: `7`

---

## Part 1: Simplified Git Strategy

Since only **one commit** separates good from bad, we don't need complex bisection.

### Step 1.1: Backup and Tag Current State

```bash
# Create backup
git branch backup-main-$(date +%Y%m%d) main
git tag backup-pre-fix main
git push origin backup-main-$(date +%Y%m%d)
git push origin backup-pre-fix

# Tag the known states
git tag last-known-good 0d41199f07d930062b943681357a7029554961f6
git tag first-known-bad 1f74f8396f7be56ad573256d4769ed0e96d9d69b
git push origin last-known-good first-known-bad
```

### Step 1.2: Identify the Problematic Commit

```bash
# View the exact commit that caused the problem
git log --oneline 0d41199f..1f74f83

# Get detailed diff of the problematic commit
git show 1f74f83 > /tmp/problematic_commit.diff

# See which files changed
git diff --name-only 0d41199f 1f74f83

# Get full diff
git diff 0d41199f 1f74f83 > /tmp/full_changes.diff
```

### Step 1.3: Branch Strategy for ~15 New Commits

Since you have ~15 commits after the bad one with new features:

```bash
# Count commits after the bad commit
git rev-list --count 1f74f83..HEAD

# Create a temporary preservation branch with all new work
git checkout -b temp-new-features HEAD

# Reset main to the good commit
git checkout main
git reset --hard 0d41199f

# Force push with safety
git push --force-with-lease origin main

# Push preservation branch
git push -u origin temp-new-features
```

**Recommended Branch Structure** (1 branch for now):
- `temp-new-features` - All 15 commits preserved together
- Later, if needed, can split into themed branches

---

## Part 2: Bug Identification Strategy

### Prime Suspect: `_aggregate_predictions` Method

Based on code review of `src/admet/model/chemprop/ensemble.py`, the most likely culprit is in the prediction aggregation logic around **Log transformation**.

**Critical Code Section** (lines 699-756):
```python
def _aggregate_predictions(self, predictions_list: List[pd.DataFrame], split_name: str) -> pd.DataFrame:
    """
    Aggregate predictions from multiple models.

    Computes mean and standard error for each target column.
    For Log columns, applies 10^x transform after averaging.
    """
    # ...
    for target in target_cols:
        # Collect predictions from all models
        pred_col = target  # Predictions are in target column
        preds = np.array([df[pred_col].values for df in predictions_list])

        # Calculate statistics
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0, ddof=1)
        stderr_pred = std_pred / np.sqrt(n_models)

        result[f"{target}_mean"] = mean_pred
        result[f"{target}_std"] = std_pred
        result[f"{target}_stderr"] = stderr_pred

        # For "Log " columns (with space), compute transformed values
        # Average first, then transform: 10^mean(x_i)
        if target.startswith("Log "):
            result[f"{target}_transformed_mean"] = np.power(10, mean_pred)
            # Propagate uncertainty
            result[f"{target}_transformed_stderr"] = np.log(10) * np.power(10, mean_pred) * stderr_pred
```

**Potential Issues to Check**:

1. **Transformation Order Bug**: 
   - CORRECT: `mean(log_values)` then `10^mean` → geometric mean
   - WRONG: `mean(10^log_values)` → arithmetic mean of non-log values
   - This would explain consistent performance degradation across metrics

2. **Column Naming Change**:
   - Check if `pred_col = target` is correct
   - Might need `f"{target}_pred"` or different naming

3. **Array Stacking Issue**:
   - `np.array([df[pred_col].values for df in predictions_list])`
   - Could be stacking in wrong axis

4. **Submission Format Bug**:
   - In `_save_ensemble_predictions`, check if submissions use correct columns:
   ```python
   if target.startswith("Log "):
       clean_name = target.replace("Log ", "")
       submissions[clean_name] = predictions[f"{target}_transformed_mean"]
   ```

### Step 2.1: Quick Diagnostic Commands

```bash
# Checkout the bad commit
git checkout 1f74f83

# View the specific changes in ensemble.py
git show 1f74f83 -- src/admet/model/chemprop/ensemble.py

# Compare ensemble.py between good and bad
git diff 0d41199f 1f74f83 -- src/admet/model/chemprop/ensemble.py > /tmp/ensemble_diff.txt

# Check other related files
git diff 0d41199f 1f74f83 -- src/admet/model/chemprop/model.py
git diff 0d41199f 1f74f83 -- src/admet/model/base.py
git diff 0d41199f 1f74f83 -- src/admet/data/
```

### Step 2.2: Detailed File-by-File Analysis

```bash
# Get list of all changed files
git diff --name-only 0d41199f 1f74f83 > /tmp/changed_files.txt

# For each file, check if it could affect predictions
while read file; do
    echo "=== $file ==="
    git diff 0d41199f 1f74f83 -- "$file" | head -50
done < /tmp/changed_files.txt
```

### Step 2.3: MLflow Comparison Script

Create a script to compare the two runs programmatically:

```python
#!/usr/bin/env python3
"""
Compare MLflow runs to identify parameter/metric differences.
Run from repository root with MLflow available.
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Set tracking URI (update if different)
mlflow.set_tracking_uri("http://127.0.0.1:8084")
client = MlflowClient()

good_run_id = "e05f94eb818943e496579d7e36459a47"
bad_run_id = "e26ac60be8b948bab4482c7589e856bb"

def compare_runs(run_id_1, run_id_2, label_1="Good", label_2="Bad"):
    run1 = client.get_run(run_id_1)
    run2 = client.get_run(run_id_2)
    
    print(f"\n{'='*80}")
    print(f"COMPARING: {label_1} vs {label_2}")
    print(f"{'='*80}\n")
    
    # Compare parameters
    params1 = run1.data.params
    params2 = run2.data.params
    
    print("PARAMETER DIFFERENCES:")
    all_params = set(params1.keys()) | set(params2.keys())
    for param in sorted(all_params):
        val1 = params1.get(param, "N/A")
        val2 = params2.get(param, "N/A")
        if val1 != val2:
            print(f"  {param}:")
            print(f"    {label_1}: {val1}")
            print(f"    {label_2}: {val2}")
    
    # Compare metrics
    metrics1 = run1.data.metrics
    metrics2 = run2.data.metrics
    
    print("\n\nKEY METRIC COMPARISONS:")
    key_metrics = [k for k in metrics1.keys() if any(x in k for x in ['r2', 'mae', 'rmse', 'mean'])]
    
    metric_df = []
    for metric in sorted(key_metrics):
        if metric in metrics1 and metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
            metric_df.append({
                'metric': metric,
                'good': val1,
                'bad': val2,
                'diff': diff,
                'pct_change': pct_change
            })
    
    df = pd.DataFrame(metric_df)
    print(df.to_string(index=False))
    
    # Identify biggest degradations
    print("\n\nBIGGEST DEGRADATIONS:")
    worst = df.nsmallest(10, 'diff')
    print(worst[['metric', 'good', 'bad', 'diff', 'pct_change']].to_string(index=False))

if __name__ == "__main__":
    compare_runs(good_run_id, bad_run_id)
```

---

## Part 3: Bug Fix Process

### Step 3.1: Isolate and Fix

```bash
# Create a fix branch from good commit
git checkout -b fix/ensemble-regression 0d41199f

# Apply the bad commit to see what breaks
git cherry-pick 1f74f83

# If it applies cleanly, test it
python -m admet.model.chemprop.ensemble --config configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml

# If performance degrades, revert the commit and examine it
git revert HEAD --no-commit
git diff  # See what the revert changes

# Now manually fix the specific bug
# Edit src/admet/model/chemprop/ensemble.py
```

### Step 3.2: Hypothesis-Driven Fixes

**Hypothesis 1: Prediction Aggregation Order**

Check if predictions are being aggregated incorrectly:

```python
# In _aggregate_predictions, verify:
for target in target_cols:
    pred_col = target  # ← Is this correct? Or should it be f"{target}_pred"?
    preds = np.array([df[pred_col].values for df in predictions_list])
```

**Test**: Print column names from predictions_list[0] to verify structure.

**Hypothesis 2: Log Transform Direction**

```python
# Current implementation (correct):
mean_pred = np.mean(preds, axis=0)  # Mean in log space
transformed = np.power(10, mean_pred)  # Then transform

# Bug might be:
# transformed = np.mean(np.power(10, preds), axis=0)  # Transform then mean - WRONG!
```

**Hypothesis 3: Missing Predictions**

```python
# Check if all models are contributing
print(f"Number of models in ensemble: {len(predictions_list)}")
print(f"Expected: 25 (5 splits × 5 folds)")
```

### Step 3.3: Validation Testing

Create a minimal test to verify the fix:

```python
# tests/test_ensemble_regression_fix.py
import pytest
import numpy as np
import pandas as pd
from admet.model.chemprop.ensemble import ModelEnsemble

def test_log_transform_aggregation():
    """Ensure log-scale predictions aggregate correctly."""
    
    # Simulate 3 models predicting for 1 molecule on "Log KSOL"
    # In log space: [-1.0, -1.5, -1.2] → mean = -1.233
    # Transform: 10^(-1.233) = 0.0585
    
    pred1 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-1.0]})
    pred2 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-1.5]})
    pred3 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-1.2]})
    
    # Mock ensemble
    # ... aggregate predictions ...
    
    # Expected: mean in log space = -1.233
    # Expected: 10^(-1.233) ≈ 0.0585
    expected_log_mean = np.mean([-1.0, -1.5, -1.2])
    expected_transformed = np.power(10, expected_log_mean)
    
    # Assert aggregation is correct
    assert np.isclose(result['Log KSOL_mean'].values[0], expected_log_mean, rtol=1e-5)
    assert np.isclose(result['Log KSOL_transformed_mean'].values[0], expected_transformed, rtol=1e-5)
```

---

## Part 4: Re-integration of New Features

### Step 4.1: Once Bug is Fixed on Main

```bash
# Merge fix into main
git checkout main
git merge fix/ensemble-regression

# Test the fix
python -m admet.model.chemprop.ensemble --config configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml

# If R² returns to ~0.52, the fix is validated
```

### Step 4.2: Rebase New Features

```bash
# Rebase temp-new-features onto fixed main
git checkout temp-new-features
git rebase main

# Resolve any conflicts
# Test after each commit during rebase
```

### Step 4.3: Selective Cherry-Pick Approach (Alternative)

If rebase is complex:

```bash
# Cherry-pick commits one by one
git checkout main
git log temp-new-features --oneline | tac > /tmp/commits_to_apply.txt

# For each commit:
while read commit_hash commit_message; do
    echo "Applying: $commit_message"
    git cherry-pick $commit_hash
    
    # Test
    pytest tests/ -k "not mlflow"  # Quick smoke test
    
    # If failure, investigate
    if [ $? -ne 0 ]; then
        echo "FAILED at $commit_hash: $commit_message"
        git cherry-pick --abort
        break
    fi
done < /tmp/commits_to_apply.txt
```

### Step 4.4: Verification Strategy

After re-integrating each commit group:

```bash
# Quick validation (no full ensemble needed)
pytest tests/test_ensemble_chemprop.py -v

# Check that individual model training still works
python -m admet.model.chemprop.train --config configs/0-experiment/chemprop.yaml

# Only run full ensemble test before final merge
```

---

## Part 5: Prevention & Documentation

### Step 5.1: Add Regression Test

```python
# tests/test_ensemble_regression_guard.py
"""
Regression test to prevent future performance degradation.
This test compares ensemble aggregation logic against known-good behavior.
"""
import pytest
import numpy as np
import pandas as pd

def test_ensemble_aggregation_golden_values():
    """
    Golden test: Ensure aggregation produces expected values.
    
    This test locks in the correct behavior observed in commit 0d41199f.
    If this test fails, ensemble aggregation logic has regressed.
    """
    # Mock predictions from 3 models for 2 molecules on 2 endpoints
    predictions = [
        pd.DataFrame({
            'SMILES': ['CCO', 'CCN'],
            'LogD': [1.5, 2.0],
            'Log KSOL': [-2.0, -1.5]
        }),
        pd.DataFrame({
            'SMILES': ['CCO', 'CCN'],
            'LogD': [1.6, 1.9],
            'Log KSOL': [-2.1, -1.6]
        }),
        pd.DataFrame({
            'SMILES': ['CCO', 'CCN'],
            'LogD': [1.4, 2.1],
            'Log KSOL': [-1.9, -1.4]
        }),
    ]
    
    # Expected behavior:
    # LogD (no "Log " prefix with space): mean directly
    expected_logd_mean = np.array([
        np.mean([1.5, 1.6, 1.4]),  # CCO
        np.mean([2.0, 1.9, 2.1])   # CCN
    ])
    
    # Log KSOL (has "Log " prefix): mean in log space, then transform
    log_ksol_log_means = np.array([
        np.mean([-2.0, -2.1, -1.9]),  # CCO: -2.0
        np.mean([-1.5, -1.6, -1.4])   # CCN: -1.5
    ])
    expected_log_ksol_transformed = np.power(10, log_ksol_log_means)
    
    # Use actual ensemble aggregation
    # ensemble = ModelEnsemble.from_config(mock_config)
    # result = ensemble._aggregate_predictions(predictions, "test")
    
    # Assert golden values (adjust based on actual implementation)
    # np.testing.assert_allclose(result['LogD_mean'], expected_logd_mean, rtol=1e-6)
    # np.testing.assert_allclose(
    #     result['Log KSOL_transformed_mean'],
    #     expected_log_ksol_transformed,
    #     rtol=1e-6
    # )
```

### Step 5.2: Add Performance Benchmark

```python
# tests/test_ensemble_performance_benchmark.py
"""
Performance benchmark test.

NOTE: This test is expensive (~1 hour) and should only be run:
1. Before major releases
2. When ensemble code changes
3. In CI on a nightly schedule
"""
import pytest

@pytest.mark.slow
@pytest.mark.benchmark
def test_ensemble_performance_baseline():
    """
    Benchmark test: Ensure ensemble achieves minimum performance threshold.
    
    Based on commit 0d41199f performance:
    - Mean R² across endpoints: >= 0.50
    - Mean MAE: <= 0.65
    """
    # This would run the full ensemble
    # config = load_config("configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml")
    # ensemble = ModelEnsemble.from_config(config)
    # ensemble.train_all()
    
    # Extract metrics from MLflow
    # mean_r2 = get_mlflow_metric(ensemble.parent_run_id, "test/mean_r2")
    # mean_mae = get_mlflow_metric(ensemble.parent_run_id, "test/mean_mae")
    
    # assert mean_r2 >= 0.50, f"R² degraded: {mean_r2} < 0.50"
    # assert mean_mae <= 0.65, f"MAE degraded: {mean_mae} > 0.65"
```

### Step 5.3: Document in Commit Message

When committing the fix:

```bash
git commit -m "fix: restore ensemble prediction performance

PROBLEM:
- Commit 1f74f83 introduced regression in ensemble aggregation
- R² decreased from 0.52 → 0.48 (7.7% degradation)
- Affected all endpoints (MAE, RAE also degraded)

ROOT CAUSE:
- [Describe exact bug found, e.g.:]
- Bug in _aggregate_predictions: predictions were averaged before
  log transformation instead of after, resulting in geometric mean
  being replaced with arithmetic mean

FIX:
- [Describe exact change made]
- Corrected order of operations in _aggregate_predictions
- Now: mean(log_values) then transform(mean)
- Was: transform(values) then mean(transformed)

VERIFICATION:
- MLflow run [NEW_RUN_ID] shows R² = 0.52 (restored)
- Added regression test: test_ensemble_aggregation_golden_values
- Compared against good run: e05f94eb818943e496579d7e36459a47

Fixes #[issue_number]
Related to commit: 0d41199f (last known good)
Breaks from commit: 1f74f83 (introduced bug)
"
```

---

## Part 6: Execution Checklist

### Phase 1: Backup & Identify (30 minutes)
- [ ] Create backup branches and tags
- [ ] Get exact diff: `git diff 0d41199f 1f74f83`
- [ ] Identify changed files, especially in `src/admet/model/chemprop/ensemble.py`
- [ ] Run MLflow comparison script

### Phase 2: Isolate Bug (1-2 hours)
- [ ] Create fix branch from good commit
- [ ] Cherry-pick bad commit
- [ ] Reproduce degraded performance
- [ ] Identify exact line(s) causing regression
- [ ] Formulate hypothesis (prediction aggregation? log transform?)

### Phase 3: Fix & Validate (2-3 hours including test run)
- [ ] Implement fix
- [ ] Run unit tests: `pytest tests/test_ensemble_chemprop.py`
- [ ] Run full ensemble: `python -m admet.model.chemprop.ensemble --config ...`
- [ ] Verify R² returns to ~0.52
- [ ] Check MLflow metrics match good run

### Phase 4: Re-integrate Features (2-4 hours)
- [ ] Merge fix into main
- [ ] Rebase or cherry-pick 15 new commits
- [ ] Test after each commit
- [ ] Final validation with full ensemble

### Phase 5: Document & Prevent (1 hour)
- [ ] Write regression test
- [ ] Update documentation
- [ ] Create detailed commit message
- [ ] Update CHANGELOG

**Total Estimated Time**: 6-10 hours (excluding long ensemble runs)

---

## Part 7: Emergency Rollback Plan

If you need working code immediately:

```bash
# Quick rollback to last known good
git checkout main
git reset --hard 0d41199f
git push --force-with-lease origin main

# This loses the 15 new commits on main, but they're preserved in temp-new-features
```

To restore later:
```bash
git merge temp-new-features
# Or rebase/cherry-pick as described above
```

---

## Additional Resources

### Useful Commands Reference

```bash
# View specific file at specific commit
git show 0d41199f:src/admet/model/chemprop/ensemble.py > /tmp/good_ensemble.py
git show 1f74f83:src/admet/model/chemprop/ensemble.py > /tmp/bad_ensemble.py
diff -u /tmp/good_ensemble.py /tmp/bad_ensemble.py

# Find when a specific line changed
git log -L 720,760:src/admet/model/chemprop/ensemble.py

# Blame to see who changed what
git blame src/admet/model/chemprop/ensemble.py

# Search for specific text in commit messages
git log --all --grep="aggregat" --oneline
```

### Files to Focus On

Based on the ensemble module structure:

1. **Primary Suspect**: `src/admet/model/chemprop/ensemble.py`
   - Lines 699-756: `_aggregate_predictions`
   - Lines 758-788: `_save_ensemble_predictions`

2. **Secondary Suspects**:
   - `src/admet/model/chemprop/model.py` - Individual model predictions
   - `src/admet/model/base.py` - Base model interface
   - `src/admet/data/` - Data loading/preprocessing

3. **Config Files**: Check if config format changed
   - `configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml`

---

## Questions to Answer During Investigation

1. **What files changed in the bad commit?**
   ```bash
   git diff --name-only 0d41199f 1f74f83
   ```

2. **Did the prediction column naming change?**
   - Check if models output `"Log KSOL"` or `"Log KSOL_pred"`

3. **Did the aggregation logic change?**
   - Compare `_aggregate_predictions` between commits

4. **Did data preprocessing change?**
   - Check if log transforms are applied during data loading

5. **Did config parsing change?**
   - Verify `target_cols` parsing is identical

6. **Are all 25 models being used?**
   - Print `len(predictions_list)` during aggregation

---

## Expected Outcomes

**Success Criteria**:
- R² returns to ≥ 0.50 (ideally ~0.52)
- MAE returns to previous levels
- All 15 new feature commits successfully re-integrated
- Regression test added to prevent future issues
- Full documentation of root cause and fix

**Timeline**: 1-2 days (including multiple ensemble training runs for validation)

---

## Contact & Support

If you encounter issues during execution:
1. Save all error messages and git output
2. Document which step failed
3. Preserve all branches (don't force push without backups)
4. MLflow runs are permanent audit trail - use them!

---

*Last Updated: 2024-12-24*
*Based on project knowledge base and code analysis*
