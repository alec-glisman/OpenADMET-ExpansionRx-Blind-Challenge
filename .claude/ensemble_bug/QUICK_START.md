# Quick Start: Fix Ensemble Regression Bug

## Immediate Actions (30 minutes)

### 1. Backup Everything
```bash
cd /path/to/your/repo

# Create safety backups
git branch backup-main-$(date +%Y%m%d) main
git tag backup-pre-fix main
git push origin backup-main-$(date +%Y%m%d) backup-pre-fix

# Tag the known states
git tag last-known-good 0d41199f07d930062b943681357a7029554961f6
git tag first-known-bad 1f74f8396f7be56ad573256d4769ed0e96d9d69b
git push origin last-known-good first-known-bad

echo "✓ Backups complete"
```

### 2. Identify the Bug

Run the investigation script:

```bash
# Make script executable
chmod +x analyze_commit_diff.sh

# Run analysis
bash analyze_commit_diff.sh

# Review output
cat /tmp/ensemble_bug_analysis/ensemble_diff.patch
```

**Look for changes in**:
- `_aggregate_predictions` method
- `_save_ensemble_predictions` method
- Array operations with `np.mean`, `np.power`
- Column naming (`pred_col = target`)
- Log transformation order

### 3. Preserve New Work

```bash
# Create branch with all new commits
git checkout -b temp-new-features main

# Reset main to good commit
git checkout main
git reset --hard 0d41199f

# Force push (your backups are safe!)
git push --force-with-lease origin main

echo "✓ Main is now at last-known-good commit"
echo "✓ New work preserved in temp-new-features branch"
```

---

## Fix the Bug (1-2 hours + test time)

### 4. Create Fix Branch

```bash
# Start from good commit
git checkout -b fix/ensemble-regression 0d41199f

# Apply the bad commit to see what it changed
git cherry-pick 1f74f83

# If this breaks performance, examine the changes:
git diff HEAD~1 HEAD
```

### 5. Most Likely Bug Locations

Based on code analysis, check these in `src/admet/model/chemprop/ensemble.py`:

**Hypothesis 1: Wrong column name**
```python
# Around line 710-720
for target in target_cols:
    pred_col = target  # ← Should this be f"{target}_pred" instead?
    preds = np.array([df[pred_col].values for df in predictions_list])
```

**Fix**: Print `predictions_list[0].columns` to verify correct column names.

**Hypothesis 2: Transform order**
```python
# Around line 745-755
if target.startswith("Log "):
    result[f"{target}_transformed_mean"] = np.power(10, mean_pred)
```

**Fix**: Ensure this is `10^(mean(log_values))` not `mean(10^log_values)`.

**Hypothesis 3: Missing predictions**
```python
# Around line 700
def _aggregate_predictions(self, predictions_list: List[pd.DataFrame], ...):
    # Add debug print
    print(f"Aggregating {len(predictions_list)} models")  # Should be 25
```

### 6. Quick Test

```python
# Create test script: test_aggregation.py
import numpy as np
import pandas as pd

# Simulate 3 models predicting Log KSOL
pred1 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-2.0]})
pred2 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-2.1]})
pred3 = pd.DataFrame({'SMILES': ['CCO'], 'Log KSOL': [-1.9]})

# Expected behavior:
log_mean = np.mean([-2.0, -2.1, -1.9])  # = -2.0
transformed = np.power(10, log_mean)     # = 0.01

print(f"Log mean: {log_mean}")
print(f"Transformed: {transformed}")

# Now test your actual aggregation function
```

### 7. Run Full Test

```bash
# After making your fix
python -m admet.model.chemprop.ensemble \
    --config configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml

# Check MLflow for R² metric - should be ~0.52
```

---

## Re-integrate New Features (2-4 hours)

### 8. Merge Fix Back

```bash
# Once fix is validated
git checkout main
git merge fix/ensemble-regression

# Test
pytest tests/test_ensemble_chemprop.py -v
```

### 9. Rebase New Features

```bash
git checkout temp-new-features
git rebase main

# Or cherry-pick selectively:
git checkout main
git log temp-new-features --oneline > /tmp/commits.txt
# Review and cherry-pick each one
```

### 10. Final Validation

```bash
# Run full test suite
pytest tests/ -v

# Run ensemble one more time
python -m admet.model.chemprop.ensemble \
    --config configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml

# Verify R² ≥ 0.52 in MLflow
```

---

## Emergency Rollback

If anything goes wrong:

```bash
# Restore from backup
git checkout main
git reset --hard backup-pre-fix
git push --force-with-lease origin main

# Your backup tags and branches are still there!
```

---

## Verification Checklist

Before declaring victory:

- [ ] R² metric returns to ≥ 0.50 (target: ~0.52)
- [ ] MAE returns to previous performance
- [ ] All unit tests pass: `pytest tests/`
- [ ] MLflow run shows comparable metrics to good run (e05f94eb...)
- [ ] All 15 new commits successfully re-integrated
- [ ] No merge conflicts remaining
- [ ] Documented the exact bug and fix in commit message

---

## What to Look For in the Diff

When reviewing `/tmp/ensemble_bug_analysis/ensemble_diff.patch`:

### Red Flags 🚩

1. **Column naming changes**:
   ```diff
   - preds = np.array([df[f"{target}_pred"].values for df in predictions_list])
   + preds = np.array([df[target].values for df in predictions_list])
   ```

2. **Transformation order changes**:
   ```diff
   - transformed = np.power(10, np.mean(preds, axis=0))
   + transformed = np.mean(np.power(10, preds), axis=0)
   ```

3. **Axis changes**:
   ```diff
   - mean_pred = np.mean(preds, axis=0)  # Average across models
   + mean_pred = np.mean(preds, axis=1)  # Average across samples (WRONG!)
   ```

4. **Array stacking**:
   ```diff
   - preds = np.stack([df[target].values for df in predictions_list], axis=0)
   + preds = np.array([df[target].values for df in predictions_list])
   ```

### Green Flags ✅

1. Documentation/comment changes only
2. Whitespace or formatting changes
3. Added logging statements
4. Import reordering

---

## Tips for Success

1. **Don't skip the backup step** - It takes 30 seconds and could save hours
2. **Read the diff carefully** - The bug is probably 1-5 lines of code
3. **Test incrementally** - Don't wait for the full 1-hour ensemble run
4. **Use MLflow** - Compare metrics between runs precisely
5. **Document everything** - Future you will thank current you

---

## Need Help?

If stuck, check:

1. **Full analysis files**: `/tmp/ensemble_bug_analysis/`
2. **Detailed plan**: `REGRESSION_BUG_FIX_PLAN.md`
3. **Investigation script**: `investigate_ensemble_bug.py`

---

## Time Estimates

- **Investigation**: 30 min - 1 hour
- **Fix implementation**: 30 min - 1 hour
- **Testing (full ensemble)**: 1 hour (automated)
- **Re-integration**: 2-4 hours
- **Total**: ~6-8 hours (including wait times)

---

*Good luck! You've got this! 🚀*
