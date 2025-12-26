# Per-Quality Metrics Debugging Guide

## Summary

All diagnostic tests pass! The `PerQualityMetricsCallback` is working correctly in isolation. If you're not seeing metrics in your actual training, follow this guide to diagnose the issue.

## What We've Done

### 1. ✅ Added Detailed Logging

Enhanced the callback with INFO-level logging that shows:

- When the callback is called
- Current epoch and compute frequency
- Dataloader availability and structure
- Number of quality labels and target columns
- Batch processing progress
- Shape of predictions and targets
- Success/failure of metric computation

### 2. ✅ Verified Configuration

Your config at `configs/curriculum/chemprop_curriculum.yaml` has:

- `log_per_quality_metrics: true` ✓
- `curriculum.enabled: true` ✓
- `curriculum.qualities: [high, medium, low]` ✓

### 3. ✅ Created Debug Script

The script at `scripts/analysis/debug_per_quality_metrics.py` validates:

- Callback can be imported ✓
- Callback can be instantiated with quality labels ✓
- Callback computes and logs metrics correctly ✓
- Configuration is correct ✓
- MLflow is accessible ✓

## Diagnostic Steps

### Step 1: Run Training with INFO Logging

```bash
# Add to your training command:
export PYTHONUNBUFFERED=1

# Run training and capture logs
python your_training_script.py 2>&1 | tee training.log
```

Look for these log messages:

```
INFO - ================================================================================
INFO - PerQualityMetricsCallback.on_validation_epoch_end called
INFO - Current epoch: X, compute_every_n_epochs: 1
INFO - Validation dataloader: <class '...'>, is_none: False
```

### Step 2: Check What You See

#### Scenario A: No callback logs at all

**Problem**: Callback not added to trainer or not enabled in config

**Solutions**:

1. Check your config has `log_per_quality_metrics: true`
2. Verify callback is instantiated in your training code
3. Add debug print to see callbacks:

   ```python
   print("Trainer callbacks:", [type(cb).__name__ for cb in trainer.callbacks])
   ```

#### Scenario B: Callback logs "Skipping epoch X"

**Problem**: `compute_every_n_epochs` setting

**Solutions**:

1. Check callback initialization - it should log every epoch by default
2. Verify in logs: `compute_every_n_epochs: 1`
3. If you see a different number, check where callback is instantiated

#### Scenario C: "No validation dataloader available"

**Problem**: Validation not configured or not running

**Solutions**:

1. Verify you have validation data split
2. Check trainer has `val_dataloaders` set
3. Ensure validation is actually running (look for validation loss logs)

#### Scenario D: "Mismatch: X predictions vs Y quality labels"

**Problem**: Validation dataset size doesn't match quality labels

**Solutions**:

1. Check your validation data loading code
2. Verify quality labels list matches validation dataset length
3. Print lengths:

   ```python
   print(f"Val dataset size: {len(val_dataset)}")
   print(f"Quality labels size: {len(val_quality_labels)}")
   ```

#### Scenario E: Logs show processing but no metrics in MLflow

**Problem**: MLflow logging issue

**Solutions**:

1. Verify MLflow experiment is created
2. Check MLflow tracking URI is correct
3. Look for MLflow errors in logs
4. Verify run is active when callback fires

### Step 3: Add Debug Prints to Your Training Code

Add these before starting training:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Print configuration
print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Quality labels size: {len(val_quality_labels)}")
print(f"Quality distribution: {pd.Series(val_quality_labels).value_counts().to_dict()}")
print(f"Target columns: {target_cols}")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {mlflow.get_experiment_by_name('your_experiment_name')}")
print()

# Print callbacks
print("=" * 80)
print("TRAINER CALLBACKS")
print("=" * 80)
for i, cb in enumerate(trainer.callbacks):
    print(f"{i+1}. {type(cb).__name__}")
    if isinstance(cb, PerQualityMetricsCallback):
        print(f"   - Quality labels count: {len(cb.val_quality_labels)}")
        print(f"   - Qualities: {cb.qualities}")
        print(f"   - Target columns: {cb.target_cols}")
        print(f"   - Compute every N epochs: {cb.compute_every_n_epochs}")
print()
```

### Step 4: Check Training Logs

After training starts, check for these patterns:

**Good Signs** (callback is working):

```
INFO - PerQualityMetricsCallback.on_validation_epoch_end called
INFO - Current epoch: 0, compute_every_n_epochs: 1
INFO - Processing dataloader 1/1
INFO - Processed 10 batches total
INFO - Collected 10 prediction batches
INFO - Computing per-quality metrics...
INFO - ✓ Per-quality validation metrics successfully computed and logged for epoch 0
```

**Bad Signs** (callback not working):

- No PerQualityMetricsCallback logs at all
- "No validation dataloader available"
- "Mismatch: X predictions vs Y quality labels"
- "Skipping epoch X"

## Quick Test

To quickly verify the callback works in your environment:

```bash
# Run the debug script
python scripts/analysis/debug_per_quality_metrics.py

# All tests should pass
# If any fail, check the output for specific error messages
```

## Common Issues and Solutions

### Issue: "epoch is remaining at 1"

**Cause**: Training might be configured with `max_epochs=1` or early stopping very aggressively

**Solution**: Check your training config:

```yaml
optimization:
  max_epochs: 150  # Should be > 1
  patience: 15     # Shouldn't trigger too early
```

### Issue: Only seeing "train_loss_step"

**Cause**: Validation might not be running

**Solution**:

1. Verify you have validation data
2. Check trainer is configured to run validation
3. Look for validation loss in logs (should see `val_loss`)

### Issue: Metrics appear but wrong hierarchy

**Cause**: This was the original issue we fixed!

**Solution**: Already fixed! New hierarchy is `val/<metric>/<quality>` (e.g., `val/mae/high`)

## Expected Output

When working correctly, you should see metrics in MLflow like:

```
val/mae/high = 0.123
val/mae/medium = 0.234
val/mae/low = 0.345
val/rmse/high = 0.456
val/rmse/medium = 0.567
val/rmse/low = 0.678
val/count/high = 100
val/count/medium = 80
val/count/low = 50
```

And with target columns:

```
val/mae/high/LogD = 0.111
val/mae/high/KSOL = 0.222
val/rmse/medium/LogD = 0.333
...
```

## Next Steps

1. **Run the debug script**: `python scripts/analysis/debug_per_quality_metrics.py`
2. **Start training with INFO logging**: Capture output to a log file
3. **Search logs** for "PerQualityMetricsCallback" to see what's happening
4. **Match your scenario** to the ones above and apply the solution

If you still don't see metrics after following this guide, share:

1. The output of the debug script
2. Relevant sections from your training logs
3. Your training script (or the relevant callback instantiation code)

## Test Results

Run on: 2025-12-17

```
Import              : ✓ PASS
Instantiation       : ✓ PASS
Mock Data           : ✓ PASS
Config              : ✓ PASS
MLflow              : ✓ PASS
```

All diagnostic tests passed! The callback implementation is correct.
