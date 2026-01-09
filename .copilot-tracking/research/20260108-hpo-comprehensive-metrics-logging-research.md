<!-- markdownlint-disable-file -->

# Task Research Notes: HPO Comprehensive Metrics Logging Enhancement

## Research Executed

### File Analysis

- [src/admet/model/chemprop/hpo_trainable.py](src/admet/model/chemprop/hpo_trainable.py)
  - Contains `RayTuneReportCallback` that reports metrics to Ray Tune during validation
  - `METRICS_TO_REPORT` tuple defines: `val_mae`, `val_loss`, `val_rmse`, `val_R2`, `val_pearson_r`, `val_spearman_rho`, `val_kendall_tau`, `train_loss`, `train_mae`, `lr`
  - **Missing test set evaluation** - only reports train/val metrics during training
  - Checkpoints built at trial end but no final test evaluation

- [src/admet/model/chemeleon/hpo.py](src/admet/model/chemeleon/hpo.py)
  - Contains `ChemeleonRayTuneCallback` with identical `METRICS_TO_REPORT` tuple
  - Same pattern as chemprop - **no test set evaluation**
  - `train_chemeleon_trial()` function trains but doesn't compute final metrics on all splits

- [src/admet/model/chemprop/ensemble.py](src/admet/model/chemprop/ensemble.py)
  - **Reference implementation for comprehensive metrics logging**
  - Logs metrics with naming convention: `{split_name}/{safe_target}_{safe_metric}`
  - Example: `test/log_ksol_mae`, `test/log_ksol_r2`, `test/mean_mae`
  - Uses `_sanitize_metric_label()` for safe MLflow metric names
  - Computes: mae, rae, mape, rmse, R2, pearson_r, spearman_rho, kendall_tau

- [src/admet/plot/metrics.py](src/admet/plot/metrics.py)
  - `METRIC_NAMES`: `("mae", "rae", "mape", "rmse", "R2", "pearson_r", "spearman_rho", "kendall_tau")`
  - `compute_metrics_df()` - computes all correlation metrics for each endpoint

- [src/admet/data/stats.py](src/admet/data/stats.py)
  - `correlation()` function returns `CorrelationMetrics` TypedDict
  - All 8 metrics: mae, rae, mape, rmse, R2, pearson_r, spearman_rho, kendall_tau

- [src/admet/model/chemprop/model.py](src/admet/model/chemprop/model.py)
  - `predict()` method with `split_name` and `log_metrics` parameters
  - `_generate_evaluation_plots()` creates parity plots and metric bar charts
  - `_log_metrics_with_retry()` handles MLflow metric logging

### Code Search Results

- Metrics naming pattern in ensemble:
  - Per-target: `{split_name}/{safe_target}_{safe_metric}`
  - Mean metrics: `{split_name}/mean_{safe_metric}`
  - Standard error: `{split_name}/{safe_target}_{safe_metric}_stderr`

- HPO currently reports (train + val only):
  - `val_mae`, `val_loss`, `val_rmse`, `val_R2`, `val_pearson_r`, `val_spearman_rho`, `val_kendall_tau`
  - `train_loss`, `train_mae`
  - `lr`, `epoch`, `early_stopped`, `model_params_millions`

### Project Conventions

- Standards referenced: Ensemble metric logging in `ensemble.py`
- Naming convention: `{split}/{target}_{metric}` pattern
- Target name sanitization: Replace spaces with `_`, `>` with `gt`, `<` with `lt`, `-` with `_`

## Key Discoveries

### Current HPO Metrics Gap

1. **No test set evaluation**: HPO trials only compute train/val metrics during training
2. **Missing train set comprehensive metrics**: Only `train_loss` and `train_mae` reported
3. **Missing RAE metric**: Important for challenge leaderboard ranking
4. **No per-target breakdown**: Only aggregate metrics reported

### Ensemble Metrics Reference (Target Format)

The ensemble implementation logs metrics with this structure:

```python
# Per-target metrics
f"{split_name}/{safe_target}_{safe_metric}"  # e.g., "test/log_ksol_mae"
f"{split_name}/{safe_target}_{safe_metric}_stderr"  # e.g., "test/log_ksol_mae_stderr"

# Aggregate metrics
f"{split_name}/mean_{safe_metric}"  # e.g., "test/mean_mae"
f"{split_name}/mean_{safe_metric}_stderr"  # e.g., "test/mean_mae_stderr"
```

### Required Metrics (All 8 from METRIC_NAMES)

- `mae` - Mean Absolute Error
- `rae` - Relative Absolute Error (competition metric)
- `mape` - Mean Absolute Percentage Error
- `rmse` - Root Mean Squared Error
- `R2` - R-squared coefficient
- `pearson_r` - Pearson correlation
- `spearman_rho` - Spearman rank correlation
- `kendall_tau` - Kendall tau correlation

### When to Compute Comprehensive Metrics

Based on user clarification needs:

**Option A: At Trial End (Preferred)**
- Compute when training completes (either by max_epochs or early stopping)
- Log to both Ray Tune (for ASHA analysis) and MLflow (for experiment tracking)
- Evaluate on train, validation, and test sets

**Option B: On Pause/Checkpoint**
- ASHA scheduler pauses poorly performing trials
- Could compute metrics at checkpoint time
- Higher overhead, may not be necessary

### Implementation Points

1. **Chemprop HPO** ([hpo_trainable.py](src/admet/model/chemprop/hpo_trainable.py))
   - Modify `train_chemprop_trial()` to compute final metrics after `model.fit()`
   - Use `model.predict()` with `split_name="train"`, `split_name="validation"`, `split_name="test"`
   - Log via `ray.tune.report()` with proper naming

2. **Chemeleon HPO** ([chemeleon/hpo.py](src/admet/model/chemeleon/hpo.py))
   - Mirror chemprop changes in `train_chemeleon_trial()`
   - Use consistent naming convention

3. **Metric Computation** (existing utilities)
   - Use `compute_metrics_df()` from `admet.plot.metrics`
   - Use `correlation()` from `admet.data.stats`

### Complete Examples

```python
# Example metric logging format (matching ensemble convention)
from admet.plot.metrics import compute_metrics_df, METRIC_NAMES

def _compute_and_log_final_metrics(model, df, split_name: str) -> dict[str, float]:
    """Compute and return all metrics for a dataset split."""
    pred_df = model.predict(df, generate_plots=False, split_name=split_name)
    metrics_df = compute_metrics_df(df, pred_df, model.target_cols)

    final_metrics = {}
    for target in model.target_cols:
        safe_target = _sanitize_target_name(target)
        for metric_name in METRIC_NAMES:
            if metric_name in metrics_df.columns:
                value = metrics_df.loc[target, metric_name]
                key = f"{split_name}/{safe_target}_{metric_name}"
                final_metrics[key] = float(value)

    # Add mean metrics across all targets
    for metric_name in METRIC_NAMES:
        if metric_name in metrics_df.columns:
            mean_val = metrics_df[metric_name].mean()
            final_metrics[f"{split_name}/mean_{metric_name}"] = float(mean_val)

    return final_metrics

def _sanitize_target_name(target: str) -> str:
    """Convert target name to safe MLflow metric name."""
    return (target.lower()
            .replace(" ", "_")
            .replace(">", "gt")
            .replace("<", "lt")
            .replace("-", "_")
            .replace("log_", ""))  # Remove redundant log_ prefix
```

### Configuration Examples

HPO configs may need a new option to control test set evaluation:

```yaml
# configs/1-hpo-single-fold/phases/phase1_explore_chemprop.yaml
hpo:
  # Existing config...

  # New: Final metrics evaluation
  final_metrics:
    enabled: true  # Compute comprehensive metrics at trial end
    splits: ["train", "validation", "test"]  # Which splits to evaluate
    log_per_target: true  # Log per-target metrics
    log_mean: true  # Log mean across targets
```

### Technical Requirements

1. **Data availability**: Test data path must be passed to HPO trial
2. **Model state**: Model must be trained before evaluation
3. **Timing**: Compute after training completes, before checkpoint upload
4. **Performance**: Add minimal overhead (single forward pass per split)

## Recommended Approach

Implement comprehensive final metrics computation in both chemprop and chemeleon HPO trainable functions:

1. **Location**: At the end of `train_chemprop_trial()` and `train_chemeleon_trial()` after `model.fit()` completes
2. **Metrics**: All 8 metrics from `METRIC_NAMES`
3. **Splits**: train, validation, test (when test data available)
4. **Naming**: Follow ensemble convention `{split_name}/{safe_target}_{metric_name}`
5. **Logging**: Report via `ray.tune.report()` for Ray Tune and MLflow

This approach:
- Adds minimal code changes
- Reuses existing `compute_metrics_df()` utility
- Maintains consistency with ensemble metrics naming
- Enables better HPO analysis via Ray Tune results

## Implementation Guidance

- **Objectives**: Enable comprehensive metrics logging for all HPO trials on train/val/test sets
- **Key Tasks**:
  1. Add final metrics computation function (shared utility)
  2. Modify `train_chemprop_trial()` to compute final metrics
  3. Modify `train_chemeleon_trial()` to compute final metrics
  4. Update HPO configs to pass test data path if needed
- **Dependencies**:
  - `admet.plot.metrics.compute_metrics_df`
  - `admet.data.stats.correlation`
  - Test data must be accessible to HPO trials
- **Success Criteria**:
  - All 8 metrics logged for train/val/test splits
  - Per-target and mean metrics available in MLflow/Ray Tune
  - Naming matches ensemble convention
