<!-- markdownlint-disable-file -->

# Task Research Notes: MLflow Logging for Ensemble Bar Plot Metrics

## Research Executed

### File Analysis

- `src/admet/model/chemprop/ensemble.py`
  - Contains `_generate_metrics_bar_plot` method (lines 863-1056) that creates bar plots for ensemble metrics
  - Already logs individual target metrics and mean metrics to MLflow (lines 990-1032)
  - Generates separate bar plots for 7 metric types: MAE, RMSE, R², RAE, Spearman ρ, Pearson r, Kendall τ
  - Each bar plot shows metrics per target endpoint with error bars (stderr)
  - Saves plots as PNG files and logs to MLflow as artifacts

- `src/admet/plot/metrics.py`
  - Contains `plot_metric_bar` function (lines 143-250) used to generate bar plots
  - Returns tuple of (Figure, Axes) objects
  - Supports error bars via `errors` parameter
  - Can show mean bar with standard error via `show_mean` parameter

### Code Search Results

- **Current MLflow metric logging pattern**
  - `self._mlflow_client.log_metric(run_id, metric_name, float(value))`
  - Individual target metrics: `{split_name}/{target}_{metric}`
  - Mean metrics: `{split_name}/mean_{metric}`
  - Standard error metrics: `{split_name}/{target}_{metric}_stderr` and `{split_name}/mean_{metric}_stderr`

- **Ensemble metric logging locations**
  - Lines 990-1006: Logs individual target metrics with stderr
  - Lines 1020-1032: Logs overall mean metrics with stderr
  - Lines 1056-1074: `_log_ensemble_metrics` aggregates all metrics

### Project Conventions

- Standards referenced: Python coding conventions from `.github/instructions/python.instructions.md`
- MLflow logging patterns: Hierarchical metric names with `/` separators
- Error handling: Try/except blocks with silent failures for metric logging

## Key Discoveries

### Current Implementation

The ensemble training already has comprehensive MLflow metric logging:

1. **Individual Target Metrics** (lines 990-1006):
   - For each target endpoint (e.g., "LogD", "KSOL")
   - Logs mean value: `test/{target}_{metric}`
   - Logs stderr: `test/{target}_{metric}_stderr`

2. **Overall Mean Metrics** (lines 1020-1032):
   - Aggregated across all targets
   - Logs mean value: `test/mean_{metric}`
   - Logs stderr: `test/mean_{metric}_stderr`

3. **Bar Plot Generation** (lines 1037-1054):
   - Creates matplotlib figures with `plot_metric_bar`
   - Saves as PNG files
   - Logs as MLflow artifacts under `plots/{split_name}/`

### Metric Types Tracked

All 7 metric types are logged for both individual targets and overall means:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)
- RAE (Relative Absolute Error)
- Spearman ρ (Spearman correlation)
- Pearson r (Pearson correlation)
- Kendall τ (Kendall correlation)

### Data Flow

```
Individual model predictions → Aggregate metrics per target → Compute mean & stderr → Log to MLflow + Generate plots
```

Each bar in the plot corresponds to:
- **Bar value**: Mean of metric across all fold/split models
- **Error bar**: Standard error of metric across models
- **MLflow metric**: Same mean and stderr values

## Recommended Approach

**The metrics are already being logged to MLflow!** The implementation is complete and follows best practices.

### What's Already Working

✅ Individual target metrics logged as `{split_name}/{target}_{metric}` and `{split_name}/{target}_{metric}_stderr`
✅ Overall mean metrics logged as `{split_name}/mean_{metric}` and `{split_name}/mean_{metric}_stderr`
✅ All 7 metric types tracked for both test and blind splits
✅ Error bars (stderr) included in both plots and MLflow metrics
✅ Hierarchical naming convention for easy filtering and visualization
✅ Silent error handling to prevent metric logging failures from crashing training

### Verification Steps

To confirm the metrics are being logged, you can:

1. **Check MLflow UI**: Navigate to the parent run and look for metrics like:
   - `test/LogD_MAE`
   - `test/LogD_MAE_stderr`
   - `test/mean_MAE`
   - `test/mean_MAE_stderr`

2. **Programmatically verify**:
   ```python
   from mlflow import MlflowClient
   client = MlflowClient()
   metrics = client.get_run(run_id).data.metrics
   print([k for k in metrics.keys() if 'test/' in k])
   ```

3. **Check the plots**: The bar plot PNG files logged as artifacts should show the same values

## Implementation Guidance

### No Changes Needed

The current implementation already logs all bar plot data as MLflow metrics. The metrics are available for:
- Programmatic access via MLflow API
- Visualization in MLflow UI
- Comparison across different runs
- Tracking experiments and hyperparameter optimization

### Optional Enhancements

If you want to enhance the logging, consider:

1. **Add per-model metrics**: Log individual model metrics before aggregation
   ```python
   for i, model_pred in enumerate(model_predictions):
       self._mlflow_client.log_metric(
           self.parent_run_id,
           f"{split_name}/model_{i}/{target}_{metric}",
           float(metric_value)
       )
   ```

2. **Log metric distributions**: Save histograms or percentiles
   ```python
   self._mlflow_client.log_metric(
       self.parent_run_id,
       f"{split_name}/{target}_{metric}_p50",
       float(np.percentile(values, 50))
   )
   ```

3. **Add confidence intervals**: Log 95% CI instead of just stderr
   ```python
   ci_95 = 1.96 * stderr
   self._mlflow_client.log_metric(
       self.parent_run_id,
       f"{split_name}/{target}_{metric}_ci95",
       float(ci_95)
   )
   ```

### Testing

Existing test coverage includes:
- `tests/test_log_evaluation_metrics.py` - Tests metric logging
- `tests/test_ensemble_blind_predictions.py` - Tests blind prediction pipeline
- `tests/test_ensemble_chemprop.py` - Tests ensemble training

Add assertions to verify metric logging:
```python
def test_ensemble_metrics_logged():
    # Train ensemble
    ensemble.train_all()

    # Verify metrics logged
    client = MlflowClient()
    metrics = client.get_run(ensemble.parent_run_id).data.metrics

    assert 'test/mean_MAE' in metrics
    assert 'test/mean_MAE_stderr' in metrics
    assert 'test/LogD_MAE' in metrics
```

## Summary

The ensemble bar plot metrics are **already being logged to MLflow**. The implementation in `src/admet/model/chemprop/ensemble.py` logs:

- Mean values for each target and each metric type
- Standard errors for all metrics
- Overall mean and stderr across all targets
- Proper hierarchical naming for easy querying

No code changes are required. If you're not seeing the metrics in MLflow UI, verify:
1. MLflow tracking is enabled in your config
2. The parent run ID is valid
3. Check the run's metrics in MLflow UI or via API

If you need additional metric logging (per-model, distributions, CIs), I can help implement those enhancements.
