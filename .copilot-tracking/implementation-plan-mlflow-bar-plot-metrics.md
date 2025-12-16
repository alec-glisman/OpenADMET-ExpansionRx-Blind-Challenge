# Implementation Plan: Log Ensemble Bar Plot Metrics to MLflow

**Date:** 2024-12-16
**Status:** Implementation in Progress
**Priority:** Medium
**Estimated Effort:** 2-3 hours
**Updated:** 2024-12-16 - Incorporated recommendations

## Executive Summary

Add dedicated MLflow metric logging for all bar plot data (values and error bars) under a `plots/` hierarchy, separate from existing `test/` and `blind/` metrics. This provides a dedicated namespace for visualization-related metrics that can be easily queried and compared across runs.

## Current State Analysis

### Existing Metric Logging (Lines 990-1032 in ensemble.py)

Currently logs metrics under split name prefix:

- **Individual targets**: `{split_name}/{target}_{metric}` and `{split_name}/{target}_{metric}_stderr`
- **Overall means**: `{split_name}/mean_{metric}` and `{split_name}/mean_{metric}_stderr`

Examples:

- `test/LogD_MAE` = 0.45
- `test/LogD_MAE_stderr` = 0.03
- `test/mean_MAE` = 0.52
- `test/mean_MAE_stderr` = 0.04

### Bar Plot Data Structure

Each bar plot contains:

- **Labels**: Target endpoint names (e.g., "LogD", "KSOL") + "Mean"
- **Values**: Mean of metric across all fold/split models
- **Errors**: Standard error of the mean
- **Metric types**: MAE, RMSE, R², RAE, Spearman ρ, Pearson r, Kendall τ

### Artifact Logging

Plots are saved as PNG files and logged to MLflow under:

- `plots/test/ensemble_{metric}.png`
- `plots/blind/ensemble_{metric}.png`

## Requirements

### Functional Requirements

1. **Separate Namespace**: All plot-related metrics under `plots/` prefix
2. **Per-Bar Metrics**: Log value and stderr for each bar in the plot
3. **All Metric Types**: Cover all 7 metric types (MAE, RMSE, R², RAE, Spearman ρ, Pearson r, Kendall τ)
4. **Both Splits**: Log for both test and blind predictions
5. **Maintain Existing**: Keep current `test/` and `blind/` metrics unchanged

### Non-Functional Requirements

1. **Consistency**: Use same naming conventions as existing code
2. **Robustness**: Explicit error handling with logging for failures
3. **Performance**: Use batch logging (`mlflow.log_metrics()`) for efficiency
4. **Maintainability**: Clear, documented code following project standards
5. **Edge Case Handling**: Handle NaN values, single models, missing targets
6. **Validation**: Ensure logged metrics match plotted values

## Implementation Design

### Metric Naming Convention

All plot metrics will use this hierarchical structure:

```
plots/{split_name}/{metric_type}/{target}
plots/{split_name}/{metric_type}/{target}_stderr
plots/{split_name}/{metric_type}/mean
plots/{split_name}/{metric_type}/mean_stderr
```

### Examples

For a test split with LogD and KSOL targets:

```python
# MAE metrics
plots/test/MAE/LogD = 0.45
plots/test/MAE/LogD_stderr = 0.03
plots/test/MAE/KSOL = 0.52
plots/test/MAE/KSOL_stderr = 0.04
plots/test/MAE/mean = 0.485
plots/test/MAE/mean_stderr = 0.05

# R² metrics
plots/test/R2/LogD = 0.89
plots/test/R2/LogD_stderr = 0.02
plots/test/R2/mean = 0.87
plots/test/R2/mean_stderr = 0.03
```

For blind predictions (prediction statistics only, no metrics requiring actual values):

```python
# Prediction statistics (mean and std of predictions across models)
plots/blind/predictions/LogD_mean = 1.23
plots/blind/predictions/LogD_stderr = 0.05
plots/blind/predictions/KSOL_mean = -0.45
plots/blind/predictions/KSOL_stderr = 0.03

# Metadata
plots/blind/predictions/n_models = 12
```

## Code Changes

### File: `src/admet/model/chemprop/ensemble.py`

#### Change 1: Add Label Sanitization Utility (after line 65)

Add shared sanitization function for consistent label formatting:

```python
def _sanitize_metric_label(label: str) -> str:
    """
    Sanitize label for use in MLflow metric names.

    Converts labels to lowercase and replaces special characters with underscores
    to ensure valid MLflow metric names.

    Parameters
    ----------
    label : str
        Raw label (e.g., "Log KSOL", "Spearman $\\rho$")

    Returns
    -------
    str
        Sanitized label (e.g., "log_ksol", "spearman_rho")
    """
    return (
        label.lower()
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("-", "_")
        .replace("$", "")
        .replace("^", "")
        .replace("\\", "")
        .replace("ρ", "rho")
        .replace("τ", "tau")
        .replace("²", "2")
    )
```

#### Change 2: Add Plot Metrics Logging Method (after line 800)

Add new helper method to log plot metrics with batch API and metadata:

```python
def _log_plot_metrics(
    self,
    split_name: str,
    metric_type: str,
    safe_metric: str,
    labels: List[str],
    means: List[float],
    errors: List[float],
    n_models: int,
) -> None:
    """
    Log bar plot data as MLflow metrics under plots/ prefix using batch API.

    This method logs all bar values and standard errors in a single batch operation
    for better performance. It also includes metadata about the ensemble size.

    Parameters
    ----------
    split_name : str
        Split name (e.g., "test", "blind")
    metric_type : str
        Display metric type (e.g., "MAE", r"$R^2$")
    safe_metric : str
        Sanitized metric name for MLflow (e.g., "MAE", "R2")
    labels : List[str]
        Bar labels (target names + "Mean")
    means : List[float]
        Bar values (metric means across models)
    errors : List[float]
        Error bar values (standard errors)
    n_models : int
        Number of models in the ensemble
    """
    if not self._mlflow_client or not self.parent_run_id:
        return

    # Prepare batch metrics dictionary
    metrics_dict = {}

    # Log each bar's value and stderr
    for label, mean_val, stderr_val in zip(labels, means, errors):
        # Handle NaN values
        if np.isnan(mean_val):
            logger.warning(
                f"Skipping NaN metric for {label} in {metric_type} ({split_name})"
            )
            continue

        # Sanitize label using shared utility
        safe_label = _sanitize_metric_label(label)

        # Add to batch
        metrics_dict[f"plots/{split_name}/{safe_metric}/{safe_label}"] = float(mean_val)
        metrics_dict[f"plots/{split_name}/{safe_metric}/{safe_label}_stderr"] = float(stderr_val)

    # Add metadata: number of models
    metrics_dict[f"plots/{split_name}/{safe_metric}/n_models"] = float(n_models)

    # Log all metrics in a single batch
    try:
        mlflow.log_metrics(metrics_dict)
        logger.debug(
            f"Logged {len(metrics_dict)} plot metrics for {safe_metric} ({split_name})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to log plot metrics for {safe_metric} ({split_name}): {e}"
        )
```

#### Change 3: Integrate Plot Metrics Logging (inside `_generate_metrics_bar_plot` after line 1054)

Add call to new helper method before closing the figure:

```python
            # Save plot with metric type in filename (sanitize special characters)
            safe_metric_name = (
                metric_type.replace("$", "").replace(" ", "_").replace("^", "").replace("ρ", "rho").replace("τ", "tau")
            )
            plot_path = plot_dir / f"ensemble_{safe_metric_name.lower()}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Log plot data as MLflow metrics under plots/ prefix
            safe_metric = metric_name_map.get(metric_type, metric_type)
            self._log_plot_metrics(
                split_name=split_name,
                metric_type=metric_type,
                safe_metric=safe_metric,
                labels=labels,
                means=means,
                errors=errors,
                n_models=len(model_predictions),
            )

            plt.close(fig)
```

## Key Improvements from Recommendations

1. **Batch Logging**: Using `mlflow.log_metrics()` instead of individual calls for better performance
2. **Helper Method**: Clean separation of concerns with `_log_plot_metrics()` method
3. **Shared Sanitization**: `_sanitize_metric_label()` utility function for consistency
4. **Edge Case Handling**: Explicit NaN checks with warnings
5. **Metadata Logging**: Includes `n_models` for context
6. **Better Logging**: Debug and warning messages for troubleshooting

## ~~Alternative: Inline Implementation~~ (Deprecated)

~~If you prefer not to add a helper method, directly add logging in the loop (after line 1054):~~

**Note**: We're using the helper method approach as recommended for better maintainability.

```python
            # Save plot with metric type in filename (sanitize special characters)
            safe_metric_name = (
                metric_type.replace("$", "").replace(" ", "_").replace("^", "").replace("ρ", "rho").replace("τ", "tau")
            )
            plot_path = plot_dir / f"ensemble_{safe_metric_name.lower()}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Log plot data as MLflow metrics under plots/ prefix
            if self._mlflow_client and self.parent_run_id:
                safe_metric = metric_name_map.get(metric_type, metric_type)
                for label, mean_val, stderr_val in zip(labels, means, errors):
                    safe_label = (
                        label.replace(" ", "_")
                        .replace(">", "gt")
                        .replace("<", "lt")
                        .replace("-", "_")
                        .lower()
                    )
                    try:
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"plots/{split_name}/{safe_metric}/{safe_label}",
                            float(mean_val),
                        )
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"plots/{split_name}/{safe_metric}/{safe_label}_stderr",
                            float(stderr_val),
                        )
                    except Exception:
                        pass  # Silently ignore metric logging failures

            plt.close(fig)
```

## Testing Strategy

### Unit Tests

Add to `tests/test_ensemble_chemprop.py`:

```python
def test_plot_metrics_logged_to_mlflow(mock_mlflow, ensemble_config):
    """Test that bar plot data is logged as MLflow metrics under plots/ prefix."""
    ensemble = ChempropEnsemble.from_config(ensemble_config)
    ensemble.train_all()

    # Verify plot metrics were logged
    client = MlflowClient()
    metrics = client.get_run(ensemble.parent_run_id).data.metrics

    # Check for plots/ prefix metrics
    plot_metrics = [k for k in metrics.keys() if k.startswith("plots/")]
    assert len(plot_metrics) > 0, "No metrics logged under plots/ prefix"

    # Check for expected structure
    assert any("plots/test/MAE/" in k for k in plot_metrics)
    assert any("_stderr" in k for k in plot_metrics)

    # Verify separation from test/ metrics
    test_metrics = [k for k in metrics.keys() if k.startswith("test/") and not k.startswith("test/mean")]
    assert len(test_metrics) > 0, "Existing test/ metrics should still be logged"

    ensemble.close()


def test_plot_metrics_all_metric_types(mock_mlflow, ensemble_config):
    """Test that all 7 metric types are logged under plots/."""
    ensemble = ChempropEnsemble.from_config(ensemble_config)
    ensemble.train_all()

    client = MlflowClient()
    metrics = client.get_run(ensemble.parent_run_id).data.metrics
    plot_metrics = [k for k in metrics.keys() if k.startswith("plots/")]

    # Check all metric types present
    metric_types = ["MAE", "RMSE", "R2", "RAE", "spearman_rho", "pearson_r", "kendall_tau"]
    for metric_type in metric_types:
        assert any(f"/{metric_type}/" in k for k in plot_metrics), \
            f"No plots/ metrics found for {metric_type}"

    ensemble.close()


def test_plot_metrics_stderr_values(mock_mlflow, ensemble_config):
    """Test that stderr values are logged for all plot metrics."""
    ensemble = ChempropEnsemble.from_config(ensemble_config)
    ensemble.train_all()

    client = MlflowClient()
    metrics = client.get_run(ensemble.parent_run_id).data.metrics

    # For every plots/ metric, check there's a corresponding _stderr metric
    plot_value_metrics = [
        k for k in metrics.keys()
        if k.startswith("plots/") and not k.endswith("_stderr")
    ]

    for value_metric in plot_value_metrics:
        stderr_metric = f"{value_metric}_stderr"
        assert stderr_metric in metrics, \
            f"Missing stderr metric for {value_metric}"

    ensemble.close()
```

### Integration Tests

Add to `tests/test_ensemble_blind_predictions.py`:

```python
def test_blind_plot_metrics_logged():
    """Test that blind predictions also log plot metrics."""
    # Setup ensemble with blind data
    ensemble = create_test_ensemble_with_blind_data()
    ensemble.train_all()

    client = MlflowClient()
    metrics = client.get_run(ensemble.parent_run_id).data.metrics

    # Check for blind plot metrics
    blind_plot_metrics = [k for k in metrics.keys() if k.startswith("plots/blind/")]
    assert len(blind_plot_metrics) > 0, "No blind plot metrics logged"

    ensemble.close()
```

## Verification Steps

### Manual Verification

1. **Run ensemble training**:

   ```bash
   python -m admet.model.chemprop.ensemble --config configs/ensemble_chemprop.yaml
   ```

2. **Check MLflow UI**:
   - Navigate to the parent run
   - Look for metrics starting with `plots/`
   - Verify hierarchical structure: `plots/test/MAE/logd`, etc.

3. **Programmatic check**:

   ```python
   from mlflow import MlflowClient

   client = MlflowClient()
   run = client.get_run("YOUR_RUN_ID")

   # Get all plot metrics
   plot_metrics = {
       k: v for k, v in run.data.metrics.items()
       if k.startswith("plots/")
   }

   # Print organized by metric type
   from collections import defaultdict
   by_metric = defaultdict(list)
   for k, v in plot_metrics.items():
       parts = k.split("/")
       if len(parts) >= 3:
           metric_type = parts[2]
           by_metric[metric_type].append((k, v))

   for metric_type, items in sorted(by_metric.items()):
       print(f"\n{metric_type}:")
       for k, v in sorted(items):
           print(f"  {k}: {v:.4f}")
   ```

4. **Compare with existing metrics**:

   ```python
   # Verify plots/ metrics match test/ metrics
   test_metrics = {
       k: v for k, v in run.data.metrics.items()
       if k.startswith("test/") and not k.startswith("test/mean")
   }

   # Should have corresponding values
   for test_key, test_value in test_metrics.items():
       # Convert test/LogD_MAE -> plots/test/MAE/logd
       parts = test_key.split("/")
       if len(parts) == 2:
           target_metric = parts[1]
           if "_stderr" in target_metric:
               target, metric = target_metric.replace("_stderr", "").rsplit("_", 1)
               plot_key = f"plots/test/{metric}/{target.lower()}_stderr"
           else:
               target, metric = target_metric.rsplit("_", 1)
               plot_key = f"plots/test/{metric}/{target.lower()}"

           if plot_key in plot_metrics:
               assert abs(plot_metrics[plot_key] - test_value) < 1e-6, \
                   f"Mismatch: {test_key}={test_value} vs {plot_key}={plot_metrics[plot_key]}"
   ```

## Rollout Plan

### Phase 1: Implementation (Day 1)

1. Add `_log_plot_metrics` helper method to `ChempropEnsemble` class
2. Add method call in `_generate_metrics_bar_plot` after figure is saved
3. Add comprehensive docstrings
4. Commit with message: "feat: log ensemble bar plot metrics under plots/ prefix"

### Phase 2: Testing (Day 1-2)

1. Add unit tests to `test_ensemble_chemprop.py`
2. Add integration tests to `test_ensemble_blind_predictions.py`
3. Run full test suite: `pytest tests/test_ensemble*.py -v`
4. Commit with message: "test: add coverage for plots/ metric logging"

### Phase 3: Verification (Day 2)

1. Run ensemble training with test config
2. Verify metrics in MLflow UI
3. Run verification script to compare values
4. Document findings in MLflow artifact structure guide

### Phase 4: Documentation (Day 2-3)

1. Update `docs/guide/mlflow_artifacts.rst` with plots/ metric structure
2. Add example queries and visualizations
3. Update README.md with new metric namespace
4. Create example notebook showing how to query plot metrics

## Risk Assessment

### Low Risk

- **Separate namespace**: New `plots/` prefix won't conflict with existing metrics
- **Silent error handling**: Failures won't break training
- **Backward compatible**: Existing metrics remain unchanged

### Mitigation Strategies

1. **Testing**: Comprehensive unit and integration tests
2. **Monitoring**: Log warnings for metric logging failures
3. **Validation**: Compare plots/ values with test/ values to ensure consistency
4. **Rollback**: Easy to disable by commenting out method call

## Success Criteria

- [ ] All bar plot data logged as MLflow metrics under `plots/` prefix
- [ ] Metrics organized by: split → metric type → target
- [ ] Both value and stderr logged for each bar
- [ ] All 7 metric types covered (MAE, RMSE, R², RAE, Spearman ρ, Pearson r, Kendall τ)
- [ ] Works for both test and blind predictions
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests verify end-to-end pipeline
- [ ] Manual verification confirms MLflow UI shows correct values
- [ ] Documentation updated
- [ ] Existing `test/` and `blind/` metrics unchanged

## Future Enhancements

### Phase 2 Features (Optional)

1. **Per-model metrics**: Log individual model metrics before aggregation

   ```python
   plots/test/MAE/logd/model_0 = 0.42
   plots/test/MAE/logd/model_1 = 0.48
   ```

2. **Metric distributions**: Log percentiles or histograms

   ```python
   plots/test/MAE/logd/p25 = 0.40
   plots/test/MAE/logd/p50 = 0.45
   plots/test/MAE/logd/p75 = 0.50
   ```

3. **Confidence intervals**: Log 95% CI instead of just stderr

   ```python
   plots/test/MAE/logd/ci95_lower = 0.39
   plots/test/MAE/logd/ci95_upper = 0.51
   ```

4. **Interactive plots**: Save plotly figures for interactive exploration in MLflow

## References

- Current implementation: `src/admet/model/chemprop/ensemble.py` lines 863-1056
- MLflow client API: `mlflow.client.MlflowClient.log_metric()`
- Project conventions: `.github/instructions/python.instructions.md`
- Metric computation: `admet.plot.metrics.plot_metric_bar()`

## Questions & Decisions

### Q: Should we log plots/ metrics for individual models too?

**Decision**: Start with ensemble-level only. Individual models can be added in Phase 2 if needed.

### Q: What about blind predictions where we don't have actual values?

**Decision**: Log plot metrics anyway (predictions only). Useful for tracking prediction distributions.

### Q: Should we use batch logging (`log_metrics()`) for performance?

**Decision**: Keep individual calls for now. Error handling is easier, and performance impact is minimal.

### Q: Where should documentation go?

**Decision**: Update `docs/guide/mlflow_artifacts.rst` and add example notebook in `notebooks/`.

---

**Next Steps**: Review plan, get approval, begin Phase 1 implementation.
