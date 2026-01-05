<!-- markdownlint-disable-file -->

# Task Research Notes: Ensemble MLflow Metric Aggregation

## Research Executed

### File Analysis

- [src/admet/model/chemprop/ensemble.py](../../src/admet/model/chemprop/ensemble.py)
  - `ModelEnsemble` class manages 5x5 CV ensemble training with Ray parallelization
  - Parent MLflow run created in `_init_mlflow()` (line 304)
  - Nested child runs created per model via `config.mlflow.nested = True` (line 944)
  - `_all_metrics` dictionary stores metrics per model after training (line 1187)
  - `_log_ensemble_metrics()` exists (line 1852) but only logs trainer callback metrics

- [src/admet/model/chemprop/model.py](../../src/admet/model/chemprop/model.py)
  - Individual models log detailed metrics via `_log_metrics_with_retry()` (line 2455)
  - Metric naming format: `{split_name}/{target}/{metric_name}` and `{split_name}/mean/{metric_name}`
  - Example: `validation/mean/mae`, `validation/mean/rmse`, `validation/LogD/r2`
  - Trainer callback metrics include: `train_loss`, `val_loss`, `validation/*` metrics

### Code Search Results

- `_all_metrics` population (ensemble.py lines 1187-1190):
  - Populated from `model.trainer.callback_metrics` after each model trains
  - Contains PyTorch Lightning logged metrics like `train_loss`, `val_loss`
  - Does NOT contain the detailed `validation/mean/*` metrics logged to MLflow

- Current `_log_ensemble_metrics()` implementation (lines 1854-1876):
  ```python
  def _log_ensemble_metrics(self) -> None:
      if not self._all_metrics or not self._mlflow_client or not self.parent_run_id:
          return

      metric_names: set[str] = set()
      for metrics in self._all_metrics.values():
          metric_names.update(metrics.keys())

      ensemble_metrics = {}
      for metric_name in metric_names:
          values = [m[metric_name] for m in self._all_metrics.values() if metric_name in m]
          if values:
              ensemble_metrics[f"ensemble_{metric_name}_mean"] = np.mean(values)
              if len(values) > 1:
                  ensemble_metrics[f"ensemble_{metric_name}_std"] = np.std(values, ddof=1)
                  ensemble_metrics[f"ensemble_{metric_name}_stderr"] = np.std(values, ddof=1) / np.sqrt(len(values))
              else:
                  ensemble_metrics[f"ensemble_{metric_name}_std"] = 0.0
                  ensemble_metrics[f"ensemble_{metric_name}_stderr"] = 0.0

      mlflow.log_metrics({k: float(v) for k, v in ensemble_metrics.items()})
  ```

### Project Conventions

- Standards referenced: NumPy-style docstrings, 120 char line limit
- Metric naming: MLflow-safe names via `_sanitize_mlflow_metric_name()`
- Error handling: Try-except with logging, graceful degradation

## Key Discoveries

### Current Metric Flow

1. **Individual Model Training** (Ray worker):
   - Model trains and logs metrics to nested MLflow child run
   - Metrics include per-target and mean values: `validation/mean/mae`, `validation/LogD/r2`, etc.
   - Trainer callback metrics (train_loss, val_loss) captured in `metrics` dict
   - Returns `(model_key, metrics, test_preds, blind_preds, prof_data)`

2. **Ensemble Aggregation** (main process):
   - Collects `metrics` dict into `self._all_metrics[model_key]`
   - **Gap**: Only trainer callback metrics are collected, NOT the detailed `validation/mean/*` metrics
   - `_log_ensemble_metrics()` aggregates only what's in `_all_metrics`

3. **Current Logged Ensemble Metrics**:
   - `ensemble_train_loss_mean/std/stderr`
   - `ensemble_val_loss_mean/std/stderr`
   - Limited utility for ensemble comparison

### The Gap

The user wants aggregated metrics like `validation/mean/mae/mean` and `validation/mean/mae/stddev` logged to the parent run. However:

1. The detailed validation metrics (mae, rmse, r2, etc.) are logged directly to MLflow by each child run
2. These metrics are NOT returned in the `metrics` dict to the ensemble orchestrator
3. Therefore, the ensemble cannot aggregate them without either:
   - **Option A**: Fetching metrics from MLflow child runs after training completes
   - **Option B**: Expanding what the worker returns to include detailed validation metrics
   - **Option C**: Computing metrics from stored predictions (already aggregated in ensemble)

## Alternative Approaches

### Option A: Fetch Metrics from MLflow Child Runs

**Description**: After all models train, query MLflow API to retrieve metrics from each child run, then aggregate.

**Benefits**:
- No changes to worker code
- Access to ALL metrics logged by individual models
- Clean separation of concerns

**Trade-offs**:
- Additional MLflow API calls (25 calls for child run metrics)
- Requires child run IDs to be stored during training
- Slight delay after training completion

**Implementation**:
```python
def _fetch_and_aggregate_child_metrics(self) -> None:
    """Fetch metrics from all child runs and aggregate to parent."""
    if not self._mlflow_client or not self.parent_run_id:
        return

    # Get all child runs under this parent
    child_runs = self._mlflow_client.search_runs(
        experiment_ids=[...],
        filter_string=f"tags.mlflow.parentRunId = '{self.parent_run_id}'"
    )

    # Collect metrics from each child
    all_metrics: Dict[str, List[float]] = {}
    for run in child_runs:
        for metric_name, value in run.data.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Aggregate and log to parent
    for metric_name, values in all_metrics.items():
        if values:
            mlflow.log_metric(f"{metric_name}/mean", np.mean(values))
            mlflow.log_metric(f"{metric_name}/stddev", np.std(values, ddof=1))
```

### Option B: Expand Worker Return Data

**Description**: Modify `train_single_model` to return additional metrics beyond trainer callbacks.

**Benefits**:
- No additional MLflow API calls
- Immediate availability of metrics
- Full control over which metrics to aggregate

**Trade-offs**:
- Requires modifying Ray worker function
- Increases return payload size
- Need to add metric computation in worker

**Implementation**: Add to worker after model.fit():
```python
# Collect detailed validation metrics for ensemble aggregation
detailed_metrics = {}
if hasattr(model, '_mlflow_client') and model.mlflow_run_id:
    run = model._mlflow_client.get_run(model.mlflow_run_id)
    for key, val in run.data.metrics.items():
        if key.startswith('validation/'):
            detailed_metrics[key] = val
# Return detailed_metrics along with existing metrics
```

### Option C: Compute from Stored Predictions (Already Implemented Partially)

**Description**: The ensemble already computes metrics from aggregated predictions in `_generate_metrics_bar_plot()`.

**Benefits**:
- Metrics computed from ensemble predictions (true ensemble behavior)
- Already implemented infrastructure
- Consistent with prediction-based evaluation

**Trade-offs**:
- Computes metrics on AGGREGATED predictions, not per-model metrics
- Different from aggregating individual model metrics
- Already logs some metrics like `test/mean_mae`

**Current Implementation** (lines 1771-1822):
```python
# Already logs individual target metrics to parent run
self._mlflow_client.log_metric(
    self.parent_run_id,
    f"{split_name}/{safe_target}_{safe_metric}",
    float(mean_val),
)
# And overall mean metrics
self._mlflow_client.log_metric(
    self.parent_run_id,
    f"{split_name}/mean_{safe_metric}",
    float(overall_mean),
)
```

## Recommended Approach

**Option A: Fetch Metrics from MLflow Child Runs** is recommended because:

1. **Minimal Code Changes**: No modification to Ray worker return signature
2. **Complete Metric Access**: All metrics logged by individual models are accessible
3. **Accurate Aggregation**: Aggregates actual per-model metrics (not ensemble predictions)
4. **Extensible**: Can aggregate any metric logged in the future without code changes
5. **Clean Architecture**: Metric aggregation happens at ensemble level, not worker level

### Implementation Details

The implementation requires:

1. **Track Child Run IDs**: Store child run IDs as models complete
2. **New Method**: `_aggregate_child_run_metrics()` to fetch and aggregate
3. **Call After Training**: Add call in `_generate_ensemble_outputs()` or `train_all()` finally block
4. **Metric Naming**: Use `{original_metric}/mean` and `{original_metric}/stddev` format

### Metric Categories to Aggregate

Based on the ChempropModel logging:
- `validation/mean/mae` → `validation/mean/mae/mean`, `validation/mean/mae/stddev`
- `validation/mean/rmse` → `validation/mean/rmse/mean`, `validation/mean/rmse/stddev`
- `validation/mean/r2` → `validation/mean/r2/mean`, `validation/mean/r2/stddev`
- `validation/mean/pearson_r` → etc.
- `train_loss`, `val_loss` → already aggregated

## Implementation Guidance

- **Objectives**: Aggregate all validation metrics from 25 child runs and log mean/stddev to parent
- **Key Tasks**:
  1. Store child run IDs during training (modify worker or use MLflow parent tag query)
  2. Create `_aggregate_child_run_metrics()` method in `ModelEnsemble`
  3. Call after all training completes (in `finally` block or `_generate_ensemble_outputs`)
  4. Log aggregated metrics with naming convention: `{metric}/mean`, `{metric}/stddev`
- **Dependencies**: MLflow client, numpy for statistics
- **Success Criteria**:
  - All `validation/mean/*` metrics aggregated across 25 models
  - Mean and stddev logged to parent run
  - Metrics visible in MLflow UI for ensemble comparison
  - No impact on training performance (metrics fetched after training)
