<!-- markdownlint-disable-file -->

# Task Details: Ensemble MLflow Metric Aggregation

## Research Reference

**Source Research**: #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md

## Phase 1: Track Child Run IDs

### Task 1.1: Store child run IDs during training

Add storage for child run IDs in `ModelEnsemble` class to enable metric fetching after training.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Add `_child_run_ids` attribute and populate during training

- **Implementation**:
  - Add `self._child_run_ids: List[str] = []` in `__init__`
  - Modify `train_single_model` Ray task to return `mlflow_run_id`
  - Store returned run IDs in results processing loop (around line 1189)
  - Return signature change: `Tuple[str, Dict[str, float], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any], Optional[str]]`

- **Success**:
  - All 25 child run IDs stored after training completes
  - IDs accessible for metric fetching

- **Research References**:
  - #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md (Lines 45-65) - Current worker return signature

- **Dependencies**:
  - None (foundational change)

## Phase 2: Implement Metric Aggregation

### Task 2.1: Create `_aggregate_child_run_metrics()` method

New method to fetch metrics from all child runs and compute aggregates.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Add new method after `_log_ensemble_metrics()`

- **Implementation**:
```python
def _aggregate_child_run_metrics(self) -> None:
    """
    Fetch metrics from all child runs and log aggregated statistics to parent.

    Queries MLflow API for each child run, collects all non-profiling metrics,
    computes mean and stddev across the ensemble, and logs to parent run
    with flattened naming convention: ensemble/{metric}_mean, ensemble/{metric}_stddev.
    """
    if not self._mlflow_client or not self.parent_run_id or not self._child_run_ids:
        logger.warning("Cannot aggregate child metrics: missing MLflow client, parent run, or child IDs")
        return

    # Collect metrics from each child run
    all_metrics: Dict[str, List[float]] = {}

    for run_id in self._child_run_ids:
        try:
            run = self._mlflow_client.get_run(run_id)
            for metric_name, value in run.data.metrics.items():
                # Filter logic applied in Task 2.2
                if self._should_aggregate_metric(metric_name):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        except Exception as e:
            logger.warning("Failed to fetch metrics from child run %s: %s", run_id, e)
            continue

    # Compute and log aggregates
    self._log_aggregated_metrics(all_metrics)
```

- **Success**:
  - Method fetches metrics from all 25 child runs
  - Handles missing/failed runs gracefully
  - Collects metrics into aggregatable structure

- **Research References**:
  - #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md (Lines 95-130) - Option A implementation approach

- **Dependencies**:
  - Task 1.1 (child run IDs stored)

### Task 2.2: Add metric filtering logic (exclude profiling, include best model metrics)

Filter metrics to exclude profiling and include only relevant model metrics.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Add `_should_aggregate_metric()` helper method

- **Implementation**:
```python
def _should_aggregate_metric(self, metric_name: str) -> bool:
    """
    Determine if a metric should be included in ensemble aggregation.

    Parameters
    ----------
    metric_name : str
        Name of the metric to check.

    Returns
    -------
    bool
        True if metric should be aggregated, False otherwise.
    """
    # Exclude profiling metrics
    if metric_name.startswith("profiling"):
        return False

    # Exclude system metrics (CPU, memory, etc.)
    if metric_name.startswith("system/"):
        return False

    # Exclude step/epoch counters
    if metric_name in ("epoch", "step", "global_step"):
        return False

    # Include all others: validation/*, train_*, test/*, best_val_loss, etc.
    return True
```

- **Success**:
  - Profiling metrics excluded (`profiling.*`)
  - System metrics excluded (`system/*`)
  - Step counters excluded
  - All model metrics included (validation, training, test, best_val_loss)

- **Research References**:
  - #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md (Lines 70-85) - Metric categories

- **Dependencies**:
  - None

## Phase 3: Integration and Logging

### Task 3.1: Call aggregation after training completes

Integrate the aggregation call into the ensemble training workflow.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Add call in `_generate_ensemble_outputs()` or after result processing

- **Implementation**:
  - Call `self._aggregate_child_run_metrics()` after `self._log_ensemble_metrics()` in `_generate_ensemble_outputs()`
  - Alternative: Add in the `finally` block of `train_all()` to ensure it runs even on partial failure
  - Log info message when aggregation completes

- **Location**: Around line 1286 after existing `_log_ensemble_metrics()` call

- **Success**:
  - Aggregation runs after all models complete
  - Works even if some models fail (partial ensemble)
  - Logged confirmation message

- **Research References**:
  - #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md (Lines 135-145) - Integration points

- **Dependencies**:
  - Task 2.1 (aggregation method exists)

### Task 3.2: Log aggregated metrics with flattened naming

Log computed aggregates to parent run with consistent naming.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Add `_log_aggregated_metrics()` helper method

- **Implementation**:
```python
def _log_aggregated_metrics(self, all_metrics: Dict[str, List[float]]) -> None:
    """
    Compute and log mean/stddev for all collected metrics.

    Parameters
    ----------
    all_metrics : Dict[str, List[float]]
        Dictionary mapping metric names to list of values from child runs.
    """
    if not all_metrics:
        logger.info("No metrics to aggregate")
        return

    aggregated: Dict[str, float] = {}

    for metric_name, values in all_metrics.items():
        if not values:
            continue

        # Compute statistics
        mean_val = float(np.mean(values))
        aggregated[f"ensemble/{metric_name}_mean"] = mean_val

        if len(values) > 1:
            std_val = float(np.std(values, ddof=1))
            aggregated[f"ensemble/{metric_name}_stddev"] = std_val
        else:
            aggregated[f"ensemble/{metric_name}_stddev"] = 0.0

    # Log all metrics in batch
    if aggregated:
        try:
            mlflow.log_metrics(aggregated)
            logger.info("Logged %d aggregated ensemble metrics to parent run", len(aggregated))
        except Exception as e:
            logger.error("Failed to log aggregated metrics: %s", e)
```

- **Naming Convention**:
  - `ensemble/validation/mean/mae_mean` and `ensemble/validation/mean/mae_stddev`
  - `ensemble/train_loss_mean` and `ensemble/train_loss_stddev`
  - `ensemble/test/LogD/r2_mean` and `ensemble/test/LogD/r2_stddev`

- **Success**:
  - All metrics logged with `ensemble/` prefix
  - Mean and stddev computed correctly (ddof=1 for sample stddev)
  - Batch logging for efficiency

- **Research References**:
  - #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md (Lines 150-170) - Naming convention

- **Dependencies**:
  - Task 2.1 (provides metrics dict)

## Phase 4: Testing

### Task 4.1: Add unit tests for metric aggregation

Create tests to verify aggregation logic.

- **Files**:
  - `tests/model/chemprop/test_ensemble_metrics.py` - New test file or add to existing

- **Test Cases**:
  1. `test_should_aggregate_metric_excludes_profiling` - Verify profiling metrics excluded
  2. `test_should_aggregate_metric_includes_validation` - Verify validation metrics included
  3. `test_log_aggregated_metrics_computes_correct_stats` - Verify mean/stddev calculation
  4. `test_aggregate_child_run_metrics_handles_missing_runs` - Verify graceful handling

- **Implementation**:
```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

class TestEnsembleMetricAggregation:
    def test_should_aggregate_metric_excludes_profiling(self):
        ensemble = ModelEnsemble(mock_config)
        assert not ensemble._should_aggregate_metric("profiling.training.duration")
        assert not ensemble._should_aggregate_metric("profiling.ensemble.n_models")

    def test_should_aggregate_metric_includes_validation(self):
        ensemble = ModelEnsemble(mock_config)
        assert ensemble._should_aggregate_metric("validation/mean/mae")
        assert ensemble._should_aggregate_metric("train_loss")
        assert ensemble._should_aggregate_metric("test/LogD/r2")

    def test_log_aggregated_metrics_computes_correct_stats(self):
        all_metrics = {
            "validation/mean/mae": [0.1, 0.2, 0.15, 0.12, 0.18],
        }
        # Expected: mean=0.15, stddev=0.0387 (approx)
        # Verify computation
```

- **Success**:
  - All test cases pass
  - Edge cases covered (empty metrics, single value, missing runs)
  - Mocking used to avoid MLflow dependencies

- **Dependencies**:
  - All Phase 1-3 tasks complete

## Dependencies

- MLflow >= 2.0 (client API)
- NumPy (statistics)

## Success Criteria

- Metrics from all 25 child runs aggregated
- Parent run shows `ensemble/*_mean` and `ensemble/*_stddev` metrics
- Profiling metrics not aggregated
- MLflow UI displays aggregated metrics for comparison
- Tests validate aggregation logic
