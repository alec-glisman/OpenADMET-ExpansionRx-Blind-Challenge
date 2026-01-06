# Changes: Ensemble MLflow Metric Aggregation

## Implementation Date

2026-01-05

## Overview

Aggregate all model metrics (validation, training, test) from 25 child MLflow runs and log mean/stddev to the parent ensemble run for easier ensemble comparison.

## Files Added
<!-- Updated as implementation progresses -->
- tests/model/chemprop/test_ensemble_metrics.py - Comprehensive unit tests for metric aggregation functionality (12 tests, all passing)

## Files Modified
<!-- Updated as implementation progresses -->
- src/admet/model/chemprop/ensemble.py - Added _child_run_ids attribute and modified train_single_model to track MLflow run IDs from child runs
- src/admet/model/chemprop/ensemble.py - Added _aggregate_child_run_metrics(), _should_aggregate_metric(), and_log_aggregated_metrics() methods for metric aggregation
- src/admet/model/chemprop/ensemble.py - Integrated _aggregate_child_run_metrics() call in _generate_ensemble_outputs() after_log_ensemble_metrics()

## Files Removed
<!-- Updated as implementation progresses -->

## Release Summary
<!-- To be completed when all phases are done -->

### Feature: Ensemble MLflow Metric Aggregation

Successfully implemented comprehensive metric aggregation from child MLflow runs to parent ensemble run.

**Key Capabilities:**

- Tracks all 25 child run IDs (5 splits × 5 folds) during ensemble training
- Fetches metrics from each child run via MLflow API
- Filters out profiling and system metrics, keeping only model performance metrics
- Computes mean and standard deviation (ddof=1) across all child runs
- Logs aggregated metrics to parent run with `ensemble/{metric}_mean` and `ensemble/{metric}_stddev` naming
- Handles missing/failed child runs gracefully with warning logs
- Supports single-model ensembles (stddev=0 when n=1)

**Implementation Details:**

- Added `_child_run_ids` attribute to ModelEnsemble class
- Modified `train_single_model` Ray task to return MLflow run ID
- Created three new methods:
  - `_should_aggregate_metric()`: Filters metrics by name pattern
  - `_log_aggregated_metrics()`: Computes and logs statistics
  - `_aggregate_child_run_metrics()`: Orchestrates fetch-filter-aggregate-log pipeline
- Integrated call to `_aggregate_child_run_metrics()` after `_log_ensemble_metrics()` in `_generate_ensemble_outputs()`

**Testing:**

- 12 comprehensive unit tests covering:
  - Metric filtering (excludes profiling, system, step counters)
  - Metric inclusion (validation, training, test metrics)
  - Statistical computation (mean, stddev with ddof=1)
  - Edge cases (single model, empty metrics, missing client)
  - Error handling (failed child run fetches)
  - End-to-end aggregation with mixed metric types
- All tests passing ✓

**Files Modified:**

1. [src/admet/model/chemprop/ensemble.py](../../src/admet/model/chemprop/ensemble.py) - Core implementation
2. [tests/model/chemprop/test_ensemble_metrics.py](../../tests/model/chemprop/test_ensemble_metrics.py) - Test suite
