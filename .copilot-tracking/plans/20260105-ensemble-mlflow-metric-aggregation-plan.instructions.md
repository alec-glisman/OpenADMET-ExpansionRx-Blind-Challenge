---
applyTo: ".copilot-tracking/changes/20260105-ensemble-mlflow-metric-aggregation-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Ensemble MLflow Metric Aggregation

## Overview

Aggregate all model metrics (validation, training, test) from 25 child MLflow runs and log mean/stddev to the parent ensemble run for easier ensemble comparison.

## Objectives

- Aggregate metrics from all 25 sub-runs (5 splits × 5 folds) to parent run
- Log mean and stddev for each metric with flattened naming convention
- Include validation, training, and test metrics; exclude profiling metrics
- Capture "best" model metrics only (not last step metrics)

## Research Summary

### Project Files

- [src/admet/model/chemprop/ensemble.py](../../src/admet/model/chemprop/ensemble.py) - Main ensemble orchestration, `_log_ensemble_metrics()` method
- [src/admet/model/chemprop/model.py](../../src/admet/model/chemprop/model.py) - Individual model metric logging patterns

### External References

- #file:../research/20260105-ensemble-mlflow-metric-aggregation-research.md - Full research with current implementation analysis

### Standards References

- #file:../../.github/instructions/python.instructions.md - Python coding conventions
- #file:../../.github/copilot-instructions.md - Project conventions (NumPy docstrings, 120 char lines)

## Implementation Checklist

### [x] Phase 1: Track Child Run IDs

- [x] Task 1.1: Store child run IDs during training
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 15-45)

### [x] Phase 2: Implement Metric Aggregation

- [x] Task 2.1: Create `_aggregate_child_run_metrics()` method
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 47-95)

- [x] Task 2.2: Add metric filtering logic (exclude profiling, include best model metrics)
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 97-125)

### [x] Phase 3: Integration and Logging

- [x] Task 3.1: Call aggregation after training completes
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 127-150)

- [x] Task 3.2: Log aggregated metrics with flattened naming
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 152-180)

### [x] Phase 4: Testing

- [x] Task 4.1: Add unit tests for metric aggregation
  - Details: .copilot-tracking/details/20260105-ensemble-mlflow-metric-aggregation-details.md (Lines 182-220)

## Dependencies

- MLflow client (`mlflow.tracking.MlflowClient`)
- NumPy for statistical calculations
- Existing ensemble training infrastructure

## Success Criteria

- All validation/training/test metrics aggregated from 25 child runs
- Metrics logged to parent run as `ensemble/{original_metric}_mean` and `ensemble/{original_metric}_stddev`
- Profiling metrics excluded from aggregation
- Only "best" model state metrics captured (not last step)
- Metrics visible in MLflow UI for ensemble comparison
- No impact on training performance (metrics fetched after training)
