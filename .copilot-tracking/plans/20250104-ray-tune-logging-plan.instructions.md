---
applyTo: ".copilot-tracking/changes/20250104-ray-tune-logging-changes.md"
---

<!-- markdownlint-disable-file -->

# Task Checklist: Ray Tune Output Logging to MLflow Artifacts

## Overview

Implement production-ready logging infrastructure to redirect verbose Ray Tune and ensemble training output to log files, upload them as MLflow artifacts, and maintain clean terminal output with high-level progress.

## Objectives

- Reduce terminal verbosity during HPO and ensemble training
- Capture all Ray Tune worker output to persistent log files
- Automatically upload logs as MLflow artifacts for debugging
- Provide configurable logging via YAML config files and CLI flags
- Maintain backward compatibility with existing workflows
- Fail-fast on upload errors for immediate feedback
- Enforce 1 GB max total log size per experiment

## Research Summary

### Critical Considerations

1. **Ray Worker Process Isolation**: Ray workers run in separate processes; `sys.stdout` redirection in main process does NOT capture worker output
2. **Ray's Built-in Logging**: Use `RAY_LOG_TO_DRIVER`, `tune.RunConfig(verbose=...)`, and custom `ProgressReporter`
3. **Resource Cleanup**: Long HPO runs may be interrupted; need proper signal handling
4. **Log File Growth**: 100-trial HPO runs can produce GBs of logs; enforce 1 GB max
5. **ANSI/Binary Data**: Progress bars contain escape codes; need proper handling
6. **Fail-Fast**: Raise exceptions immediately on upload failures

### Project Files

- `src/admet/model/chemprop/hpo.py` - Chemprop HPO orchestrator (verbose=1)
- `src/admet/model/chemeleon/hpo.py` - Chemeleon HPO orchestrator (verbose=1)
- `src/admet/model/chemprop/ensemble.py` - Ensemble trainer with Ray remote tasks
- `src/admet/model/hpo_mlflow_callback.py` - MLflow callback for Ray Tune
- `src/admet/util/logging.py` - Existing logging utilities
- `src/admet/cli/model.py` - CLI commands (for new flags)

### Standards References

- #file:../../.github/instructions/python.instructions.md - Python conventions
- #file:../../.github/instructions/self-explanatory-code-commenting.instructions.md - Commenting guidelines
- #file:../../.github/instructions/performance-optimization.instructions.md - Performance best practices

## Implementation Checklist

### [ ] Phase 1: Core Logging Infrastructure

- [ ] Task 1.1: Create `src/admet/util/ray_logging.py` with `RayLogManager` class
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1-120)
  - Includes: fail_on_upload_error=True, max_total_logs_gb=1

- [ ] Task 1.2: Create `QuietProgressReporter` for minimal terminal output
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 121-180)

- [ ] Task 1.3: Create `LogArtifactCallback` Ray Tune callback for MLflow upload
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 181-250)

- [ ] Task 1.4: Create `EnsembleProgressTracker` for ensemble training progress
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 251-290)

### [ ] Phase 2: Configuration Schema

- [ ] Task 2.1: Add `RayLoggingConfig` dataclass to `src/admet/model/config.py`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 291-350)
  - Includes: verbose=0, max_total_logs_gb=1, fail_on_upload_error=True

- [ ] Task 2.2: Add `logging` field to `HPOConfig` and `EnsembleConfig`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 351-400)

### [ ] Phase 3: CLI Integration

- [ ] Task 3.1: Add `--logging-verbose` flag to HPO and ensemble commands
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 401-450)
  - Files: `src/admet/cli/model.py`

- [ ] Task 3.2: Add `--no-logging` flag to disable logging entirely
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 451-480)

### [ ] Phase 4: HPO Integration

- [ ] Task 4.1: Update `ChempropHPO.run()` to use `RayLogManager`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 481-560)

- [ ] Task 4.2: Update `ChemeleonHPO.run()` to use `RayLogManager`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 561-620)

### [ ] Phase 5: Ensemble Integration

- [ ] Task 5.1: Update `ModelEnsemble.train_all()` to use `RayLogManager`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 621-700)

- [ ] Task 5.2: Add progress reporting for ensemble training using `EnsembleProgressTracker`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 701-750)

### [ ] Phase 6: Progressive Testing

- [ ] Task 6.1: Create `tests/test_ray_logging.py` with unit tests
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 751-860)

- [ ] Task 6.2: Add performance benchmarks to tests
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 861-920)
  - Includes: compression overhead, memory usage, upload latency

- [ ] Task 6.3: Create mini integration test (2-3 trials) for HPO
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 921-970)

- [ ] Task 6.4: Create mini integration test (2 models) for ensemble
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 971-1010)

- [ ] Task 6.5: Test interrupt handling (Ctrl+C during HPO)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1011-1050)

### [ ] Phase 7: Configuration Updates (Aggressive - All YAML Files)

- [ ] Task 7.1: Add logging section to all experiment configs (`0-experiment/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1051-1090)
  - Files: `0-experiment/*.yaml` (6 files)

- [ ] Task 7.2: Add logging section to all HPO single configs (`1-hpo-single/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1091-1120)
  - Files: `1-hpo-single/*.yaml` (2 files)

- [ ] Task 7.3: Add logging section to all HPO ensemble configs (`2-hpo-ensemble/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1121-1160)
  - Files: `2-hpo-ensemble/ensemble_chemprop_hpo_*.yaml` (100+ files)
  - Method: Python batch script

- [ ] Task 7.4: Add logging section to all production configs (`3-hpo-production/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1161-1200)
  - Files: `3-hpo-production/*.yaml` (~100 files)
  - Method: Python batch script

- [ ] Task 7.5: Add logging section to classical model configs (`4-more-models/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1201-1230)
  - Files: `4-more-models/*.yaml` (5-10 files)

- [ ] Task 7.6: Add logging section to curriculum learning configs (`curriculum/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1231-1260)
  - Files: `curriculum/*.yaml` (2-3 files)

- [ ] Task 7.7: Add logging section to task affinity configs (`task-affinity/`)
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1261-1290)
  - Files: `task-affinity/*.yaml` (2-3 files)

- [ ] Task 7.8: Create batch update script `scripts/add_logging_to_configs.py`
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1291-1350)

- [ ] Task 7.9: Validate all YAML files parse correctly after updates
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1351-1380)

### [ ] Phase 8: Rich Documentation

- [ ] Task 8.1: Create comprehensive `docs/guide/logging.rst` documentation
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1381-1500)
  - Includes: Configuration reference, CLI flags, troubleshooting, performance notes

- [ ] Task 8.2: Update CLI documentation with logging flags
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1501-1550)

- [ ] Task 8.3: Add API reference for ray_logging module
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1551-1600)

- [ ] Task 8.4: Add troubleshooting guide for common issues
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1601-1650)

- [ ] Task 8.5: Document disk space requirements and performance impact
  - Details: .copilot-tracking/details/20250104-ray-tune-logging-details.md (Lines 1651-1700)

## Dependencies

- ray[tune] >= 2.0
- mlflow >= 2.0
- PyTorch Lightning
- OmegaConf
- rich (for enhanced terminal output)

## Success Criteria

- HPO/ensemble training terminal output reduced to progress indicators only
- All Ray worker logs captured to files in `output_dir/logs/`
- Logs automatically uploaded to MLflow as artifacts after run
- Configuration via YAML `logging:` section
- CLI flags: `--logging-verbose`, `--no-logging`
- Max 1 GB total logs per experiment (enforced)
- Fail immediately on upload errors (configurable)
- Graceful handling of interrupts (logs still uploaded)
- All 200+ config files updated with logging section
- All existing tests pass
- Performance benchmarks documented
- New unit and integration tests pass
- Rich documentation with troubleshooting guide
