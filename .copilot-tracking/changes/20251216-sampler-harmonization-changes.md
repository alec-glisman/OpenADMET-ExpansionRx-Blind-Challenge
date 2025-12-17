# Changes: Sampler Harmonization

## Overview

Implemented unified JointSampler combining task-aware oversampling and curriculum learning with multiplicative weight composition (`w_joint[i] = w_task[i] × w_curriculum[i]`).

## Summary

Successfully integrated task-aware and curriculum-aware sampling into a unified **JointSampler** class. The implementation maintains full backward compatibility while providing a cleaner, more flexible API for sampling configuration.

## Files Changed

### New Files Created

- **src/admet/model/chemprop/joint_sampler.py** - JointSampler class with multiplicative weight composition
- **tests/test_joint_sampler.py** - Comprehensive test suite (9 test classes, 20+ tests)
- **scripts/lib/migrate_sampling_configs.py** - Automated config migration script
- **configs/0-experiment/ensemble_joint_sampling_example.yaml** - Example config

### Modified Files

- **src/admet/model/chemprop/config.py** - Added TaskOversamplingConfig and JointSamplingConfig dataclasses
- **src/admet/model/chemprop/model.py** - Integrated JointSampler into training pipeline
- **configs/0-experiment/ensemble_chemprop_production.yaml** - Migrated to new schema

## Implementation Status

### ✅ Phase 1: Configuration Schema - COMPLETED
- Created TaskOversamplingConfig and JointSamplingConfig dataclasses
- Updated ChempropConfig and EnsembleConfig

### ✅ Phase 2: JointSampler Implementation - COMPLETED
- Multiplicative weight composition
- Rarest task selection for multi-label samples
- Dynamic weight recomputation
- Weight statistics logging

### ✅ Phase 3: Model Integration - COMPLETED
- Added joint_sampling_config parameter
- Updated _prepare_dataloaders() with priority order
- Maintained backward compatibility

### ✅ Phase 4: Testing - COMPLETED
- 20+ tests covering all scenarios
- Task-only, curriculum-only, and combined tests
- Edge case and error handling tests

### ✅ Phase 5: Documentation - COMPLETED
- Example YAML config with annotations
- Comprehensive docstrings
- Migration guide

### 🔄 Phase 6: HPO Integration and MLflow - PARTIAL
- ✅ Weight statistics logging implemented
- ⏸️ HPO search space integration deferred

### ✅ Phase 7: Configuration Migration - COMPLETED
- Automated migration script created
- Example config migrated
- Batch migration available via script

## Key Features

### Multiplicative Weight Composition
```
w_joint[i] = w_task[i] × w_curriculum[i]

where:
- w_task[i] ∝ (count[primary_task[i]])^(-α)
- w_curriculum[i] from CurriculumState.sampling_probs()
- primary_task = rarest task sample has labels for
```

### Per-Quality Metrics

Curriculum learning tracks metrics separately for each quality level:

**During Training:**
- `val_mae_high`, `val_rmse_high`, `val_loss_high`
- `val_mae_medium`, `val_rmse_medium`, `val_loss_medium`
- `val_mae_low`, `val_rmse_low`, `val_loss_low`

**Post-Training:**
- Comprehensive correlation metrics per quality
- CSV artifacts logged to MLflow

### Backward Compatibility

1. Legacy `task_sampling_alpha` still works
2. Legacy `curriculum.enabled` still works
3. Priority: JointSampler > Legacy samplers > Standard shuffle

## Usage Examples

**Task Oversampling Only:**
```yaml
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.5
  curriculum:
    enabled: false
```

**Both Strategies Combined:**
```yaml
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.3
  curriculum:
    enabled: true
    quality_col: "Quality"
```

## Testing

```bash
# Run joint sampler tests
pytest tests/test_joint_sampler.py -v

# Run integration tests
pytest tests/test_chemprop*.py tests/test_curriculum*.py -v
```

## Migration

```bash
# Single config
python scripts/lib/migrate_sampling_configs.py --config-file configs/my_config.yaml

# Directory (dry-run)
python scripts/lib/migrate_sampling_configs.py --config-dir configs/2-hpo-ensemble --dry-run

# Migrate all
python scripts/lib/migrate_sampling_configs.py --config-dir configs
```

## Notes

- All existing tests pass
- Per-quality metrics already implemented in existing code
- Weight statistics provide monitoring capability
- Migration script tested and working
