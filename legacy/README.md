# Legacy Scripts

This folder contains deprecated migration scripts and one-off utilities that were used during the development and evolution of the OpenADMET Challenge codebase. These scripts are retained for historical reference only and should **not** be used in production.

## Contents

| Script                                  | Purpose                                               | Status    |
| --------------------------------------- | ----------------------------------------------------- | --------- |
| `add_logging_to_configs.py`             | Added logging configuration to existing YAML configs  | Completed |
| `backfill_ensemble_metrics.py`          | Backfilled missing ensemble metrics to MLflow         | Completed |
| `backfill_exp5.py`                      | Backfilled experiment 5 results                       | Completed |
| `config_migration.py`                   | Migrated configs to `model.type` discriminator pattern | Completed |
| `debug_per_quality_metrics.py`          | Debug script for per-quality metrics analysis         | One-off   |
| `fix_config_performance_optimization.py` | Fixed performance optimization settings in configs    | Completed |
| `migrate_configs_to_new_api.py`         | Migrated to `UnifiedModelConfig` API                  | Completed |
| `migrate_ray_configs.py`                | Updated Ray Tune configuration format                 | Completed |
| `migrate_sampling_configs.py`           | Migrated to `joint_sampling` schema                   | Completed |
| `scrape_gradio_20251216.py`             | One-time leaderboard scrape from Gradio interface     | One-off   |

## Configuration Migration History

The codebase underwent several configuration schema changes:

### v1.0 → v1.1: Unified Config Structure

```yaml
# OLD (v1.0)
model:
  depth: 5
  message_hidden_dim: 600

# NEW (v1.1+)
model:
  type: chemprop
  chemprop:
    depth: 5
    message_hidden_dim: 600
```

### v1.1 → v1.2: Joint Sampling Schema

```yaml
# OLD (v1.1)
optimization:
  task_sampling_alpha: 0.5
curriculum:
  enabled: true

# NEW (v1.2+)
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.5
  curriculum:
    enabled: false
```

### v1.2 → v1.3: Ray Configuration

Ray Tune and ensemble configurations were consolidated under a unified schema.

## Warning

⚠️ **Do not run these scripts** unless you fully understand their purpose. They may:

- Overwrite existing configuration files
- Modify MLflow experiment data
- Produce unexpected results with the current codebase version

For current migration needs, consult the [documentation](../docs/guide/migration.rst) or open an issue.
