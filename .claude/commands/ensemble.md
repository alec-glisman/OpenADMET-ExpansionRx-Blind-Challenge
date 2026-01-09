# Ensemble Training

Train model ensembles across split/fold directory structure.

## Command

```bash
admet model ensemble -c <config_path> --max-parallel <N>
```

## Directory Structure

Expects data organized as:
```
assets/dataset/splits/
├── split_0/
│   ├── fold_0/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── fold_1/
│   └── ...
├── split_1/
└── ...
```

## Example

```bash
# Train 5 splits x 5 folds = 25 models
admet model ensemble -c configs/3-production/ensemble_chemprop.yaml --max-parallel 4
```

## Config Requirements

```yaml
ensemble:
  enabled: true
  n_models: 5
  aggregation: mean  # or median

data:
  data_dir: assets/dataset/splits/  # Parent directory
```

## Resource Management

- `--max-parallel 4` - Limit concurrent training jobs
- Uses Ray for parallelization
- Each fold trains independently

## Output

- Per-fold models saved to split_N/fold_M directories
- Ensemble predictions with uncertainty (mean +/- std)
