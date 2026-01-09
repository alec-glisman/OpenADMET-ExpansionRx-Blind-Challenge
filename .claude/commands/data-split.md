# Data Splitting

Split data using cluster-based methods for robust cross-validation.

## Command

```bash
admet data split <input_csv> --output <output_dir> [options]
```

## Example

```bash
admet data split data.csv --output assets/dataset/splits/ \
    --cluster-method bitbirch \
    --n-splits 5 \
    --n-folds 5
```

## Clustering Methods

- `bitbirch` - BitBirch (default, recommended)
- `butina` - Butina clustering
- `kmeans` - K-means on fingerprints
- `scaffold` - Scaffold-based splitting
- `random` - Random assignment

## Output Structure

```
output_dir/
├── split_0/
│   ├── fold_0/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── fold_1/
│   └── ...
├── split_1/
└── ...
```

## Options

```bash
--cluster-method bitbirch  # Clustering algorithm
--n-splits 5               # Number of random splits
--n-folds 5                # Cross-validation folds per split
--smiles-col SMILES        # SMILES column name
--quality-col quality      # Quality category column (optional)
```

## Key Files

- `src/admet/data/split.py` - Splitting implementation
- `src/admet/cli/data.py` - CLI commands
