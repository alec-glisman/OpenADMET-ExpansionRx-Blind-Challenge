# Scripts Directory

This directory contains shell scripts for training, evaluation, and development workflows.

## Training Scripts

### `train_chemprop_ensembles.sh`

Trains Chemprop ensemble models across multiple data splits and folds using Ray parallelization.

**Usage:**

```bash
./scripts/train_chemprop_ensembles.sh [--dry-run] [--max-parallel N] [--log-level LEVEL]
```

**Options:**

- `--dry-run` - Print commands without executing
- `--max-parallel N` - Override maximum parallel models (default: from config)
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

- Iterates through all configured data directories
- Creates temporary configs with updated `data_dir` paths
- Tracks success/failure/skipped counts
- Logs results to MLflow

### `train_chemprop_model.sh`

Trains individual Chemprop models for each split/fold combination sequentially.

**Usage:**

```bash
./scripts/train_chemprop_model.sh [--dry-run] [--log-level LEVEL] [--splits N] [--folds N]
```

**Options:**

- `--dry-run` - Print commands without executing
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--splits N` - Limit number of splits to train (default: all)
- `--folds N` - Limit number of folds per split (default: all)

**Features:**

- Discovers split/fold directory structure automatically
- Trains models one at a time without Ray parallelization
- Useful for debugging or resource-constrained environments

## Utility Scripts

### `mlflow_server.sh`

Starts a local MLflow tracking server for experiment logging.

**Usage:**

```bash
./scripts/mlflow_server.sh
```

**Details:**

- Starts MLflow server on `http://127.0.0.1:8080`
- Run this before training to enable experiment tracking
- Access the MLflow UI at <http://127.0.0.1:8080>

### `rebuild-docs-precommit.sh`

Pre-commit hook script that rebuilds documentation.

**Usage:**

This script is called automatically by pre-commit hooks. Do not run manually.

**Features:**

- Builds Sphinx documentation
- Stages new documentation files for commit
- Does not fail the commit if docs build fails

## Library

### `lib/common.sh`

Shared library containing common functions and variables used by training scripts.

**Features:**

- Data directory definitions
- Logging functions (info, warn, error, cmd, section)
- Configuration helpers (create/cleanup temp configs)
- Validation helpers (check files and directories)
- Directory discovery (find splits and folds)
- Summary reporting

**Usage in scripts:**

```bash
source "$SCRIPT_DIR/lib/common.sh"
```

## Configuration

Training scripts use configuration files from the `configs/` directory:

- `configs/ensemble_chemprop.yaml` - Ensemble training configuration
- `configs/single_chemprop.yaml` - Single model training configuration

## Data Directory Structure

Training scripts expect data in the following structure:

```text
assets/dataset/split_train_val/v3/
├── quality_high/
│   └── bitbirch/
│       ├── multilabel_stratified_kfold/data/
│       ├── stratified_kfold/data/
│       └── group_kfold/data/
├── quality_high_medium/
│   └── ...
└── quality_high_medium_low/
    └── ...
```

Each `data/` directory contains:

```text
data/
├── split_0/
│   ├── fold_0/
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── fold_1/
│   │   └── ...
│   └── ...
├── split_1/
│   └── ...
└── ...
```
