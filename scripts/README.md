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

### `run_chemprop_hpo.sh`

Runs hyperparameter optimization (HPO) for Chemprop models using Ray Tune with ASHA scheduler.

**Usage:**

```bash
./scripts/run_chemprop_hpo.sh [--config PATH] [--num-samples N] [--gpus-per-trial N] \
                              [--cpus-per-trial N] [--output-dir PATH] [--log-level LEVEL]
```

**Options:**

- `--config PATH` - Path to HPO config file (default: `configs/hpo_chemprop.yaml`)
- `--num-samples N` - Number of HPO trials to run (default: from config)
- `--gpus-per-trial N` - GPUs per trial (default: 0.25)
- `--cpus-per-trial N` - CPUs per trial (default: 2)
- `--output-dir PATH` - Output directory for results (default: `hpo_results/`)
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

- ASHA (Asynchronous Successive Halving Algorithm) early stopping
- Conditional search spaces for FFN architectures (MoE, branched, standard)
- MLflow integration for experiment tracking
- GPU resource management for parallel trials
- Outputs `top_k_configs.json` with best hyperparameter configurations
- Transfer learning support from pretrained checkpoints

**Example:**

```bash
# Run HPO with 50 trials using default config
./scripts/run_chemprop_hpo.sh --num-samples 50

# Run HPO with custom config and output directory
./scripts/run_chemprop_hpo.sh --config configs/my_hpo.yaml --output-dir my_hpo_results/
```

## Data Processing Scripts

### `run_data_splits.sh`

Runs data splitting across multiple configurations of split methods, clustering methods, and quality filters.

**Usage:**

```bash
./scripts/run_data_splits.sh --input data.csv --output-dir outputs/
./scripts/run_data_splits.sh -i data.csv -o outputs/ --cluster-methods bitbirch scaffold
./scripts/run_data_splits.sh -i data.csv -o outputs/ --qualities "high" "high,medium"
```

**Options:**

- `-i, --input FILE` - Input CSV file path (required)
- `-o, --output-dir DIR` - Output directory (default: `assets/dataset/split_train_val`)
- `--smiles-col COL` - SMILES column name (default: `SMILES`)
- `--quality-col COL` - Quality column name (default: `Quality`)
- `-s, --split-methods M...` - Split methods to use (space-separated)
- `-c, --cluster-methods M...` - Clustering methods to use (space-separated)
- `-q, --qualities Q...` - Quality filter combinations (comma-delimited, space-separated)
- `-t, --target-cols T...` - Target columns for stratification
- `--dry-run` - Print commands without executing
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Available Split Methods:**

- `group_kfold` - Splits by clusters without stratification
- `stratified_kfold` - Stratifies on single-label cluster presence vectors
- `multilabel_stratified_kfold` - Stratifies on multi-label cluster presence vectors

**Available Clustering Methods:**

- `random` - Random cluster assignment
- `scaffold` - Bemis-Murcko scaffold-based clustering
- `kmeans` - K-means clustering on fingerprints
- `umap` - UMAP dimensionality reduction + clustering
- `butina` - Butina clustering with Tanimoto similarity
- `bitbirch` - BitBirch hierarchical clustering (recommended)

**Example:**

```bash
# Run all configurations (all methods × all clusters × all quality combos)
./scripts/run_data_splits.sh -i data.csv -o outputs/splits/

# Run specific methods only
./scripts/run_data_splits.sh -i data.csv -s multilabel_stratified_kfold -c bitbirch scaffold

# Run with specific quality filters
./scripts/run_data_splits.sh -i data.csv -q "high" "high,medium" "high,medium,low"

# Dry run to preview commands
./scripts/run_data_splits.sh -i data.csv --dry-run
```

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
- `configs/hpo_chemprop.yaml` - Hyperparameter optimization configuration

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
