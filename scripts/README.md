# Scripts Directory

This directory contains organized scripts for training, data processing, analysis, and infrastructure management.

## Directory Structure

```text
scripts/
├── training/          # Model training scripts
├── data/              # Data processing and splitting scripts
├── analysis/          # Analysis and visualization tools
├── infra/             # Infrastructure (MLflow, docs, etc.)
└── lib/               # Shared library functions
```

## Training Scripts (`training/`)

### `train_chemprop_ensembles.sh`

Trains Chemprop ensemble models across multiple data splits and folds using Ray parallelization.

**Usage:**

```bash
./scripts/training/train_chemprop_ensembles.sh [--dry-run] [--max-parallel N] [--log-level LEVEL]
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
./scripts/training/train_chemprop_model.sh [--dry-run] [--log-level LEVEL] [--splits N] [--folds N]
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

### `train_chemprop_hpo.sh`

Runs hyperparameter optimization (HPO) for Chemprop models using Ray Tune with ASHA scheduler.

**Usage:**

```bash
./scripts/training/train_chemprop_hpo.sh [--config PATH] [--num-samples N] [--gpus-per-trial N] \
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
./scripts/training/train_chemprop_hpo.sh --num-samples 50

# Run HPO with custom config and output directory
./scripts/training/train_chemprop_hpo.sh --config configs/my_hpo.yaml --output-dir my_hpo_results/
```

### `train_chemprop_hpo_ensembles.sh`

Trains ensemble models using the top 100 HPO configurations in rank order.

**Usage:**

```bash
./scripts/training/train_chemprop_hpo_ensembles.sh [--start N] [--end N] [--ranks N,N,N] [--max-parallel N]
```

**Options:**

- `--start N` - Starting rank (default: 1)
- `--end N` - Ending rank (default: 100)
- `--ranks N,N,N` - Specific ranks to train (comma-separated)
- `--max-parallel N` - Maximum parallel models (default: 4)
- `--dry-run` - Print commands without executing

### `generate_ensemble_configs.py`

Python script to generate ensemble configuration files from HPO results.

**Usage:**

```bash
python scripts/training/generate_ensemble_configs.py
```

### `test_integration.py`

Integration tests for task affinity with ChempropModel.

**Usage:**

```bash
python scripts/training/test_integration.py
```

### `train_production_ensembles.sh`

Train all production ensemble models from `configs/3-production/`. These are the final selected configurations for production deployment.

**Usage:**

```bash
# Train all production configs
./scripts/training/train_production_ensembles.sh

# Train specific config
./scripts/training/train_production_ensembles.sh --config ensemble_chemprop_hpo_001.yaml

# Continue from a specific config (useful after failures)
./scripts/training/train_production_ensembles.sh --continue-from 19

# With custom parallelization
./scripts/training/train_production_ensembles.sh --max-parallel 2

# Dry run to see what would be executed
./scripts/training/train_production_ensembles.sh --dry-run
```

## Data Processing Scripts (`data/`)

### `run_data_splits.sh`

Runs data splitting across multiple configurations of split methods, clustering methods, and quality filters.

**Usage:**

```bash
./scripts/data/run_data_splits.sh --input data.csv --output-dir outputs/
./scripts/data/run_data_splits.sh -i data.csv -o outputs/ --cluster-methods bitbirch scaffold
./scripts/data/run_data_splits.sh -i data.csv -o outputs/ --qualities "high" "high,medium"
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
./scripts/data/run_data_splits.sh -i data.csv -o outputs/splits/

# Run specific methods only
./scripts/data/run_data_splits.sh -i data.csv -s multilabel_stratified_kfold -c bitbirch scaffold

# Run with specific quality filters
./scripts/data/run_data_splits.sh -i data.csv -q "high" "high,medium" "high,medium,low"

# Dry run to preview commands
./scripts/data/run_data_splits.sh -i data.csv --dry-run
```

### `create_dataset_splits.py`

Python script for creating dataset splits with fingerprints and multiple splitting strategies.
Originally a Jupyter notebook (`1_dataset_splits.ipynb`), converted to a standalone script.

**Usage:**

```bash
python scripts/data/create_dataset_splits.py
```

**Features:**

- Loads high/medium/low quality datasets
- Calculates Morgan fingerprints
- Creates temporal and k-fold splits
- Saves datasets in HuggingFace format
- Generates visualizations

## Analysis Scripts (`analysis/`)

### `compute_task_affinity.py`

Computes task affinity using gradient cosine approach and saves artifacts.

**Usage:**

```bash
python scripts/analysis/compute_task_affinity.py data.csv \
  --smiles SMILES \
  --targets "LogD,KSOL,CLint" \
  --outdir output/ \
  --n_groups 3 \
  --save_plots
```

**Options:**

- `--smiles` - SMILES column name (default: SMILES)
- `--targets` - Comma-separated target column names (required)
- `--outdir` - Output directory for artifacts (default: .)
- `--n_groups` - Number of task groups (default: 3)
- `--epochs` - Affinity computation epochs (default: 1)
- `--batch_size` - Batch size (default: 64)
- `--save_plots` - Save heatmap and clustermap visualizations

**Outputs:**

- `affinity_matrix.csv` - Task affinity matrix
- `affinity_heatmap.png` - Heatmap visualization (if --save_plots)
- `affinity_clustermap.png` - Hierarchical clustering visualization (if --save_plots)

### `calculate_weights.py`

Calculates target weights based on sample counts for handling class imbalance.

**Usage:**

```bash
python scripts/analysis/calculate_weights.py
```

**Features:**

- Computes linear, clipped, and sqrt weights
- Recommends clipped (10.0) weights for stability
- Outputs weights in config-ready format

### `select_diverse_configs.py`

Analyzes HPO results and selects diverse high-performing configurations.

**Usage:**

```bash
python scripts/analysis/select_diverse_configs.py
```

**Features:**

- Performance distribution analysis
- Hyperparameter correlation analysis
- PCA and clustering for diversity selection
- Generates comprehensive visualizations
- Outputs top configurations to YAML

## Infrastructure Scripts (`infra/`)

### `mlflow_server.sh`

Starts a local MLflow tracking server for experiment logging.

**Usage:**

```bash
./scripts/infra/mlflow_server.sh
```

**Details:**

- Starts MLflow server on `http://127.0.0.1:8080`
- Run this before training to enable experiment tracking
- Access the MLflow UI at <http://127.0.0.1:8080>

### `setup_mlflow_postgres.sh`

Sets up MLflow with PostgreSQL backend using Docker.

**Usage:**

```bash
./scripts/infra/setup_mlflow_postgres.sh [start|stop|restart|status|logs]
```

**Commands:**

- `start` - Start PostgreSQL and MLflow server (default)
- `stop` - Stop both services
- `restart` - Restart both services
- `status` - Show service status
- `logs [postgres|mlflow|all]` - Show service logs

**Configuration:**

See `scripts/infra/README_mlflow_postgres.md` for detailed configuration options.

### `rebuild-docs-precommit.sh`

Pre-commit hook script that rebuilds Sphinx documentation.

**Usage:**

This script is called automatically by pre-commit hooks. Do not run manually.

**Features:**

- Builds Sphinx documentation
- Stages new documentation files for commit
- Does not fail the commit if docs build fails

## Shared Libraries (`lib/`)

### `common.sh`

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
