==========================
Shell Scripts Reference
==========================

Comprehensive reference for all shell scripts in the ``scripts/`` directory for training, data processing, analysis, and infrastructure management.

Directory Structure
===================

.. code-block:: text

   scripts/
   ├── training/          # Model training scripts
   ├── hpo/               # Hyperparameter optimization
   ├── data/              # Data processing and splitting
   ├── analysis/          # Analysis and visualization
   ├── mlflow/            # MLflow server and cleanup
   ├── infra/             # Infrastructure utilities
   └── lib/               # Shared library functions

Training Scripts
================

train_chemprop_ensembles.sh
----------------------------

Trains Chemprop ensemble models across multiple data splits and folds using Ray parallelization.

**Usage:**

.. code-block:: bash

   ./scripts/training/train_chemprop_ensembles.sh [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--dry-run``
     - Print commands without executing
   * - ``--max-parallel N``
     - Override maximum parallel models (default: from config)
   * - ``--log-level LEVEL``
     - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

- Iterates through all configured data directories
- Creates temporary configs with updated ``data_dir`` paths
- Tracks success/failure/skipped counts
- Logs results to MLflow

train_chemprop_model.sh
------------------------

Trains individual Chemprop models for each split/fold combination sequentially.

**Usage:**

.. code-block:: bash

   ./scripts/training/train_chemprop_model.sh [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--dry-run``
     - Print commands without executing
   * - ``--log-level LEVEL``
     - Set logging level (DEBUG, INFO, WARNING, ERROR)
   * - ``--splits N``
     - Limit number of splits to train (default: all)
   * - ``--folds N``
     - Limit number of folds per split (default: all)

**Features:**

- Discovers split/fold directory structure automatically
- Trains models one at a time without Ray parallelization
- Useful for debugging or resource-constrained environments

train_chemprop_hpo.sh
---------------------

Runs hyperparameter optimization (HPO) for Chemprop models using Ray Tune with ASHA scheduler.

**Usage:**

.. code-block:: bash

   ./scripts/training/train_chemprop_hpo.sh [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--config PATH``
     - Path to HPO config file (default: configs/1-hpo-single-fold/hpo_chemprop.yaml)
   * - ``--num-samples N``
     - Number of HPO trials to run (default: from config)
   * - ``--gpus-per-trial N``
     - GPUs per trial (default: 0.25)
   * - ``--cpus-per-trial N``
     - CPUs per trial (default: 2)
   * - ``--output-dir PATH``
     - Output directory for results (default: hpo_results/)
   * - ``--log-level LEVEL``
     - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

- ASHA (Asynchronous Successive Halving Algorithm) early stopping
- Conditional search spaces for FFN architectures (MoE, branched, standard)
- MLflow integration for experiment tracking
- GPU resource management for parallel trials
- Outputs ``top_k_configs.json`` with best hyperparameter configurations
- Transfer learning support from pretrained checkpoints

**Example:**

.. code-block:: bash

   # Run HPO with 50 trials
   ./scripts/training/train_chemprop_hpo.sh --num-samples 50

   # Custom config and output directory
   ./scripts/training/train_chemprop_hpo.sh \
       --config configs/my_hpo.yaml \
       --output-dir my_hpo_results/

train_chemprop_hpo_ensembles.sh
--------------------------------

Trains ensemble models using the top 100 HPO configurations in rank order.

**Usage:**

.. code-block:: bash

   ./scripts/training/train_chemprop_hpo_ensembles.sh [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--start N``
     - Starting rank (default: 1)
   * - ``--end N``
     - Ending rank (default: 100)
   * - ``--ranks N,N,N``
     - Specific ranks to train (comma-separated)
   * - ``--max-parallel N``
     - Maximum parallel models (default: 4)
   * - ``--dry-run``
     - Print commands without executing

train_production_ensembles.sh
------------------------------

Train all production ensemble models from ``configs/3-hpo-ensemble-production/``. These are the final selected configurations for production deployment.

**Usage:**

.. code-block:: bash

   # Train all production configs
   ./scripts/training/train_production_ensembles.sh

   # Train specific config
   ./scripts/training/train_production_ensembles.sh \
       --config ensemble_chemprop_hpo_001.yaml

   # Continue from a specific config (useful after failures)
   ./scripts/training/train_production_ensembles.sh --continue-from 19

   # With custom parallelization
   ./scripts/training/train_production_ensembles.sh --max-parallel 2

   # Dry run to see what would be executed
   ./scripts/training/train_production_ensembles.sh --dry-run

Hyperparameter Optimization Scripts
====================================

generate_ensemble_configs.py
-----------------------------

Python script to generate ensemble configuration files from HPO results.

**Usage:**

.. code-block:: bash

   python scripts/hpo/generate_ensemble_configs.py

**Features:**

- Extracts top-performing configurations from HPO results
- Generates production-ready ensemble config files
- Outputs to ``configs/3-hpo-ensemble-production/`` directory

generate_chemeleon_ensemble_configs.py
---------------------------------------

Python script to generate Chemeleon ensemble configuration files from HPO results.

**Usage:**

.. code-block:: bash

   python scripts/hpo/generate_chemeleon_ensemble_configs.py

**Features:**

- Specialized for Chemeleon model ensemble generation
- Extracts best configurations from Chemeleon HPO runs
- Creates ensemble configs in standardized format

select_diverse_configs.py
--------------------------

Analyzes HPO results and selects diverse high-performing configurations.

**Usage:**

.. code-block:: bash

   python scripts/hpo/select_diverse_configs.py

**Features:**

- Performance distribution analysis
- Hyperparameter correlation analysis
- PCA and clustering for diversity selection
- Generates comprehensive visualizations
- Outputs top configurations to YAML

Data Processing Scripts
=======================

run_data_splits.sh
------------------

Runs data splitting across multiple configurations of split methods, clustering methods, and quality filters.

**Usage:**

.. code-block:: bash

   ./scripts/data/run_data_splits.sh --input data.csv --output-dir outputs/
   ./scripts/data/run_data_splits.sh -i data.csv -o outputs/ \
       --cluster-methods bitbirch scaffold
   ./scripts/data/run_data_splits.sh -i data.csv -o outputs/ \
       --qualities "high" "high,medium"

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-i, --input FILE``
     - Input CSV file path (required)
   * - ``-o, --output-dir DIR``
     - Output directory (default: assets/dataset/split_train_val)
   * - ``--smiles-col COL``
     - SMILES column name (default: SMILES)
   * - ``--quality-col COL``
     - Quality column name (default: Quality)
   * - ``-s, --split-methods M...``
     - Split methods to use (space-separated)
   * - ``-c, --cluster-methods M...``
     - Clustering methods to use (space-separated)
   * - ``-q, --qualities Q...``
     - Quality filter combinations (comma-delimited, space-separated)
   * - ``-t, --target-cols T...``
     - Target columns for stratification
   * - ``--dry-run``
     - Print commands without executing
   * - ``--log-level LEVEL``
     - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Available Split Methods:**

- ``group_kfold`` - Splits by clusters without stratification
- ``stratified_kfold`` - Stratifies on single-label cluster presence vectors
- ``multilabel_stratified_kfold`` - Stratifies on multi-label cluster presence vectors

**Available Clustering Methods:**

- ``random`` - Random cluster assignment
- ``scaffold`` - Bemis-Murcko scaffold-based clustering
- ``kmeans`` - K-means clustering on fingerprints
- ``umap`` - UMAP dimensionality reduction + clustering
- ``butina`` - Butina clustering with Tanimoto similarity
- ``bitbirch`` - BitBirch hierarchical clustering (recommended)

**Example:**

.. code-block:: bash

   # Run all configurations
   ./scripts/data/run_data_splits.sh -i data.csv -o outputs/splits/

   # Run specific methods only
   ./scripts/data/run_data_splits.sh -i data.csv \
       -s multilabel_stratified_kfold \
       -c bitbirch scaffold

   # Run with specific quality filters
   ./scripts/data/run_data_splits.sh -i data.csv \
       -q "high" "high,medium" "high,medium,low"

   # Dry run to preview commands
   ./scripts/data/run_data_splits.sh -i data.csv --dry-run

create_dataset_splits.py
-------------------------

Python script for creating dataset splits with fingerprints and multiple splitting strategies. Originally a Jupyter notebook (``1_dataset_splits.ipynb``), converted to a standalone script.

**Usage:**

.. code-block:: bash

   python scripts/data/create_dataset_splits.py

**Features:**

- Loads high/medium/low quality datasets
- Calculates Morgan fingerprints
- Creates temporal and k-fold splits
- Saves datasets in HuggingFace format
- Generates visualizations

Analysis Scripts
================

compute_task_affinity.py
-------------------------

Computes task affinity using gradient cosine approach and saves artifacts.

**Usage:**

.. code-block:: bash

   python scripts/analysis/compute_task_affinity.py data.csv \
       --smiles SMILES \
       --targets "LogD,KSOL,CLint" \
       --outdir output/ \
       --n_groups 3 \
       --save_plots

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--smiles``
     - SMILES column name (default: SMILES)
   * - ``--targets``
     - Comma-separated target column names (required)
   * - ``--outdir``
     - Output directory for artifacts (default: .)
   * - ``--n_groups``
     - Number of task groups (default: 3)
   * - ``--epochs``
     - Affinity computation epochs (default: 1)
   * - ``--batch_size``
     - Batch size (default: 64)
   * - ``--save_plots``
     - Save heatmap and clustermap visualizations

**Outputs:**

- ``affinity_matrix.csv`` - Task affinity matrix
- ``affinity_heatmap.png`` - Heatmap visualization (if --save_plots)
- ``affinity_clustermap.png`` - Hierarchical clustering visualization (if --save_plots)

calculate_weights.py
--------------------

Calculates target weights based on sample counts for handling class imbalance.

**Usage:**

.. code-block:: bash

   python scripts/analysis/calculate_weights.py

**Features:**

- Computes linear, clipped, and sqrt weights
- Recommends clipped (10.0) weights for stability
- Outputs weights in config-ready format

MLflow Scripts
==============

mlflow_server.sh
----------------

Starts a local MLflow tracking server for experiment logging.

**Usage:**

.. code-block:: bash

   ./scripts/mlflow/mlflow_server.sh

**Details:**

- Starts MLflow server on ``http://127.0.0.1:8080``
- Run this before training to enable experiment tracking
- Access the MLflow UI at http://127.0.0.1:8080

setup_mlflow_postgres.sh
------------------------

Sets up MLflow with PostgreSQL backend using Docker.

**Usage:**

.. code-block:: bash

   ./scripts/mlflow/setup_mlflow_postgres.sh [COMMAND]

**Commands:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``start``
     - Start PostgreSQL and MLflow server (default)
   * - ``stop``
     - Stop both services
   * - ``restart``
     - Restart both services
   * - ``status``
     - Show service status
   * - ``logs [postgres|mlflow|all]``
     - Show service logs

**Configuration:**

See ``scripts/mlflow/README_mlflow_postgres.md`` for detailed configuration options.

mlflow_cleanup.py
-----------------

Cleans up old MLflow experiments and runs.

**Usage:**

.. code-block:: bash

   python scripts/mlflow/mlflow_cleanup.py

**Features:**

- Removes old or failed experiment runs
- Optionally archives experiments before deletion
- Helps manage MLflow storage usage

cleanup_storage.py
------------------

Storage cleanup utility for MLflow artifacts and database.

**Usage:**

.. code-block:: bash

   python scripts/mlflow/cleanup_storage.py

**Features:**

- Removes orphaned artifact files
- Cleans up old checkpoints and logs
- Frees up disk space

mlflow_experiment_info.py
-------------------------

Displays information about MLflow experiments and runs.

**Usage:**

.. code-block:: bash

   python scripts/mlflow/mlflow_experiment_info.py

**Features:**

- Lists all experiments and their run counts
- Shows experiment metrics and parameters
- Useful for monitoring training progress

Shared Libraries
================

common.sh
---------

Shared library containing common functions and variables used by training scripts.

**Features:**

- Data directory definitions
- Logging functions (info, warn, error, cmd, section)
- Configuration helpers (create/cleanup temp configs)
- Validation helpers (check files and directories)
- Directory discovery (find splits and folds)
- Summary reporting

**Usage in scripts:**

.. code-block:: bash

   source "$SCRIPT_DIR/lib/common.sh"

Configuration Files
===================

Training scripts use configuration files from the ``configs/`` directory:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Configuration
     - Purpose
   * - ``configs/0-experiment/0-single-fold/chemprop.yaml``
     - Single model training configuration
   * - ``configs/0-experiment/1-ensemble/ensemble_chemprop_production.yaml``
     - Ensemble training configuration
   * - ``configs/1-hpo-single-fold/hpo_chemprop.yaml``
     - Hyperparameter optimization configuration
   * - ``configs/3-hpo-ensemble-production/``
     - Production ensemble configurations

Expected Data Directory Structure
==================================

Training scripts expect data in the following structure:

.. code-block:: text

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

Each ``data/`` directory contains:

.. code-block:: text

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

See Also
========

- :doc:`/guide/cli` - Python CLI commands
- :doc:`/guide/modeling` - Model training workflows
- :doc:`/guide/hpo` - Hyperparameter optimization guide
- :doc:`/guide/configuration` - Configuration system reference
