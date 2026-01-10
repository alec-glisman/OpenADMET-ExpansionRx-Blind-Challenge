======================
CLI Command Reference
======================

The ``admet`` command-line interface provides commands for model training, data processing, and leaderboard analysis.

Installation and Setup
======================

After installing the package, the ``admet`` command is available:

.. code-block:: bash

   # Verify installation
   admet --help

   # Show version
   admet --version

Available top-level commands: ``model``, ``data``, ``leaderboard``

Model Commands
==============

The ``admet model`` command group handles model training, hyperparameter optimization, and ensemble workflows.

admet model train
-----------------

Train a single model or ensemble using a configuration file.

**Usage:**

.. code-block:: bash

   admet model train --config CONFIG_PATH

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to YAML configuration file (required)

**Examples:**

.. code-block:: bash

   # Train single model
   admet model train -c configs/0-experiment/0-single-fold/chemprop.yaml

   # Train with custom config
   admet model train -c my_config.yaml

**Configuration File Requirements:**

- Must specify ``model.type`` (chemprop, chemeleon, xgboost, lightgbm, catboost)
- Must specify ``data.data_dir`` or ``data.path``
- Must specify ``data.target_cols``

See :doc:`/guide/configuration` for complete config reference.

admet model ensemble
--------------------

Train ensemble models across multiple data splits and folds using Ray parallelization.

**Usage:**

.. code-block:: bash

   admet model ensemble --config CONFIG_PATH [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to ensemble configuration file (required)
   * - ``--max-parallel N``
     - Maximum number of parallel models (default: from config)

**Examples:**

.. code-block:: bash

   # Train ensemble with default parallelization
   admet model ensemble -c configs/0-experiment/1-ensemble/ensemble_chemprop_production.yaml

   # Limit parallel models
   admet model ensemble -c configs/3-hpo-ensemble-production/0_chemprop_v1/ensemble_chemprop_hpo_001.yaml --max-parallel 4

**Configuration Requirements:**

- Must specify ``ensemble.enabled: true``
- Must specify ``ensemble.data_dirs`` (list of split/fold directories)
- Ray will parallelize training across available resources

See :doc:`/guide/modeling` for ensemble training workflows.

admet model hpo
---------------

Run hyperparameter optimization using Ray Tune with ASHA scheduler.

**Usage:**

.. code-block:: bash

   admet model hpo --config CONFIG_PATH [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to HPO configuration file (required)
   * - ``--num-samples N``
     - Number of trials to run (overrides config)
   * - ``--gpus-per-trial N``
     - GPUs per trial (default: 0.25)
   * - ``--cpus-per-trial N``
     - CPUs per trial (default: 2)

**Examples:**

.. code-block:: bash

   # Run HPO with 50 trials
   admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50

   # Custom resource allocation
   admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --gpus-per-trial 0.5 --cpus-per-trial 4

**Configuration Requirements:**

- Must specify ``hpo.enabled: true``
- Must define ``hpo.search_space`` (hyperparameter ranges)
- Supports warmstart from previous Optuna studies

**Outputs:**

- Results logged to MLflow
- Top configurations saved to ``hpo_results/top_k_configs.json``

See :doc:`/guide/hpo` for complete HPO guide.

admet model hpo-list-studies
-----------------------------

List available Optuna studies for warmstart configuration.

**Usage:**

.. code-block:: bash

   admet model hpo-list-studies [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--verbose``
     - Show detailed study information

**Example:**

.. code-block:: bash

   # List all studies
   admet model hpo-list-studies

   # Show detailed info
   admet model hpo-list-studies --verbose

**Output Format:**

.. code-block:: text

   Available Optuna Studies:
   1. chemprop_hpo_2024 (125 trials, best_value=0.456)
   2. chemeleon_hpo_2024 (87 trials, best_value=0.523)

See the Warmstarting section in :doc:`/guide/hpo` for using studies to warmstart HPO.

admet model list
----------------

List all registered model types.

**Usage:**

.. code-block:: bash

   admet model list

**Example Output:**

.. code-block:: text

   Available model types:
   - chemprop: Graph neural network (MPNN)
   - chemeleon: Foundation model with pre-trained encoder
   - xgboost: XGBoost gradient boosting
   - lightgbm: LightGBM gradient boosting
   - catboost: CatBoost gradient boosting

admet model train-chemprop (deprecated)
----------------------------------------

Legacy command for Chemprop training. Use ``admet model train`` instead.

Data Commands
=============

The ``admet data`` command group handles data processing and splitting.

admet data split
----------------

Generate cluster-aware train/validation splits with multiple splitting strategies.

**Usage:**

.. code-block:: bash

   admet data split INPUT_FILE [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``INPUT_FILE``
     - Path to input CSV file (required, positional)
   * - ``-o, --output DIR``
     - Output directory (default: assets/dataset/split_train_val)
   * - ``--smiles-col COL``
     - SMILES column name (default: SMILES)
   * - ``--quality-col COL``
     - Quality column name (default: Quality)
   * - ``-s, --split-method METHOD``
     - Split method (default: multilabel_stratified_kfold)
   * - ``-c, --cluster-method METHOD``
     - Clustering method (default: bitbirch)
   * - ``--n-splits N``
     - Number of splits (default: 5)
   * - ``--n-folds N``
     - Number of folds per split (default: 5)
   * - ``--qualities Q...``
     - Quality filters to apply (space-separated)

**Available Split Methods:**

- ``multilabel_stratified_kfold`` - Multi-label stratification (recommended)
- ``stratified_kfold`` - Single-label stratification
- ``group_kfold`` - Group-based splitting without stratification

**Available Clustering Methods:**

- ``bitbirch`` - BitBirch hierarchical clustering (recommended)
- ``scaffold`` - Bemis-Murcko scaffold-based
- ``kmeans`` - K-means on fingerprints
- ``umap`` - UMAP + clustering
- ``butina`` - Butina clustering with Tanimoto similarity
- ``random`` - Random assignment

**Examples:**

.. code-block:: bash

   # Basic usage with defaults
   admet data split data/admet_train.csv

   # Custom output and clustering
   admet data split data/admet_train.csv -o outputs/ -c bitbirch -s multilabel_stratified_kfold

   # Custom splits and folds
   admet data split data/admet_train.csv --n-splits 3 --n-folds 5

   # Filter by quality
   admet data split data/admet_train.csv --qualities high high,medium

**Output Structure:**

.. code-block:: text

   output_dir/
   └── quality_high/
       └── bitbirch/
           └── multilabel_stratified_kfold/data/
               ├── split_0/
               │   ├── fold_0/
               │   │   ├── train.csv
               │   │   └── validation.csv
               │   └── fold_1/...
               └── split_1/...

See :doc:`/guide/splitting` for detailed splitting strategies.

Leaderboard Commands
====================

The ``admet leaderboard`` command group handles competition leaderboard analysis.

admet leaderboard scrape
-------------------------

Scrape leaderboard data from HuggingFace Spaces and generate analysis reports.

**Usage:**

.. code-block:: bash

   admet leaderboard scrape --user USERNAME [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-u, --user USERNAME``
     - HuggingFace username to track (required)
   * - ``-o, --output DIR``
     - Output directory (default: assets/submissions/YYYY-MM-DD)
   * - ``--space NAME``
     - HuggingFace Space name (default: auto-detect)
   * - ``--no-plots``
     - Skip plot generation
   * - ``--verbose``
     - Detailed logging

**Examples:**

.. code-block:: bash

   # Basic scrape
   admet leaderboard scrape --user your_username

   # Custom output directory
   admet leaderboard scrape --user your_username --output ./results

   # Skip plots for faster execution
   admet leaderboard scrape --user your_username --no-plots

   # Different HuggingFace Space
   admet leaderboard scrape --user your_username --space owner/space-name

**Outputs:**

- ``leaderboard_full.csv`` - Complete leaderboard data
- ``user_submissions.csv`` - User's submissions only
- ``analysis_report.md`` - Markdown report with rankings
- ``plots/`` - Visualizations (if --no-plots not set)

admet leaderboard report
-------------------------

Generate analysis report from existing leaderboard data.

**Usage:**

.. code-block:: bash

   admet leaderboard report LEADERBOARD_FILE [OPTIONS]

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``LEADERBOARD_FILE``
     - Path to leaderboard CSV (required, positional)
   * - ``-u, --user USERNAME``
     - Username to highlight
   * - ``-o, --output FILE``
     - Output report file (default: analysis_report.md)

**Example:**

.. code-block:: bash

   # Generate report
   admet leaderboard report leaderboard_full.csv --user your_username

See :doc:`/guide/leaderboard` for complete leaderboard analysis guide.

Programmatic Usage
==================

All CLI commands can also be invoked programmatically:

**Using Typer CliRunner:**

.. code-block:: python

   from typer.testing import CliRunner
   from admet.cli import app as main_app

   runner = CliRunner()
   result = runner.invoke(main_app, ["model", "train", "-c", "config.yaml"])
   assert result.exit_code == 0

**Direct Module Access:**

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.registry import ModelRegistry
   from admet.model.config import UnifiedModelConfig

   # Load configuration
   config = OmegaConf.merge(
       OmegaConf.structured(UnifiedModelConfig),
       OmegaConf.load("configs/0-experiment/0-single-fold/chemprop.yaml")
   )

   # Train model
   model = ModelRegistry.create(config)
   model.fit()

Configuration Files
===================

Training scripts use configuration files from the ``configs/`` directory:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Configuration
     - Purpose
   * - ``configs/0-experiment/0-single-fold/chemprop.yaml``
     - Single model training
   * - ``configs/0-experiment/1-ensemble/ensemble_chemprop_production.yaml``
     - Ensemble training
   * - ``configs/1-hpo-single-fold/hpo_chemprop.yaml``
     - Hyperparameter optimization
   * - ``configs/3-hpo-ensemble-production/``
     - Production ensemble configs

See :doc:`/guide/configuration` for complete configuration reference.

MLflow Integration
==================

All training runs automatically log to MLflow. Configure in YAML:

.. code-block:: yaml

   mlflow:
     tracking: true
     tracking_uri: "http://127.0.0.1:8084"
     experiment_name: "chemprop_admet"

Start the MLflow server:

.. code-block:: bash

   mlflow server --host 127.0.0.1 --port 8084

See :doc:`/guide/mlflow_artifacts` for MLflow workflow details.

Testing the CLI
===============

When writing unit tests, prefer invoking the top-level app (``admet``) so subcommand
parsing behaves the same as when the CLI is installed as a console script. Use
Typer's ``CliRunner`` to exercise commands in tests:

.. code-block:: python

   from typer.testing import CliRunner
   from admet.cli import app as main_app

   runner = CliRunner()
   result = runner.invoke(main_app, ["data", "split", "--output", "./out", "data.csv"])
   assert result.exit_code == 0

Avoid invoking sub-``Typer`` instances (e.g. ``data_app``) directly when
testing command-line parsing; doing so can lead to different argument parsing
behavior and unexpected errors.

See Also
========

- :doc:`/getting-started/quickstart` - 5-minute training tutorial
- :doc:`/reference/scripts` - Shell scripts reference
- :doc:`/guide/modeling` - Detailed training workflows
- :doc:`/guide/configuration` - Configuration file structure
- :doc:`/guide/hpo` - Hyperparameter optimization guide
- :doc:`/api/admet` - Python API reference
