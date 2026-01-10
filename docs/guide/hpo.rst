Hyperparameter Optimization Guide
==================================

Hyperparameter optimization systematically explores model configurations using
Ray Tune and the ASHA scheduler. Efficient search with early stopping identifies
optimal settings for depth, hidden dimensions, dropout, and learning rates.

HPO Workflow
------------

The HPO process consists of trial execution, early stopping, and config selection:

.. mermaid::

   flowchart LR
      subgraph Configure
         A[Define Search Space] --> B[Set ASHA Params]
      end

      subgraph "Run Trials"
         C[Sample Config] --> D[Train Model]
         D --> E{Rung Check}
         E -->|Underperforms| F[Early Stop]
         E -->|Promising| D
      end

      subgraph "Select Results"
         G[Rank by Val MAE] --> H[Select Top K]
         H --> I[Generate Configs]
      end

      B --> C
      D --> G
      F --> G

      style A fill:#e1f5fe
      style I fill:#c8e6c9
      style F fill:#ffcdd2

**Workflow Steps:**

1. **Configure**: Define search space for hyperparameters in YAML
2. **Run Trials**: Ray Tune explores configurations with ASHA scheduler
3. **Early Stopping**: ASHA halts poor-performing trials to save resources
4. **Rank**: Sort trials by validation MAE (lower is better)
5. **Select Top K**: Extract best N configurations (typically 10-100)
6. **Generate Configs**: Create ensemble configs from top performers

Overview
--------

The HPO system provides:

- **Ray Tune Integration**: Distributed hyperparameter search with efficient scheduling
- **ASHA Scheduler**: Asynchronous successive halving for early stopping of poor trials
- **MLflow Tracking**: Automatic logging of trial results and best configurations
- **Checkpoint Recovery**: Trial checkpoints for fault tolerance
- **Comprehensive Metrics**: Validation MAE, RMSE, R², Pearson r, Spearman ρ

Quick Start
-----------

Run HPO using the CLI:

.. code-block:: bash

   admet model hpo --config configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50

Or programmatically:

.. code-block:: python

   from admet.model.chemprop.hpo import ChempropHPO
   from omegaconf import OmegaConf

   config = OmegaConf.load("configs/1-hpo-single-fold/hpo_chemprop.yaml")
   hpo = ChempropHPO(config)
   best_config, results = hpo.run()

   print(f"Best validation MAE: {results.best_result['val_mae']:.4f}")

Configuration
-------------

HPO is configured through a YAML file with the following sections:

Data Configuration
^^^^^^^^^^^^^^^^^^

Specify training and validation data paths:

.. code-block:: yaml

   # Data paths
   data_path: assets/dataset/split_train_val/v3/train.csv
   val_data_path: assets/dataset/split_train_val/v3/val.csv

   # Column configuration
   smiles_column: SMILES
   target_columns:
     - "LogD"
     - "Log KSOL"
     - "Log HLM CLint"

Search Space Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Define hyperparameter distributions for optimization:

.. code-block:: yaml

   search_space:
     # Learning rate (log-uniform for wide range)
     learning_rate:
       type: loguniform
       low: 1.0e-5
       high: 1.0e-2

     # Learning rate schedule parameters
     lr_warmup_ratio:
       type: uniform
       low: 0.01
       high: 0.2

     lr_final_ratio:
       type: uniform
       low: 0.01
       high: 0.2

     # Regularization
     dropout:
       type: uniform
       low: 0.0
       high: 0.4

     # Weight decay (L2 regularization via AdamW)
     weight_decay_enabled:
       type: choice
       values: [true, false]

     weight_decay:
       type: loguniform
       low: 1.0e-6
       high: 1.0e-3

     # Message passing architecture
     depth:
       type: choice
       values: [3, 4, 5, 6]

     message_hidden_dim:
       type: choice
       values: [256, 512, 768, 1024]

     # FFN architecture
     ffn_num_layers:
       type: choice
       values: [1, 2, 3]

     ffn_hidden_dim:
       type: choice
       values: [256, 512, 768, 1024]

     # Training
     batch_size:
       type: choice
       values: [32, 64, 128]

Supported distribution types:

- ``uniform``: Uniform distribution between ``low`` and ``high``
- ``loguniform``: Log-uniform distribution (for learning rates)
- ``quniform``: Quantized uniform distribution
- ``choice``: Categorical choice from ``values`` list

Conditional Parameters
^^^^^^^^^^^^^^^^^^^^^^

For architecture-specific parameters, use conditional sampling:

.. code-block:: yaml

   # FFN type selection
   ffn_type:
     type: choice
     values: [mlp, moe, branched]

   # MoE-specific (only sampled when ffn_type=moe)
   n_experts:
     type: choice
     values: [2, 4, 8]
     conditional_on: ffn_type
     conditional_values: [moe]

   # Branched-specific (only sampled when ffn_type=branched)
   trunk_depth:
     type: choice
     values: [1, 2, 3]
     conditional_on: ffn_type
     conditional_values: [branched]

ASHA Scheduler Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the ASHA scheduler for efficient early stopping:

.. code-block:: yaml

   asha:
     # Metric to optimize
     metric: val_mae
     mode: min  # 'min' for loss metrics, 'max' for accuracy

     # Training epochs
     max_t: 100           # Maximum epochs for full training
     grace_period: 15     # Minimum epochs before early stopping

     # Successive halving parameters
     reduction_factor: 3  # Keep top 1/3 of trials at each rung
     brackets: 1          # Number of brackets (1 is standard ASHA)

Key parameters:

- ``max_t``: Maximum training epochs for the best trials (default: 100)
- ``grace_period``: Minimum epochs before a trial can be stopped (default: 15)
- ``reduction_factor``: Fraction of trials promoted at each rung (default: 3)

Search Algorithm Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure advanced search algorithms for Bayesian optimization:

.. code-block:: yaml

   search_algorithm:
     type: optuna        # Options: random, optuna, bayesopt, hyperopt
     seed: 42            # Random seed for reproducibility
     n_initial_points: 20  # Random exploration before Bayesian phase

Supported search algorithms:

- ``random`` (default): Pure random sampling across search space
- ``optuna``: Bayesian optimization with TPE (Tree-structured Parzen Estimator)
- ``bayesopt``: Gaussian Process-based Bayesian optimization
- ``hyperopt``: Tree-structured Parzen Estimator from HyperOpt library

**Bayesian Optimization Benefits:**

- **Adaptive Sampling**: Learns which hyperparameter regions perform well
- **Improved Efficiency**: 3-5x fewer trials to find optimal configurations
- **Exploration vs. Exploitation**: Balances trying new regions vs. refining known good areas

**When to Use:**

- Use ``optuna`` for most cases (default recommendation)
- Use ``random`` for initial broad exploration or when search space is small
- Use ``bayesopt`` or ``hyperopt`` for specific algorithm preferences

**Installation:**

.. code-block:: bash

   # Optuna (recommended)
   pip install optuna

   # BayesOpt
   pip install bayesian-optimization

   # HyperOpt
   pip install hyperopt

Resource Configuration
^^^^^^^^^^^^^^^^^^^^^^

Specify computational resources for HPO:

.. code-block:: yaml

   resources:
     # Number of HPO trials to run
     num_samples: 500

     # Resources per trial
     cpus_per_trial: 4
     gpus_per_trial: 0.25  # 4 concurrent trials per GPU

     # Concurrency limit (null = auto based on resources)
     max_concurrent_trials: null

Per-Target Weight Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-task models, optimize per-endpoint loss weights:

.. code-block:: yaml

   target_weights:
     type: uniform
     low: 0.05
     high: 50.0

This samples a separate weight for each target column, allowing the optimizer
to balance learning across endpoints with different scales and difficulties.

Search Space Parameters
-----------------------

The following hyperparameters are available for optimization:

Learning Rate Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Description
     - Recommended Range
   * - ``learning_rate``
     - Peak learning rate (max_lr)
     - 1e-5 to 1e-2 (loguniform)
   * - ``lr_warmup_ratio``
     - init_lr = max_lr × warmup_ratio
     - 0.01 to 0.2
   * - ``lr_final_ratio``
     - final_lr = max_lr × final_ratio
     - 0.01 to 0.2

The learning rate follows a warmup-plateau-decay schedule:

1. **Warmup**: Linear increase from ``init_lr`` to ``max_lr``
2. **Plateau**: Constant at ``max_lr`` for most of training
3. **Decay**: Cosine decay from ``max_lr`` to ``final_lr``

Architecture Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Description
     - Recommended Values
   * - ``depth``
     - Number of message passing layers
     - 3, 4, 5, 6
   * - ``message_hidden_dim``
     - MPNN hidden layer width
     - 256, 512, 768, 1024
   * - ``ffn_num_layers``
     - Number of FFN layers
     - 1, 2, 3
   * - ``ffn_hidden_dim``
     - FFN hidden layer width
     - 256, 512, 768, 1024
   * - ``dropout``
     - Dropout rate
     - 0.0 to 0.4
   * - ``weight_decay``
     - L2 regularization via AdamW
     - 1e-6 to 1e-3 (loguniform)

**Weight Decay:**

Weight decay provides L2 regularization through the AdamW optimizer, which
implements decoupled weight decay (Loshchilov & Hutter, 2019). This is more
effective than traditional L2 penalty for neural networks.

- Set ``weight_decay: 0.0`` to disable (default)
- Use ``weight_decay_enabled: choice([true, false])`` in HPO to explore on/off
- Typical effective range: 1e-6 to 1e-3 (use loguniform sampling)

FFN Type Parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Description
     - Values
   * - ``ffn_type``
     - Prediction head architecture
     - mlp, moe, branched
   * - ``n_experts``
     - Number of experts (MoE only)
     - 2, 4, 8
   * - ``trunk_depth``
     - Shared trunk layers (branched only)
     - 1, 2, 3
   * - ``trunk_hidden_dim``
     - Trunk hidden width (branched only)
     - 256, 512, 768

Reported Metrics
----------------

Each trial reports comprehensive metrics for analysis:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Description
     - Usage
   * - ``val_mae``
     - Validation Mean Absolute Error
     - Primary optimization metric
   * - ``val_rmse``
     - Validation Root Mean Square Error
     - Alternative error metric
   * - ``val_R2``
     - Validation R² score
     - Explained variance
   * - ``val_pearson_r``
     - Validation Pearson correlation
     - Linear correlation strength
   * - ``val_spearman_rho``
     - Validation Spearman correlation
     - Rank correlation strength
   * - ``train_mae``
     - Training MAE
     - Overfitting detection
   * - ``train_rmse``
     - Training RMSE
     - Overfitting detection
   * - ``train_R2``
     - Training R²
     - Overfitting detection
   * - ``train_loss``
     - Training loss
     - Learning progress
   * - ``epoch``
     - Current epoch
     - Training progress

Checkpoint Recovery
-------------------

Trials automatically save checkpoints for fault tolerance:

.. code-block:: python

   # Checkpoints are saved in the trial directory
   # Ray Tune handles automatic recovery on failure

   # To resume a failed HPO run:
   from ray import tune

   # Restore from previous experiment
   tuner = tune.Tuner.restore(
       path="assets/models/chemprop/hpo/ray_results/experiment_name",
       trainable=train_chemprop_trial,
   )
   results = tuner.fit()

Transfer Learning Workflow
--------------------------

After HPO, use the best configurations for full ensemble training:

.. code-block:: yaml

   transfer_learning:
     top_k: 10           # Number of top configurations to use
     full_epochs: 50     # Full training epochs
     ensemble_size: 5    # Models per configuration

Workflow:

1. Run HPO to find optimal hyperparameters
2. Select top-k configurations based on validation metrics
3. Train full ensemble models using each configuration
4. Combine predictions with uncertainty quantification

.. code-block:: python

   from admet.model.chemprop.hpo import ChempropHPO

   # Run HPO
   hpo = ChempropHPO.from_config(config)
   best_config, results = hpo.run()

   # Get top-k configurations
   top_configs = hpo.get_top_configs(k=10)

   # Train ensemble with best config
   ensemble_config = top_configs[0]
   # ... proceed with ensemble training

Best Practices
--------------

Search Space Design
^^^^^^^^^^^^^^^^^^^

1. **Start broad, then narrow**: Begin with wide ranges and refine based on results
2. **Use log-uniform for learning rates**: Orders of magnitude matter more than linear differences
3. **Consider architecture interactions**: Deeper networks may need different learning rates

Resource Allocation
^^^^^^^^^^^^^^^^^^^

1. **Balance trials vs. epochs**: More short trials often beats fewer long trials
2. **Set appropriate grace_period**: Allow enough epochs for learning rate warmup
3. **Use fractional GPUs**: Run multiple trials per GPU for better utilization

Monitoring
^^^^^^^^^^

1. **Watch for overfitting**: Compare train vs. validation metrics
2. **Check correlation metrics**: R², Pearson r, and Spearman ρ provide different insights
3. **Use MLflow UI**: ``mlflow ui --port 5000`` for interactive analysis

MLflow Artifact Logging
------------------------

All HPO runs automatically log comprehensive artifacts to the **master parent MLflow run**
for complete reproducibility and analysis. This includes Optuna study information,
search space details, and configuration files.

Master Run Structure
^^^^^^^^^^^^^^^^^^^^

The master HPO run (named ``hpo_master_{timestamp}`` for Chemprop, ``hpo_{timestamp}`` for CheMeleon)
contains all experiment metadata and artifacts:

.. code-block:: text

   Master HPO Run (Parent)
   ├── Parameters
   │   ├── experiment_name, timestamp
   │   ├── data_path, smiles_column, target_columns
   │   ├── search_algorithm.* (type, seed, n_initial_points, persist_study, etc.)
   │   ├── asha.* (metric, mode, max_t, grace_period, reduction_factor)
   │   ├── resources.* (num_samples, cpus_per_trial, gpus_per_trial)
   │   └── transfer_learning.* (top_k, full_epochs, ensemble_size)
   │
   ├── Artifacts
   │   ├── config/
   │   │   └── hpo_config_{timestamp}.yaml  # Full HPO configuration
   │   ├── optuna/  # Optuna study artifacts (when persist_study: true)
   │   │   ├── optuna_trials.csv  # All trial results
   │   │   ├── optuna_study_summary.json  # Best trials and study metadata
   │   │   ├── optuna_param_importance.json  # Parameter importance (≥10 trials)
   │   │   └── optuna_studies.db  # Complete SQLite database
   │   ├── storage_dir/  # All files from search_algorithm.storage_dir
   │   │   └── **/*.{json,csv,yaml,txt,md,db}  # Excludes model checkpoints
   │   ├── hpo_results.csv  # Ray Tune results dataframe
   │   ├── top_k_configs.json  # Top K configurations for ensemble
   │   ├── study_metadata.json  # Study metadata
   │   └── best_model/
   │       └── best-*.ckpt  # Best trial checkpoint
   │
   └── Metrics
       ├── best.* (best trial: val_mae, val_loss, val_rmse, R², etc.)
       └── trials.*.{mean,std,min,max}  # Aggregate trial statistics

   Child Trial Runs (one per HPO trial)
   ├── Per-epoch metrics (val_mae, train_loss, lr, etc.)
   ├── Final trial metrics
   └── mlflow.parentRunId = {master_run_id}  # Linked to parent

Optuna Study Artifacts
^^^^^^^^^^^^^^^^^^^^^^^

When using Optuna with persistent studies (``persist_study: true``), comprehensive
study information is automatically logged:

**optuna_trials.csv**
   Complete trial history with all hyperparameters, metrics, and metadata.
   Useful for post-hoc analysis and visualization.

**optuna_study_summary.json**
   Study metadata including:

   - Best trial configuration and value
   - Top 10 trials
   - Study direction (minimize/maximize)
   - User and system attributes
   - Trial timing information

**optuna_param_importance.json**
   Parameter importance scores computed using fANOVA (when ≥10 trials).
   Identifies which hyperparameters have the strongest impact on performance.

**optuna_studies.db**
   Complete SQLite database containing the full Optuna study.
   Can be used to:

   - Resume optimization with warmstart
   - Perform custom analysis with Optuna API
   - Query trial history programmatically
   - Generate visualizations with Optuna's plotting functions

Example usage:

.. code-block:: python

   import optuna

   # Load study from logged database
   study = optuna.load_study(
       study_name="my_hpo_study",
       storage="sqlite:///path/to/optuna_studies.db"
   )

   # Access best trial
   print(f"Best trial: {study.best_trial.params}")
   print(f"Best value: {study.best_value}")

   # Generate visualization
   from optuna.visualization import plot_optimization_history
   fig = plot_optimization_history(study)
   fig.show()

Storage Directory Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All relevant files from ``search_algorithm.storage_dir`` are automatically logged
to MLflow, preserving directory structure:

**Included file types:**
   - Configuration files: ``*.yaml``, ``*.yml``, ``*.json``
   - Data files: ``*.csv``
   - Documentation: ``*.txt``, ``*.md``
   - Databases: ``*.db``

**Excluded file types:**
   Model checkpoints and large binary files are excluded to save storage:

   - ``*.ckpt``, ``*.pth``, ``*.pt``
   - ``*.h5``, ``*.pkl``, ``*.pickle``

This ensures all summary documents are preserved for reproducibility without
bloating the MLflow artifact store with large model files.

Configuration Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable comprehensive artifact logging, configure your HPO with:

.. code-block:: yaml

   search_algorithm:
     type: optuna  # Required for Optuna-specific artifacts
     persist_study: true  # Enable study persistence
     study_name: "my_hpo_experiment"  # Optional, auto-generated if not provided
     storage_dir: /path/to/hpo/storage  # Directory to log artifacts from

Accessing Artifacts via MLflow UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Start MLflow UI:

   .. code-block:: bash

      mlflow ui --port 5000

2. Navigate to your experiment
3. Find the master HPO run (sorted by start time, named ``hpo_master_*`` or ``hpo_*``)
4. View logged parameters, metrics, and artifacts
5. Download specific artifacts or the entire artifact directory

Using Artifacts for Reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The logged artifacts enable complete reproducibility:

1. **Configuration files**: Exact HPO settings for rerunning experiments
2. **Optuna database**: Resume optimization or analyze search trajectory
3. **Top-K configs**: Configurations for downstream ensemble training
4. **Study metadata**: Experiment context and provenance

Example workflow:

.. code-block:: python

   import mlflow

   # Access master HPO run
   client = mlflow.tracking.MlflowClient()
   run = client.get_run(run_id="master_run_id")

   # Download specific artifact
   artifact_path = client.download_artifacts(
       run_id="master_run_id",
       path="optuna/optuna_study_summary.json"
   )

   # Load top-K configs for ensemble training
   top_k_path = client.download_artifacts(
       run_id="master_run_id",
       path="top_k_configs.json"
   )

   with open(top_k_path) as f:
       top_configs = json.load(f)

Benefits
^^^^^^^^

**Complete Audit Trail**
   All experiment details captured in one place, making it easy to
   understand what was tried and why.

**Easy Collaboration**
   Share MLflow experiment URL with team members for instant access
   to all HPO results and artifacts.

**Warmstart Capability**
   Optuna database enables continuing optimization from previous runs
   without starting from scratch.

**Storage Efficiency**
   Only summary documents are logged; model checkpoints are excluded
   to prevent artifact store bloat.

**Post-Hoc Analysis**
   Download artifacts for custom analysis, visualization, or reporting
   without re-running expensive HPO.

Warmstarting Optimization
--------------------------

Continue optimization from previous runs using persistent :term:`Optuna` studies.
Warmstarting leverages historical trial data to accelerate convergence.

**Key Benefits:**

- **30-50% fewer trials** to reach optimal configurations
- Reuse expensive trial evaluations from previous runs
- Iteratively refine search without starting from scratch
- Build institutional knowledge of hyperparameter landscapes

Configuration
^^^^^^^^^^^^^

Enable study persistence in your HPO config:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     seed: 42
     n_initial_points: 100
     persist_study: true
     study_name: "chemprop_hpo_v1"
     storage_dir: "hpo_results/optuna_studies"

To warmstart from a previous study:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     persist_study: true
     study_name: "chemprop_hpo_v2"       # New study name
     warmstart_from: "chemprop_hpo_v1"   # Load trials from this study
     warmstart_n_trials: 15              # Number of top trials to enqueue

CLI Usage
^^^^^^^^^

List available studies:

.. code-block:: bash

   admet model hpo-list-studies
   admet model hpo-list-studies --storage-dir my_studies/ --verbose

Run HPO with warmstart:

.. code-block:: bash

   admet model hpo -c configs/hpo_warmstart.yaml --num-samples 50

Workflow: Iterative Refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Phase 1: Broad Search**

.. code-block:: yaml

   experiment_name: chemprop_broad_search
   search_algorithm:
     persist_study: true
     study_name: "chemprop_v1_broad"

   search_space:
     learning_rate:
       type: loguniform
       low: 0.0001
       high: 0.03  # Wide range

   resources:
     num_samples: 1000

**Phase 2: Focused Search**

.. code-block:: yaml

   experiment_name: chemprop_focused_search
   search_algorithm:
     persist_study: true
     study_name: "chemprop_v2_focused"
     warmstart_from: "chemprop_v1_broad"
     warmstart_n_trials: 20

   search_space:
     learning_rate:
       type: loguniform
       low: 0.001
       high: 0.01  # Narrowed based on v1 results

   resources:
     num_samples: 500

Warmstart Parameters
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``persist_study``
     - bool
     - Enable persistent study storage (default: False)
   * - ``study_name``
     - str
     - Unique study identifier (auto-generated if None)
   * - ``storage_dir``
     - str
     - Directory for SQLite database (default: hpo_results/optuna_studies)
   * - ``warmstart_from``
     - str
     - Name of previous study to load trials from
   * - ``warmstart_n_trials``
     - int
     - Number of top trials to enqueue (default: 10)

Example Configurations
----------------------

Minimal HPO Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   experiment_name: minimal_hpo
   data_path: data/train.csv
   val_data_path: data/val.csv
   smiles_column: SMILES
   target_columns: ["LogD"]

   search_space:
     learning_rate:
       type: loguniform
       low: 1.0e-4
       high: 1.0e-2

   asha:
     metric: val_mae
     mode: min
     max_t: 50
     grace_period: 10

   resources:
     num_samples: 100
     gpus_per_trial: 0.5

Production HPO Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See ``configs/1-hpo-single/hpo_chemprop.yaml`` for a complete production configuration
with all available parameters.

API Reference
-------------

See the API documentation for detailed class and function references:

- :mod:`admet.model.chemprop.hpo` - Main HPO orchestrator
- :mod:`admet.model.chemprop.hpo_config` - Configuration dataclasses
- :mod:`admet.model.chemprop.hpo_search_space` - Search space builders
- :mod:`admet.model.chemprop.hpo_trainable` - Ray Tune trainable function

Cross-References
----------------

- See :doc:`modeling` for general modeling guide
- See :doc:`profiling` for performance profiling and optimization during HPO
- See :doc:`configuration` for configuration file format
- See :doc:`splitting` for dataset preparation
