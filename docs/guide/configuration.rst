Configuration Guide
===================

This page explains how model hyperparameters, training options, and experiment
settings are configured using OmegaConf dataclasses and YAML files.

Configuration System
--------------------

The package uses OmegaConf-based dataclasses for type-safe configuration.
YAML files under ``configs/`` define settings that are loaded and merged
with structured defaults.

Available Configurations
------------------------

- ``ChempropConfig``: Single model training
- ``EnsembleConfig``: Ensemble training across splits/folds
- ``HPOConfig``: Hyperparameter optimization settings

Single Model Configuration
--------------------------

Example ``configs/0-experiment/chemprop.yaml``:

.. code-block:: yaml

   # Data configuration
   data:
     data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0"
     test_file: "assets/dataset/set/local_test.csv"
     blind_file: "assets/dataset/set/blind_test.csv"
     smiles_col: "SMILES"
     target_cols:
       - "LogD"
       - "Log KSOL"
       - "Log HLM CLint"
       - "Log MLM CLint"
       - "Log Caco-2 Permeability Papp A>B"
       - "Log Caco-2 Permeability Efflux"
       - "Log MPPB"
       - "Log MBPB"
       - "Log MGMB"
     target_weights:
       - 0.5
       - 2.0
       - 3.0
       - 3.0
       - 3.0
       - 2.0
       - 2.0
       - 3.0
       - 4.0

   # Model architecture
   model:
     depth: 5                  # Message passing iterations
     message_hidden_dim: 600   # MPNN hidden dimension
     num_layers: 2             # FFN layers
     hidden_dim: 600           # FFN hidden dimension
     dropout: 0.1
     batch_norm: true
     ffn_type: "regression"    # or "mixture_of_experts", "branched"

   # Optimization settings
   optimization:
     criterion: "MSE"
     init_lr: 1.0e-4
     max_lr: 1.0e-3
     final_lr: 1.0e-4
     warmup_epochs: 5
     max_epochs: 150
     patience: 15
     batch_size: 32
     seed: 12345

   # MLflow tracking
   mlflow:
     tracking: true
     tracking_uri: "http://127.0.0.1:8084"
     experiment_name: "chemprop_single"

Ensemble Configuration
----------------------

Example ``configs/0-experiment/ensemble_chemprop_production.yaml``:

.. code-block:: yaml

   # Data root containing split_*/fold_*/ subdirectories
   data:
     data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"
     test_file: "assets/dataset/set/local_test.csv"
     blind_file: "assets/dataset/set/blind_test.csv"
     splits: null    # null = use all, or [0, 1, 2]
     folds: null     # null = use all, or [0, 1, 2, 3, 4]

   # Model and optimization same as single config...

   mlflow:
     tracking: true
     experiment_name: "ensemble_chemprop"
     nested: true    # Log each fold as child run

   # Ray parallelization settings
   ray:
     max_parallel: 4    # Models trained concurrently
     num_cpus: null     # null = use all available
     num_gpus: null     # null = auto-detect

   # Joint sampling configuration (task oversampling + curriculum)
   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.5       # Balance factor for task-aware sampling
     curriculum:
       enabled: true
       quality_col: "Quality"
       qualities: ["high", "medium", "low"]
       patience: 5
       strategy: "sampled"  # or "deterministic"
       log_per_quality_metrics: true

Programmatic Loading
--------------------

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.chemprop import ChempropConfig, ChempropModel

   # Load and merge configuration
   config = OmegaConf.merge(
       OmegaConf.structured(ChempropConfig),
       OmegaConf.load("configs/0-experiment/chemprop.yaml")
   )

   # Access nested settings
   print(config.model.ffn_type)      # "regression"
   print(config.optimization.max_lr)  # 0.001

   # Create model from config
   model = ChempropModel.from_config(config)

Ray Parallelization Configuration
---------------------------------

The ``ray`` configuration section controls distributed ensemble training:

.. code-block:: yaml

   ray:
     max_parallel: 4    # Maximum models trained in parallel
     num_cpus: 8        # CPU allocation (null = all available)
     num_gpus: 1        # GPU allocation (null = auto-detect)

Key parameters:

- ``max_parallel``: How many models train simultaneously. Set based on GPU memory.
  For example, if each model needs 0.5 GPU, set ``max_parallel: 2``.
- ``num_cpus``: Total CPUs for Ray cluster. ``null`` uses all available.
- ``num_gpus``: Total GPUs for Ray cluster. ``null`` auto-detects available GPUs.

Joint Sampling Configuration
----------------------------

The ``joint_sampling`` section enables unified task-aware oversampling and curriculum learning:

.. code-block:: yaml

   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.5         # Balance factor (0=uniform, 1=fully weighted)
     curriculum:
       enabled: true
       quality_col: "Quality"
       qualities: ["high", "medium", "low"]
       patience: 5        # Epochs before quality phase change
       strategy: "sampled"  # or "deterministic"
       reset_early_stopping_on_phase_change: false
       log_per_quality_metrics: true
     num_samples: null    # null = use full dataset size
     seed: 42
     increment_seed_per_epoch: true
     log_to_mlflow: true

**Task Oversampling**: Balances multi-task learning by oversampling tasks with fewer samples.
The ``alpha`` parameter controls the balance: 0 = uniform sampling, 1 = fully inverse-weighted.

**Curriculum Learning**: Progressively trains on data of increasing difficulty (e.g., high → medium → low quality).
The sampler automatically advances through quality phases based on validation performance and ``patience``.

See :doc:`curriculum` for detailed curriculum learning documentation.

Configuration Classes
---------------------

The dataclass hierarchy:

- ``DataConfig``: Data paths and column names
- ``ModelConfig``: Model architecture parameters
- ``OptimizationConfig``: Training hyperparameters
- ``MlflowConfig``: Experiment tracking settings
- ``RayConfig``: Ray parallelization settings for ensemble training
- ``JointSamplingConfig``: Unified task oversampling and curriculum learning
- ``ChempropConfig``: Combines all above for single model
- ``EnsembleDataConfig``: Extends DataConfig for ensemble
- ``EnsembleConfig``: Combines all for ensemble training

MLflow Integration
------------------

Training runs automatically log to MLflow:

- All configuration values as parameters
- Training metrics per epoch
- Model checkpoints as artifacts
- Ensemble runs use nested child runs

Start the MLflow server:

.. code-block:: bash

   mlflow server --host 127.0.0.1 --port 8080

Best Practices
--------------

- Keep model-specific parameters grouped by section
- Use ``seed`` for reproducibility
- Set ``target_weights`` to emphasize important endpoints
- Use ``nested: true`` for ensemble runs to group folds
