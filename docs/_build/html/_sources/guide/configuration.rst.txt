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
       - 1.031
       - 1.0
       - 1.2252
       - 1.1368
       - 2.1227
       - 2.1174
       - 3.7877
       - 5.2722
       - 10.0

   # Model configuration (unified format)
   model:
     type: chemprop              # Model type: chemprop, chemeleon, xgboost, lightgbm, catboost
     chemprop:                   # Model-specific parameters
       depth: 5                  # Message passing iterations
       message_hidden_dim: 600   # MPNN hidden dimension
       num_layers: 2             # FFN layers
       hidden_dim: 600           # FFN hidden dimension
       dropout: 0.1
       batch_norm: true
       ffn_type: regression      # regression, mixture_of_experts, branched
       aggregation: mean

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

Supported Model Types
---------------------

The unified configuration supports multiple model types via ``model.type``:

**PyTorch Models** (support JointSampling):

- ``chemprop``: Message-passing neural network
- ``chemeleon``: Pretrained foundation model with fine-tuning

**Classical Models** (fingerprint-based, no JointSampling):

- ``xgboost``: XGBoost gradient boosting
- ``lightgbm``: LightGBM gradient boosting
- ``catboost``: CatBoost gradient boosting

Example for classical model (XGBoost):

.. code-block:: yaml

   model:
     type: xgboost
     fingerprint:
       type: morgan
       morgan:
         radius: 2
         n_bits: 2048
     xgboost:
       n_estimators: 500
       max_depth: 8
       learning_rate: 0.05

Ensemble Configuration
----------------------

Example ``configs/4-more-models/chemprop.yaml``:

.. code-block:: yaml

   # Data root containing split_*/fold_*/ subdirectories
   data:
     data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"
     test_file: "assets/dataset/set/local_test.csv"
     blind_file: "assets/dataset/set/blind_test.csv"
     splits: null    # null = use all, or [0, 1, 2]
     folds: null     # null = use all, or [0, 1, 2, 3, 4]

   # Model configuration (unified format)
   model:
     type: chemprop
     chemprop:
       depth: 3
       message_hidden_dim: 700
       aggregation: norm
       ffn_type: regression
       num_layers: 4
       hidden_dim: 200
       dropout: 0.15
       batch_norm: true

   # Optimization
   optimization:
     criterion: MAE
     init_lr: 0.00113
     max_lr: 0.000227
     final_lr: 0.000113
     warmup_epochs: 5
     max_epochs: 150
     patience: 15
     batch_size: 128

   mlflow:
     tracking: true
     experiment_name: "ensemble_chemprop"
     nested: true    # Log each fold as child run

   # Ray parallelization settings
   ray:
     max_parallel: 5    # Models trained concurrently
     num_cpus: null     # null = use all available
     num_gpus: null     # null = auto-detect

   # Joint sampling configuration (task oversampling + curriculum)
   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.02      # Balance factor for task-aware sampling
     curriculum:
       enabled: false
       quality_col: "Quality"
       qualities: ["high", "medium", "low"]
       patience: 5
       strategy: "sampled"  # or "deterministic"
       log_per_quality_metrics: true

Programmatic Loading
--------------------

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.config import UnifiedModelConfig
   from admet.model.registry import create_model_from_config

   # Load and merge configuration
   config = OmegaConf.merge(
       OmegaConf.structured(UnifiedModelConfig),
       OmegaConf.load("configs/4-more-models/chemprop.yaml")
   )

   # Access nested settings (unified format)
   print(config.model.type)              # "chemprop"
   print(config.model.chemprop.depth)    # 3
   print(config.optimization.max_lr)     # 0.000227

   # Create model from config (automatic type detection)
   model = create_model_from_config(config)

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

The ``joint_sampling`` section enables unified task-aware oversampling and curriculum learning
via the ``JointSampler``, which uses a two-stage sampling algorithm:

**Two-Stage Sampling Algorithm:**

1. **Stage 1 (Task Selection):** Sample task ``t`` with probability ``p_t ∝ count_t^(-α)``
2. **Stage 2 (Within-Task):** Sample molecule from task ``t``'s valid indices, weighted by curriculum

.. code-block:: yaml

   joint_sampling:
     enabled: true

     # Task-aware oversampling configuration
     task_oversampling:
       alpha: 0.5         # [0, 1] - 0=uniform, 0.5=moderate, 1=full inverse-weighted

     # Curriculum learning configuration
     curriculum:
       enabled: true
       quality_col: "Quality"
       qualities: ["high", "medium", "low"]
       patience: 5        # Epochs before phase transition
       strategy: "sampled"

       # Count normalization (recommended for imbalanced datasets)
       count_normalize: true
       min_high_quality_proportion: 0.25

       # Metric alignment
       monitor_metric: "val/mae/high"
       early_stopping_metric: null  # Uses monitor_metric if null

       # Adaptive curriculum (auto-adjust proportions)
       adaptive_enabled: false
       adaptive_improvement_threshold: 0.02
       adaptive_max_adjustment: 0.1
       adaptive_lookback_epochs: 5

       # Loss weighting (scale gradients by quality)
       loss_weighting_enabled: false
       loss_weights:
         high: 1.0
         medium: 0.5
         low: 0.3

       # Optional: customize phase proportions
       # warmup_proportions: [0.80, 0.15, 0.05]
       # expand_proportions: [0.60, 0.30, 0.10]
       # robust_proportions: [0.50, 0.35, 0.15]
       # polish_proportions: [0.70, 0.20, 0.10]

       log_per_quality_metrics: true
       reset_early_stopping_on_phase_change: false

     # Global sampling settings
     num_samples: null              # null = use full dataset size per epoch
     seed: 42
     increment_seed_per_epoch: true # Different samples each epoch
     log_to_mlflow: true            # Log weight statistics

**Task Oversampling (``task_oversampling.alpha``):**

- ``α=0``: Uniform task sampling (no rebalancing)
- ``α=0.5``: Moderate rebalancing (default, recommended)
- ``α=1.0``: Full inverse-proportional (rare tasks heavily favored)

**Curriculum Learning:** Progressively trains through four phases:

1. **warmup**: Focus on high-quality data (default: 80% high, 15% medium, 5% low)
2. **expand**: Include medium-quality (default: 60% high, 30% medium, 10% low)
3. **robust**: Include low-quality for robustness (default: 50% high, 35% medium, 15% low)
4. **polish**: Return focus to high-quality while maintaining diversity (default: 70% high, 20% medium, 10% low)

**Important:** When using with ``num_workers > 0`` in DataLoader, the sampler state
(epoch counter, curriculum phase) is NOT synchronized across workers. Use ``num_workers=0``
for reliable curriculum learning.

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

  - ``TaskOversamplingConfig``: Task-aware oversampling settings (alpha parameter)
  - ``CurriculumConfig``: Curriculum learning settings (phases, proportions, metrics)

- ``TaskAffinityConfig``: Legacy task affinity grouping (pre-training phase)
- ``InterTaskAffinityConfig``: Inter-task affinity computation during training
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
