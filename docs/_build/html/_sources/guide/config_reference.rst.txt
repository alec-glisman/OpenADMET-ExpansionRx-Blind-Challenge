===============================
Configuration File Reference
===============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

This project uses YAML configuration files to define all aspects of model training,
hyperparameter optimization, and evaluation. All configs are located in the ``configs/``
directory and can be loaded with OmegaConf.

Configuration files are organized by purpose:

- **Base configs**: ``chemprop.yaml``
- **Task affinity**: ``chemprop_task_affinity*.yaml``
- **Ensemble training**: ``ensemble_chemprop*.yaml``
- **Curriculum learning**: ``*curriculum*.yaml``
- **HPO**: ``hpo_chemprop.yaml``
- **Utilities**: ``task_affinity_compute.yaml``

Available Configuration Files
==============================

Base Configurations
-------------------

chemprop.yaml
^^^^^^^^^^^^^

**Purpose**: Standard single-model Chemprop training without task affinity or curriculum learning.

**Use case**: Baseline training, quick experiments, or when you want a simple multi-task setup.

**Key features**:

- 9 ADMET target columns
- Message passing depth: 5
- FFN layers: 2
- Hidden dimension: 600
- Per-target loss weights included
- 100 epochs with cosine LR scheduling

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.model --config configs/chemprop.yaml

**Key parameters**:

.. code-block:: yaml

    model:
      depth: 5
      message_hidden_dim: 600
      num_layers: 2
      hidden_dim: 600
      ffn_type: "regression"
    
    optimization:
      epochs: 100
      batch_size: 64
      learning_rate: 0.001

Task Affinity Configurations
-----------------------------

chemprop_task_affinity.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Standard task affinity training (recommended starting point).

**Use case**: Production training with automatic task grouping to improve multi-task learning.

**Key features**:

- Task affinity enabled with 3 groups
- 1 epoch for affinity computation
- Cosine similarity for gradient affinity
- Agglomerative clustering
- Same architecture as base config

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.model --config configs/chemprop_task_affinity.yaml

**Key parameters**:

.. code-block:: yaml

    task_affinity:
      enabled: true
      n_groups: 3
      affinity_epochs: 1
      affinity_batch_size: 64
      clustering_method: "agglomerative"

**See also**: :doc:`task_affinity` for detailed documentation.

chemprop_task_affinity_advanced.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Advanced task affinity with all parameters explicitly configured.

**Use case**: Fine-tuning task grouping for complex datasets or when default settings are insufficient.

**Key features**:

- 4 task groups (more granular)
- 2 epochs for affinity computation (more stable estimates)
- Smaller batch size (32) for more gradient samples
- Larger model (800 hidden dim)
- 150 training epochs
- Conservative learning rate (0.0005 for affinity)

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.model --config configs/chemprop_task_affinity_advanced.yaml

**Key parameters**:

.. code-block:: yaml

    task_affinity:
      enabled: true
      n_groups: 4
      affinity_epochs: 2
      affinity_batch_size: 32
      affinity_lr: 0.0005
    
    model:
      depth: 6
      message_hidden_dim: 800
      num_layers: 3
      hidden_dim: 800

chemprop_task_affinity_explore.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Quick experiments to find optimal number of task groups.

**Use case**: Exploration phase before production training. Run with different ``n_groups`` values.

**Key features**:

- 30 epochs only (quick experiments)
- 2 affinity epochs for stable estimates
- Moderate model size (400 hidden dim)
- Easy CLI override of ``n_groups``
- Separate MLflow experiment for exploration

**Usage**:

.. code-block:: bash

    # Try different group numbers
    for n in 2 3 4 5; do
      python -m admet.model.chemprop.model \
        --config configs/chemprop_task_affinity_explore.yaml \
        --task-affinity.n-groups $n \
        --mlflow.run-name "explore_n${n}"
    done

**Key parameters**:

.. code-block:: yaml

    task_affinity:
      enabled: true
      n_groups: 3  # Override via CLI
      affinity_epochs: 2
    
    optimization:
      epochs: 30  # Short for exploration

task_affinity_compute.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Compute task affinity matrix without full training.

**Use case**: Analysis and visualization of task relationships before deciding on training strategy.

**Key features**:

- Affinity computation only (no model training)
- Outputs: affinity matrix (NPZ), CSV, heatmap PNG, groups JSON
- 2 epochs for stable affinity estimates
- Configurable output paths

**Usage**:

.. code-block:: bash

    python -m admet.cli.compute_task_affinity \
      --config configs/task_affinity_compute.yaml \
      --save-path results/task_affinity.npz \
      --plot-heatmap results/affinity_heatmap.png

**Key parameters**:

.. code-block:: yaml

    task_affinity:
      affinity_epochs: 2
      n_groups: 3
      log_affinity_matrix: true
    
    output:
      affinity_matrix_path: "results/task_affinity_matrix.npz"
      heatmap_path: "results/task_affinity_heatmap.png"

Ensemble Configurations
-----------------------

ensemble_chemprop_1.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: First ensemble configuration variant.

**Use case**: Ensemble training across multiple data splits and folds.

**Key features**:

- Discovers all split_*/fold_*/ directories automatically
- Parallel training with Ray
- Ensemble predictions with uncertainty estimates
- Nested MLflow runs (one parent, multiple children)

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.ensemble \
      --config configs/ensemble_chemprop_1.yaml \
      --max-parallel 2

**Key parameters**:

.. code-block:: yaml

    ensemble:
      data_dir: "assets/dataset/splits/data"
      splits: null  # null = use all
      folds: null   # null = use all
    
    # max_parallel set via CLI or in config

ensemble_chemprop_2.yaml, ensemble_chemprop_3.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Alternative ensemble configurations with different hyperparameters.

**Use case**: Exploring different architectures or training strategies for ensemble diversity.

**Key differences**: Likely vary in model architecture, dropout rates, or other hyperparameters
to increase ensemble diversity (specifics depend on file contents).

ensemble_chemprop_production.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Production-ready ensemble configuration.

**Use case**: Final training after hyperparameter optimization and validation.

**Key features**:

- Optimized hyperparameters from HPO
- Full training epochs (100+)
- All splits and folds
- Production MLflow experiment name

Curriculum Learning Configurations
-----------------------------------

chemprop_curriculum.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Single model with curriculum learning enabled.

**Use case**: Training on datasets with quality annotations where gradual learning improves results.

**Key features**:

- Curriculum learning based on data quality
- Pacing function to control difficulty progression
- Quality-based sample weighting
- Same base architecture as standard config

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.model --config configs/chemprop_curriculum.yaml

**Key parameters**:

.. code-block:: yaml

    curriculum:
      enabled: true
      pacing_function: "linear"
      quality_column: "Quality"
      start_epoch: 0
      end_epoch: 50

**See also**: :doc:`curriculum` for detailed documentation.

ensemble_curriculum.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Ensemble training with curriculum learning.

**Use case**: Combining benefits of ensemble diversity with curriculum learning's gradual difficulty increase.

**Key features**:

- Curriculum learning across all ensemble members
- Consistent pacing across splits/folds
- Ensemble predictions after curriculum training

Hyperparameter Optimization
----------------------------

hpo_chemprop.yaml
^^^^^^^^^^^^^^^^^

**Purpose**: Ray Tune hyperparameter optimization with ASHA scheduler.

**Use case**: Finding optimal hyperparameters before production training.

**Key features**:

- ASHA scheduler for early stopping
- Conditional search spaces (MoE, branched FFN)
- Multiple trials in parallel
- Top-k config export
- Full MLflow integration

**Usage**:

.. code-block:: bash

    python -m admet.model.chemprop.hpo \
      --config configs/hpo_chemprop.yaml \
      --num-samples 50

    # Or use the convenience script
    ./scripts/training/train_chemprop_hpo.sh configs/hpo_chemprop.yaml

**Key parameters**:

.. code-block:: yaml

    hpo:
      num_samples: 50
      max_concurrent_trials: 4
      grace_period: 10
      reduction_factor: 3
      metric: "val_loss"
      mode: "min"
    
    search_space:
      depth: [3, 4, 5, 6]
      message_hidden_dim: [300, 400, 600, 800]
      learning_rate: [1e-4, 1e-3, 5e-3]
      dropout: [0.0, 0.1, 0.2, 0.3]

**See also**: :doc:`hpo` for detailed documentation.

Configuration Structure
=======================

All configuration files share a common structure with these main sections:

Data Section
------------

Controls data loading, column mapping, and file paths.

.. code-block:: yaml

    data:
      # Single model
      data_dir: "path/to/train_val"  # Contains train.csv, val.csv
      
      # OR Ensemble
      data_dir: "path/to/splits"     # Contains split_*/fold_*/
      splits: null                    # null = all, or [0, 1, 2]
      folds: null                     # null = all, or [0, 1, 2, 3, 4]
      
      # Common
      test_file: "path/to/test.csv"
      blind_file: "path/to/blind.csv"
      output_dir: null                # null = temp directory
      
      # Columns
      smiles_col: "SMILES"
      target_cols:
        - "LogD"
        - "Log KSOL"
        # ... more targets
      
      # Optional per-target weights
      target_weights:
        - 1.0
        - 1.5
        # ... one per target

Model Section
-------------

Defines model architecture parameters.

.. code-block:: yaml

    model:
      # Message Passing Network
      depth: 5                    # Message passing iterations
      message_hidden_dim: 600     # MPNN hidden size
      batch_norm: true            # Batch normalization
      
      # Feed-Forward Network
      ffn_type: "regression"      # "regression", "mixture_of_experts", "branched"
      num_layers: 2               # FFN layers
      hidden_dim: 600             # FFN hidden size
      dropout: 0.1                # Dropout probability
      
      # Branched FFN (when ffn_type="branched")
      trunk_n_layers: 2
      trunk_hidden_dim: 600
      
      # Mixture of Experts (when ffn_type="mixture_of_experts")
      n_experts: 4

Optimization Section
--------------------

Training hyperparameters and learning rate schedule.

.. code-block:: yaml

    optimization:
      # Loss and metrics
      criterion: "MSE"            # "MAE", "MSE", "RMSE"
      
      # Training duration
      epochs: 100
      batch_size: 64
      
      # Learning rate
      learning_rate: 0.001
      scheduler: "cosine"         # "constant", "cosine", "exponential", "step"
      warmup_epochs: 5
      max_lr: 0.001
      final_lr: 0.0001
      
      # Optimizer
      optimizer: "Adam"
      weight_decay: 0.0
      gradient_clip: null         # null or float (e.g., 1.0)
      
      # Early stopping
      patience: 20
      min_delta: 0.0001

Task Affinity Section
---------------------

Task grouping for improved multi-task learning (optional).

.. code-block:: yaml

    task_affinity:
      enabled: false              # Set to true to enable
      
      # Grouping
      n_groups: 3
      clustering_method: "agglomerative"  # "agglomerative" or "spectral"
      
      # Affinity computation
      affinity_epochs: 1
      affinity_batch_size: 64
      affinity_lr: 0.001
      
      # Scoring
      affinity_type: "cosine"     # "cosine" or "dot_product"
      normalize_gradients: true
      
      # Advanced
      encoder_param_patterns: []
      device: "auto"
      seed: 42
      log_affinity_matrix: true

Curriculum Learning Section
----------------------------

Gradual learning based on sample difficulty (optional).

.. code-block:: yaml

    curriculum:
      enabled: false              # Set to true to enable
      
      # Pacing
      pacing_function: "linear"   # "linear", "exponential", "step"
      start_epoch: 0
      end_epoch: 50
      
      # Data
      quality_column: "Quality"
      sort_ascending: true        # true = easy to hard
      
      # Weighting
      use_sample_weights: true
      min_weight: 0.1
      max_weight: 1.0

MLflow Section
--------------

Experiment tracking configuration.

.. code-block:: yaml

    mlflow:
      # Tracking
      experiment_name: "chemprop_experiment"
      run_name: null              # Auto-generated if null
      tracking_uri: "mlruns"      # Local or "http://host:port"
      
      # Logging
      log_models: true
      log_plots: true
      log_frequency: 10           # Log every N batches
      
      # Ensemble-specific
      nested: true                # Use nested runs for ensemble

System Section
--------------

Hardware and reproducibility settings.

.. code-block:: yaml

    # Device
    device: "auto"                # "auto", "cpu", "cuda", "cuda:N"
    
    # Reproducibility
    seed: 42
    
    # Logging
    log_level: "INFO"             # "DEBUG", "INFO", "WARNING", "ERROR"
    log_frequency: 10

Usage Patterns
==============

Loading Configurations
----------------------

**Python**:

.. code-block:: python

    from omegaconf import OmegaConf
    from admet.model.chemprop import ChempropConfig, ChempropModel
    
    # Load config
    config = OmegaConf.merge(
        OmegaConf.structured(ChempropConfig),
        OmegaConf.load("configs/chemprop.yaml")
    )
    
    # Use config
    model = ChempropModel.from_config(config)

**CLI**:

.. code-block:: bash

    python -m admet.model.chemprop.model --config configs/chemprop.yaml

Overriding Parameters
---------------------

**Via CLI**:

.. code-block:: bash

    python -m admet.model.chemprop.model \
      --config configs/chemprop.yaml \
      --model.depth 6 \
      --optimization.learning-rate 0.0005 \
      --task-affinity.enabled true \
      --task-affinity.n-groups 4

**Via Python**:

.. code-block:: python

    config = OmegaConf.load("configs/chemprop.yaml")
    config.model.depth = 6
    config.optimization.learning_rate = 0.0005
    config.task_affinity.enabled = True
    config.task_affinity.n_groups = 4

Merging Multiple Configs
-------------------------

.. code-block:: python

    from omegaconf import OmegaConf
    
    # Load base and override configs
    base = OmegaConf.load("configs/chemprop.yaml")
    override = OmegaConf.load("configs/my_overrides.yaml")
    
    # Merge (override takes precedence)
    config = OmegaConf.merge(base, override)

Configuration Best Practices
=============================

1. **Start Simple**: Begin with ``chemprop.yaml`` or ``chemprop_task_affinity.yaml``

2. **Version Control**: Commit all config changes with meaningful messages

3. **Naming Convention**: Use descriptive names that indicate purpose::

    chemprop_<feature>_<variant>.yaml
    ensemble_<feature>_<variant>.yaml

4. **Document Changes**: Add comments explaining non-obvious parameter choices

5. **Reproducibility**: Always set ``seed`` and save config with results

6. **Hyperparameters**: Use HPO to find optimal values before production

7. **Validation**: Test configs on small data subset before full training

8. **Modular Design**: Keep reusable settings in base configs, override specifics

Quick Reference: Config Selection
==================================

Choose your configuration based on your goal:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Goal
     - Config File
     - Notes
   * - Quick baseline
     - ``chemprop.yaml``
     - Standard multi-task
   * - Task affinity (production)
     - ``chemprop_task_affinity.yaml``
     - 3 groups, 1 affinity epoch
   * - Find optimal groups
     - ``chemprop_task_affinity_explore.yaml``
     - Try n=2,3,4,5
   * - Advanced task affinity
     - ``chemprop_task_affinity_advanced.yaml``
     - 4 groups, 2 affinity epochs
   * - Analyze task relationships
     - ``task_affinity_compute.yaml``
     - No training, just affinity
   * - Ensemble training
     - ``ensemble_chemprop_production.yaml``
     - All splits/folds
   * - Curriculum learning
     - ``chemprop_curriculum.yaml``
     - Quality-based pacing
   * - Hyperparameter search
     - ``hpo_chemprop.yaml``
     - Ray Tune + ASHA

See Also
========

- :doc:`configuration` - General configuration guide
- :doc:`task_affinity` - Task affinity detailed guide
- :doc:`curriculum` - Curriculum learning guide
- :doc:`hpo` - Hyperparameter optimization guide
- :doc:`modeling` - Model training guide
