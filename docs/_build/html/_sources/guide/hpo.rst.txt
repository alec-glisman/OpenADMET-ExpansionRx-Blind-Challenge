Hyperparameter Optimization Guide
==================================

This guide covers hyperparameter optimization (HPO) for Chemprop models using
Ray Tune with the ASHA scheduler. HPO enables systematic exploration of model
configurations to find optimal hyperparameters for your ADMET prediction tasks.

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

   python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml

Or programmatically:

.. code-block:: python

   from admet.model.chemprop.hpo import ChempropHPO
   from omegaconf import OmegaConf

   config = OmegaConf.load("configs/hpo_chemprop.yaml")
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

See ``configs/hpo_chemprop.yaml`` for a complete production configuration
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
- See :doc:`configuration` for configuration file format
- See :doc:`splitting` for dataset preparation

