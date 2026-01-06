Classical Models Guide
======================

This guide explains how to use traditional machine learning models (XGBoost,
LightGBM, CatBoost) with molecular fingerprints for ADMET prediction.

Overview
--------

Classical models use molecular fingerprints as features instead of graph-based
message-passing. They are:

- **Fast to train**: Minutes instead of hours for neural models
- **No GPU required**: Pure CPU-based training
- **Strong baselines**: Often competitive with deep learning
- **Interpretable**: Feature importance is straightforward

Supported Models
----------------

- **XGBoost**: Gradient boosting with highly-tuned hyperparameters
- **LightGBM**: Histogram-based gradient boosting for speed
- **CatBoost**: Ordered boosting with native categorical support

All three models support multi-output regression via scikit-learn's
``MultiOutputRegressor`` wrapper.

Fingerprint Types
-----------------

The ``fingerprint`` configuration controls feature generation:

**Morgan Fingerprints** (default, recommended):

- Circular fingerprints capturing molecular substructures
- ``radius``: Neighborhood radius (default: 2)
- ``n_bits``: Fingerprint size (default: 2048)

**RDKit Fingerprints**:

- Path-based topological fingerprints
- Fixed 2048-bit representation

**MACCS Keys**:

- 167-bit structural key fingerprints
- Fixed set of predefined substructures

**Mordred Descriptors**:

- ~1800 molecular descriptors
- Physicochemical properties, topological indices, etc.

Quick Start
-----------

1. Create Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Create ``configs/4-more-models/xgboost_example.yaml``:

.. code-block:: yaml

   data:
     data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0"
     test_file: "assets/dataset/set/local_test.csv"
     smiles_col: "SMILES"
     target_cols:
       - "LogD"
       - "Log KSOL"
       - "Log HLM CLint"
       # ... other targets
     target_weights:
       - 1.0
       - 1.0
       - 1.2
       # ... per-target weights

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
       subsample: 0.8
       colsample_bytree: 0.8
       reg_alpha: 0.1
       reg_lambda: 1.0

   mlflow:
     enabled: true  # Use 'enabled' (canonical) or 'tracking' (legacy alias)
     tracking_uri: http://127.0.0.1:8084
     experiment_name: "xgboost_morgan"

2. Train Model
^^^^^^^^^^^^^^

.. code-block:: bash

   admet model train -c configs/4-more-models/xgboost_example.yaml

Or programmatically:

.. code-block:: python

   from admet.model.registry import ModelRegistry
   from omegaconf import OmegaConf

   config = OmegaConf.load("configs/4-more-models/xgboost_example.yaml")
   model = ModelRegistry.create(config)

   # Load your data
   model.fit(smiles_train, targets_train)
   predictions = model.predict(smiles_test)

Model-Specific Configuration
-----------------------------

XGBoost
^^^^^^^

.. code-block:: yaml

   model:
     type: xgboost
     xgboost:
       n_estimators: 500        # Number of boosting rounds
       max_depth: 8             # Maximum tree depth
       learning_rate: 0.05      # Step size shrinkage
       subsample: 0.8           # Row sampling ratio
       colsample_bytree: 0.8    # Column sampling ratio
       reg_alpha: 0.1           # L1 regularization
       reg_lambda: 1.0          # L2 regularization
       min_child_weight: 3      # Minimum sum of instance weight
       gamma: 0.0               # Minimum loss reduction for split

**Key Hyperparameters**:

- ``n_estimators``: More trees = better fit but slower training. Start with 100-500.
- ``max_depth``: Controls model complexity. 4-10 is typical. Deeper = more overfitting risk.
- ``learning_rate``: Lower values need more estimators but may generalize better.
- Regularization (``reg_alpha``, ``reg_lambda``): Reduce overfitting.

LightGBM
^^^^^^^^

.. code-block:: yaml

   model:
     type: lightgbm
     lightgbm:
       n_estimators: 500
       max_depth: 8
       learning_rate: 0.05
       num_leaves: 31           # Maximum leaves per tree
       subsample: 0.8
       colsample_bytree: 0.8
       reg_alpha: 0.1
       reg_lambda: 1.0
       min_child_samples: 20    # Minimum samples in leaf

**Key Differences from XGBoost**:

- ``num_leaves`` instead of ``max_depth`` (leaf-wise growth)
- Faster training on large datasets
- Lower memory usage
- Built-in categorical feature support

CatBoost
^^^^^^^^

.. code-block:: yaml

   model:
     type: catboost
     catboost:
       iterations: 500          # Equivalent to n_estimators
       depth: 8                 # Tree depth
       learning_rate: 0.05
       l2_leaf_reg: 3.0         # L2 regularization
       bagging_temperature: 1.0 # Bayesian bootstrap intensity
       random_strength: 1.0     # Randomness for scoring splits
       border_count: 128        # Feature discretization

**Key Features**:

- Ordered boosting reduces overfitting
- Native categorical feature handling
- GPU acceleration available
- Automatic handling of missing values

Ensemble Training
-----------------

Classical models support ensemble training across splits/folds:

.. code-block:: yaml

   data:
     data_dir: "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"
     splits: null    # Use all splits
     folds: null     # Use all folds

   model:
     type: lightgbm
     # ... model config

   ensemble:
     enabled: true
     max_parallel: 5  # Train 5 models simultaneously

   mlflow:
     nested: true     # Log each fold as child run

Train ensemble:

.. code-block:: bash

   admet model ensemble -c configs/4-more-models/ensemble_lightgbm.yaml --max-parallel 5

Hyperparameter Optimization
----------------------------

Classical models support Ray Tune HPO:

.. code-block:: yaml

   hpo:
     enabled: true
     num_samples: 50
     max_concurrent: 4
     scheduler: "asha"
     metric: "val_loss"
     mode: "min"
     search_space:
       model.xgboost.n_estimators:
         type: "randint"
         lower: 100
         upper: 1000
       model.xgboost.max_depth:
         type: "randint"
         lower: 4
         upper: 12
       model.xgboost.learning_rate:
         type: "loguniform"
         lower: 0.001
         upper: 0.3

Run HPO:

.. code-block:: bash

   admet model hpo -c configs/1-hpo-single/hpo_xgboost.yaml --num-samples 50

Best Practices
--------------

1. **Start with Morgan Fingerprints**

   Morgan (radius=2, n_bits=2048) is a strong default. Try MACCS or Mordred
   if performance is underwhelming.

2. **Use Appropriate Regularization**

   Classical models can overfit. Use ``reg_alpha``/``reg_lambda`` (XGBoost/LightGBM)
   or ``l2_leaf_reg`` (CatBoost) to prevent this.

3. **Tune n_estimators with Early Stopping**

   Use validation loss to determine optimal number of trees. Monitor training
   vs validation performance.

4. **Leverage Parallelism**

   All three models support multi-threading. Use ``n_jobs=-1`` or similar
   to utilize all CPU cores.

5. **Compare Multiple Models**

   XGBoost, LightGBM, and CatBoost often have different strengths.
   Train all three and ensemble their predictions.

6. **Use Ensemble Training**

   Train across multiple splits/folds for robust predictions and
   uncertainty quantification.

Performance Comparison
----------------------

Typical performance characteristics:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - Training Speed
     - Memory Usage
     - Accuracy
     - Tuning Difficulty
   * - XGBoost
     - Fast
     - Medium
     - High
     - Medium
   * - LightGBM
     - Very Fast
     - Low
     - High
     - Medium
   * - CatBoost
     - Medium
     - Medium
     - Very High
     - Low

**When to Use Each**:

- **XGBoost**: Default choice, well-documented, stable
- **LightGBM**: Large datasets (>10k samples), speed critical
- **CatBoost**: Categorical features, minimal tuning needed

Limitations
-----------

Classical models have some limitations compared to neural models:

- **No curriculum learning**: Cannot use quality-aware progressive training
- **No task affinity**: Cannot leverage gradient-based task grouping
- **Fixed features**: Fingerprints don't adapt during training
- **No joint sampling**: Cannot use task oversampling strategies

For these advanced features, use Chemprop or Chemeleon models.

Example Configs
---------------

Complete example configurations are available in ``configs/4-more-models/``:

- ``xgboost.yaml``: XGBoost with Morgan fingerprints
- ``lightgbm.yaml``: LightGBM with MACCS keys
- ``catboost.yaml``: CatBoost with Mordred descriptors
- ``ensemble_xgboost.yaml``: Ensemble training across folds

Related Documentation
---------------------

- :doc:`configuration`: Complete configuration reference
- :doc:`cli`: Command-line interface for training
- :doc:`hpo`: Hyperparameter optimization guide
- :doc:`../api/admet.features`: Fingerprint generation API
- :doc:`../api/admet.model`: Model API documentation
