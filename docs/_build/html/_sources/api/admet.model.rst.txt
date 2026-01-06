Model Package (``admet.model``)
===============================

The ``admet.model`` package provides model implementations for ADMET property
prediction. It supports three main model families:

- **Chemprop Models** (``admet.model.chemprop``): Message-passing neural networks
  using the Chemprop library with PyTorch Lightning integration.
- **Chemeleon Models** (``admet.model.chemeleon``): Pre-trained molecular encoder
  with transfer learning for property prediction.
- **Classical Models** (``admet.model.classical``): Traditional machine learning
  models like XGBoost, LightGBM, and CatBoost.

.. automodule:: admet.model
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Model Registry
--------------

The ModelRegistry provides a factory pattern for creating model instances
from configuration files.

.. autoclass:: admet.model.registry.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from admet.model.registry import ModelRegistry
   from omegaconf import OmegaConf

   # Load configuration
   config = OmegaConf.load("config.yaml")

   # Create model based on config.model.type
   model = ModelRegistry.create(config)

   # Training API depends on model type:
   # - Chemprop: model.fit()  # Data loaded from config.data
   # - Chemeleon/Classical: model.fit(smiles, targets, val_smiles, val_targets)

   # Example for Chemprop (data in config)
   if config.model.type == "chemprop":
       model.fit()  # Loads data from config.data.data_dir
   else:
       # For classical/chemeleon models, provide data explicitly
       model.fit(train_smiles, train_targets, val_smiles, val_targets)

   predictions = model.predict(test_smiles)

Chemprop Subpackage
-------------------

The ``admet.model.chemprop`` subpackage provides a comprehensive toolkit for
training Chemprop MPNN models with advanced features including:

- **Configuration Management**: OmegaConf-based configuration for reproducible experiments
- **Ensemble Training**: Train multiple models across splits/folds with Ray parallelization
- **Hyperparameter Optimization**: Ray Tune integration with ASHA scheduler
- **Curriculum Learning**: Quality-aware training with adaptive data weighting
- **Custom FFN Architectures**: Mixture of Experts and Branched FFN options
- **MLflow Integration**: Full experiment tracking with nested runs

Key Classes
^^^^^^^^^^^

.. autoclass:: admet.model.chemprop.ChempropModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.ModelEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.ChempropHPO
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: admet.model.chemprop.ChempropConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.EnsembleConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.HPOConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.DataConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.OptimizationConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.MlflowConfig
   :members:
   :undoc-members:
   :show-inheritance:

Curriculum Learning
^^^^^^^^^^^^^^^^^^^

.. autoclass:: admet.model.chemprop.curriculum.CurriculumState
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.curriculum.CurriculumCallback
   :members:
   :undoc-members:
   :show-inheritance:

Custom FFN Architectures
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: admet.model.chemprop.ffn.MixtureOfExpertsRegressionFFN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.chemprop.ffn.BranchedFFN
   :members:
   :undoc-members:
   :show-inheritance:

Classical ML Subpackage
-----------------------

The ``admet.model.classical`` subpackage provides wrappers for traditional
machine learning models that use molecular fingerprints as features.

.. automodule:: admet.model.classical
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

XGBoost Model
^^^^^^^^^^^^^

Gradient boosting implementation using XGBoost.

.. autoclass:: admet.model.classical.XGBoostModel
   :members:
   :undoc-members:
   :show-inheritance:

**Key Features:**

- Gradient boosting with highly-tuned hyperparameters
- Supports multi-output regression via MultiOutputRegressor
- Configurable tree depth, learning rate, and regularization
- Fast training and prediction with parallel tree construction

LightGBM Model
^^^^^^^^^^^^^^

Fast gradient boosting using LightGBM.

.. autoclass:: admet.model.classical.LightGBMModel
   :members:
   :undoc-members:
   :show-inheritance:

**Key Features:**

- Histogram-based gradient boosting for speed
- Lower memory usage than XGBoost
- Categorical feature support
- Built-in handling of missing values

CatBoost Model
^^^^^^^^^^^^^^

Gradient boosting with CatBoost library.

.. autoclass:: admet.model.classical.CatBoostModel
   :members:
   :undoc-members:
   :show-inheritance:

**Key Features:**

- Ordered boosting to reduce overfitting
- Native categorical feature handling
- Built-in early stopping
- GPU acceleration support

Chemeleon Subpackage
--------------------

The ``admet.model.chemeleon`` subpackage provides pre-trained molecular
encoder models for transfer learning.

.. automodule:: admet.model.chemeleon
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ChemeleonModel
^^^^^^^^^^^^^^

Pre-trained Chemeleon encoder with frozen/unfrozen transfer learning.

.. autoclass:: admet.model.chemeleon.ChemeleonModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

**Key Features:**

- Auto-download of pre-trained weights from Zenodo
- Frozen encoder by default for efficient transfer learning
- Optional gradual unfreezing schedule
- Consistent BaseModel interface
- Curriculum learning support
- Task affinity grouping

**Example Usage:**

.. code-block:: python

   from admet.model.registry import ModelRegistry
   from omegaconf import OmegaConf

   config = OmegaConf.create({
       "model": {
           "type": "chemeleon",
           "chemeleon": {
               "freeze_encoder": True,
               "unfreeze_schedule": None,
           },
           "ffn": {"type": "regression", "hidden_dim": 1200},
       },
       "mlflow": {"enabled": False},
   })

   model = ModelRegistry.create(config)
   model.fit(smiles_train, targets_train)
   predictions = model.predict(smiles_test)
