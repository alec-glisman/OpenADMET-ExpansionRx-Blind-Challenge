Model Package (``admet.model``)
===============================

The ``admet.model`` package provides model implementations for ADMET property
prediction. It currently supports two main model families:

- **Chemprop Models** (``admet.model.chemprop``): Message-passing neural networks
  using the Chemprop library with PyTorch Lightning integration.
- **Classical Models** (``admet.model.classical``): Traditional machine learning
  models like XGBoost and LightGBM.

.. automodule:: admet.model
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

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

.. autoclass:: admet.model.chemprop.ChempropEnsemble
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
machine learning models.

.. automodule:: admet.model.classical
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
