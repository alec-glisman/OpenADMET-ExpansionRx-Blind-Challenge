Architecture Overview
======================

This page gives you a high-level mental model of the ``admet`` package: how
modules are layered, how data flows through the system, and where you can
extend functionality.

Layered Package Map
-------------------

The codebase is organized into clear functional layers:

1. **Data Layer** (``admet.data``): Data loading, chemical feature extraction, and
   cluster-based dataset splitting with quality-aware stratification.

2. **Modeling Layer** (``admet.model``): Model implementations organized by type:

   - ``admet.model.chemprop``: Chemprop MPNN models with ensemble training,
     hyperparameter optimization, and curriculum learning support.
   - ``admet.model.classical``: Traditional ML models (XGBoost, LightGBM).

3. **Visualization Layer** (``admet.plot``): Performance figures, parity plots,
   metric visualizations, and dataset exploration plots.

4. **Utilities** (``admet.util``): Logging configuration and helper functions.

Data Flow Lifecycle
-------------------

The typical end-to-end pipeline follows these stages:

.. code-block:: text

   raw data --> standardization / cleaning --> SMILES canonicalization
             --> cluster-based splitting (BitBirch + stratification)
             --> model training (Chemprop + JointSampler)
             --> ensemble predictions (with uncertainty)
             --> evaluation (metrics) --> visualization

Key Modules and Responsibilities
--------------------------------

Data Layer
^^^^^^^^^^

- ``admet.data.smiles``: SMILES processing and canonicalization
- ``admet.data.fingerprint``: Molecular fingerprint generation (Morgan, ECFP)
- ``admet.data.property``: Property calculation and feature engineering
- ``admet.data.split``: Cluster-aware splitting with BitBirch and stratification
- ``admet.data.constant``: Dataset constants and transformation helpers
- ``admet.data.stats``: Statistical analysis utilities

Modeling Layer
^^^^^^^^^^^^^^

- ``admet.model.chemprop.model``: ``ChempropModel`` class for single model training
- ``admet.model.chemprop.ensemble``: ``ModelEnsemble`` for multi-fold training
- ``admet.model.chemprop.hpo``: ``ChempropHPO`` for hyperparameter optimization
- ``admet.model.chemprop.joint_sampler``: ``JointSampler`` for unified two-stage sampling
- ``admet.model.chemprop.curriculum``: Curriculum state and callbacks
- ``admet.model.chemprop.ffn``: Custom FFN architectures (MoE, Branched)
- ``admet.model.chemprop.config``: OmegaConf dataclass configurations
- ``admet.model.classical``: Classical ML model wrappers

Visualization Layer
^^^^^^^^^^^^^^^^^^^

- ``admet.plot.parity``: Parity plots for regression evaluation
- ``admet.plot.metrics``: Metric computation and bar chart visualization
- ``admet.plot.density``: Distribution and density plots
- ``admet.plot.heatmap``: Correlation heatmaps
- ``admet.plot.split``: Dataset split diagnostics
- ``admet.plot.latex``: LaTeX-safe formatting for publications

Configuration System
--------------------

The package uses OmegaConf dataclasses for configuration management:

.. code-block:: python

   from admet.model.chemprop import ChempropConfig, EnsembleConfig
   from omegaconf import OmegaConf

   # Load from YAML
   config = OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
   cfg = OmegaConf.structured(EnsembleConfig(**config))

   # Access nested configuration
   print(cfg.model.ffn_type)  # "regression"
   print(cfg.data.smiles_col)  # "SMILES"

Key configuration classes:

- ``ChempropConfig``: Single model configuration
- ``EnsembleConfig``: Multi-model ensemble settings
- ``HPOConfig``: Hyperparameter optimization settings
- ``DataConfig``: Dataset paths and column names
- ``ModelConfig``: Model architecture parameters
- ``OptimizationConfig``: Training hyperparameters
- ``JointSamplingConfig``: Unified task + curriculum sampling settings
- ``MlflowConfig``: Experiment tracking settings

Extensibility Points
--------------------

You can add new functionality without modifying core internals by:

- **New Splitting Strategy**: Add a function in ``admet.data.split`` implementing
  the cluster/stratification logic.
- **Custom FFN Architecture**: Implement in ``admet.model.chemprop.ffn`` following
  the existing MoE/Branched patterns.
- **New Visualization**: Add plotting functions in ``admet.plot`` and expose
  via ``__init__.py``.
- **Additional Metrics**: Extend ``admet.plot.metrics`` with new metric functions.

Design Principles
-----------------

- **Separation of Concerns**: Data preparation, modeling, and visualization isolated.
- **Configuration-Driven**: All hyperparameters externalized to YAML files.
- **Reproducibility**: Deterministic seeds, cluster-based splitting, MLflow tracking.
- **Parallelization**: Ray-based distributed training and HPO.
- **Quality Awareness**: Curriculum learning and stratification by quality tiers.

Cross-References
----------------

- See :doc:`data_sources` for input provenance and curation
- See :doc:`splitting` for dataset partitioning methodology
- See :doc:`modeling` for training details and model usage
- See :doc:`configuration` for configuration file format
