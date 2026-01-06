<!-- markdownlint-disable-file -->

# Task Details: Sphinx Documentation Update for v1 API

## Research Reference

**Source Research**: #file:../research/20260105-sphinx-docs-v1-update-research.md

---

## Phase 1: Fix Build Errors

### Task 1.1: Add orphaned documents to index.rst toctree

Add `guide/logging` and `guide/debugging_per_quality_metrics` to the appropriate toctree in `docs/index.rst`.

- **Files**:
  - `docs/index.rst` - Add entries to Guides toctree section
- **Success**:
  - No "document isn't included in any toctree" warnings
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 85-90) - Orphaned documents analysis
- **Dependencies**:
  - None

**Implementation**:

```restructuredtext
.. toctree::
   :maxdepth: 1
   :caption: Guides

   guide/overview
   guide/leaderboard
   guide/development
   guide/architecture
   guide/data_sources
   guide/splitting
   guide/configuration
   guide/config_reference
   guide/modeling
   guide/hpo
   guide/curriculum
   guide/task_affinity
   guide/mlflow_artifacts
   guide/profiling
   guide/performance_optimization
   guide/logging
   guide/debugging_per_quality_metrics
   planning/index
```

### Task 1.2: Fix RST formatting in logging.rst

Fix the title level inconsistency at line 514 where `Test Coverage` uses `~~~` but should use `^^^` based on section hierarchy.

- **Files**:
  - `docs/guide/logging.rst` - Fix title underline at line 514
- **Success**:
  - No "Title level inconsistent" warning
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 73-75) - RST formatting error
- **Dependencies**:
  - None

**Implementation**:

Change line 514-515 from:

```restructuredtext
Test Coverage
~~~~~~~~~~~~~
```

To:

```restructuredtext
Test Coverage
^^^^^^^^^^^^^
```

### Task 1.3: Fix broken references in logging.rst

Replace `:ref:` directives with `:doc:` directives for cross-document references at lines 527-529.

- **Files**:
  - `docs/guide/logging.rst` - Fix lines 527-529
- **Success**:
  - No "undefined label" warnings
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 76-80) - Broken refs analysis
- **Dependencies**:
  - Task 1.2 completion

**Implementation**:

Change lines 527-529 from:

```restructuredtext
- :ref:`configuration`: Complete configuration reference
- :ref:`cli`: Command-line interface documentation
- :ref:`hpo`: Hyperparameter optimization guide
```

To:

```restructuredtext
- :doc:`configuration`: Complete configuration reference
- :doc:`cli`: Command-line interface documentation
- :doc:`hpo`: Hyperparameter optimization guide
```

### Task 1.4: Verify build completes with 0 warnings

Run clean Sphinx build and verify no warnings.

- **Files**:
  - None (verification only)
- **Success**:
  - `rm -rf docs/_build && make -C docs html` shows "build succeeded" with 0 warnings
- **Dependencies**:
  - Tasks 1.1, 1.2, 1.3 completion

---

## Phase 2: Update Version and Core API Docs

### Task 2.1: Update version in docs/api/admet.rst

Update version example from `0.0.1` to `1.2.0`.

- **Files**:
  - `docs/api/admet.rst` - Update version in code block
- **Success**:
  - Version shows `1.2.0`
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 30-32) - Version mismatch
- **Dependencies**:
  - Phase 1 completion

**Implementation**:

Change:

```python
import admet
print(admet.__version__)  # "0.0.1"
```

To:

```python
import admet
print(admet.__version__)  # "1.2.0"
```

### Task 2.2: Create docs/api/admet.features.rst

Create new API documentation file for the `admet.features` subpackage.

- **Files**:
  - `docs/api/admet.features.rst` - New file
- **Success**:
  - File exists and autodoc generates content
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 50-52) - Missing features docs
  - `src/admet/features/__init__.py` - Module exports
  - `src/admet/features/fingerprints.py` - FingerprintGenerator class
- **Dependencies**:
  - Task 2.1 completion

**Implementation**:

```restructuredtext
Feature Generation (``admet.features``)
=======================================

The ``admet.features`` package provides molecular fingerprint and descriptor
generation utilities for use with classical machine learning models.

.. automodule:: admet.features
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Fingerprint Generator
---------------------

The ``FingerprintGenerator`` class provides a unified interface for generating
various types of molecular fingerprints from SMILES strings.

.. autoclass:: admet.features.FingerprintGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Supported Fingerprint Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Morgan**: Circular fingerprints (ECFP-like), configurable radius and bits
- **RDKit**: Path-based topological fingerprints
- **MACCS**: Fixed 167-bit structural keys
- **Mordred**: Comprehensive molecular descriptors (~1800 features)

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from admet.features import FingerprintGenerator
   from admet.model.config import FingerprintConfig

   # Create Morgan fingerprint generator
   config = FingerprintConfig(
       type="morgan",
       morgan={"radius": 2, "n_bits": 2048}
   )
   generator = FingerprintGenerator(config)

   # Generate fingerprints for SMILES list
   smiles = ["CCO", "CCCO", "c1ccccc1"]
   features = generator.generate(smiles)
   print(features.shape)  # (3, 2048)

Configuration Classes
^^^^^^^^^^^^^^^^^^^^^

Fingerprint configuration is specified via ``FingerprintConfig`` from
``admet.model.config``:

.. autoclass:: admet.model.config.FingerprintConfig
   :members:
   :noindex:

.. autoclass:: admet.model.config.MorganFingerprintConfig
   :members:
   :noindex:

.. autoclass:: admet.model.config.RDKitFingerprintConfig
   :members:
   :noindex:

.. autoclass:: admet.model.config.MACCSConfig
   :members:
   :noindex:

.. autoclass:: admet.model.config.MordredConfig
   :members:
   :noindex:
```

### Task 2.3: Update docs/api/admet.util.rst with missing modules

Add documentation for `profiling` and `ray_logging` modules.

- **Files**:
  - `docs/api/admet.util.rst` - Add new module sections
- **Success**:
  - All util modules documented
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 53-56) - Missing util modules
  - `src/admet/util/profiling.py` - Profiling utilities
  - `src/admet/util/ray_logging.py` - Ray logging utilities
- **Dependencies**:
  - Task 2.2 completion

**Implementation** - Add after existing modules:

```restructuredtext
profiling
^^^^^^^^^

Performance profiling utilities for model training.

.. automodule:: admet.util.profiling
   :members:
   :undoc-members:
   :show-inheritance:

ray_logging
^^^^^^^^^^^

Ray Tune and ensemble logging management utilities.

.. automodule:: admet.util.ray_logging
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

- ``RayLogManager``: Manages log collection, compression, and MLflow upload
- ``LogConfig``: Configuration for Ray logging behavior
```

### Task 2.4: Add admet.features to docs/api/admet.rst toctree

Update the toctree and subpackage list in `docs/api/admet.rst`.

- **Files**:
  - `docs/api/admet.rst` - Add features to toctree and description
- **Success**:
  - Features subpackage listed and linked
- **Dependencies**:
  - Task 2.2 completion

**Implementation**:

Update subpackage list:

```restructuredtext
- :doc:`admet.data` - Data loading, chemistry utilities, and dataset splitting
- :doc:`admet.features` - Molecular fingerprint and descriptor generation
- :doc:`admet.model` - Model implementations (Chemprop, Chemeleon, Classical ML)
- :doc:`admet.plot` - Visualization utilities for plots and figures
- :doc:`admet.util` - Utility functions and logging configuration

.. toctree::
   :maxdepth: 1

   admet.data
   admet.features
   admet.model
   admet.plot
   admet.util
```

---

## Phase 3: Update Model API Documentation

### Task 3.1: Update docs/api/admet.model.rst with current classes

Update the model API documentation to reflect current class structure including `ModelRegistry`, `Ensemble`, and unified config classes.

- **Files**:
  - `docs/api/admet.model.rst` - Major update
- **Success**:
  - All exported classes documented
  - Current API patterns shown
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 40-48) - Model exports
  - `src/admet/model/__init__.py` - Current exports
- **Dependencies**:
  - Phase 2 completion

**Implementation** - Replace intro section:

```restructuredtext
Model Package (``admet.model``)
===============================

The ``admet.model`` package provides model implementations for ADMET property
prediction. It supports multiple model families through a unified configuration
and registry system.

**Model Types:**

- **Chemprop** (``admet.model.chemprop``): Message-passing neural networks
- **Chemeleon** (``admet.model.chemeleon``): Pre-trained foundation model
- **Classical** (``admet.model.classical``): XGBoost, LightGBM, CatBoost

.. automodule:: admet.model
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Model Registry
--------------

The ``ModelRegistry`` provides a factory pattern for creating models from
unified configuration:

.. autoclass:: admet.model.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

**Usage:**

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   # Load configuration
   config = OmegaConf.load("configs/4-more-models/chemprop.yaml")

   # Create model via registry
   model = ModelRegistry.create(config)
   model.fit(train_smiles, train_targets)

Base Classes
------------

.. autoclass:: admet.model.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.Ensemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.MLflowMixin
   :members:
   :undoc-members:
   :show-inheritance:
```

### Task 3.2: Add ModelRegistry documentation

Expand ModelRegistry section with detailed usage patterns.

- **Files**:
  - `docs/api/admet.model.rst` - Add after base classes section
- **Success**:
  - Registry pattern fully documented
- **Research References**:
  - `src/admet/model/registry.py` - Registry implementation
  - `src/admet/cli/model.py` - CLI usage pattern
- **Dependencies**:
  - Task 3.1 completion

**Implementation** - Add extended example:

```restructuredtext
Creating Models via Registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The registry automatically selects the correct model class based on
``config.model.type``:

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   # Chemprop model
   chemprop_config = OmegaConf.load("configs/chemprop.yaml")
   chemprop_model = ModelRegistry.create(chemprop_config)

   # XGBoost model
   xgb_config = OmegaConf.load("configs/xgboost.yaml")
   xgb_model = ModelRegistry.create(xgb_config)

   # Both follow the same interface
   for model, config in [(chemprop_model, chemprop_config), (xgb_model, xgb_config)]:
       model.fit(train_smiles, train_targets)
       predictions = model.predict(test_smiles)

Registering Custom Models
^^^^^^^^^^^^^^^^^^^^^^^^^

Custom models can be registered with the registry:

.. code-block:: python

   from admet.model import ModelRegistry, BaseModel

   @ModelRegistry.register("my_model")
   class MyCustomModel(BaseModel):
       def fit(self, smiles, targets, **kwargs):
           ...

       def predict(self, smiles):
           ...
```

### Task 3.3: Document Chemeleon subpackage

Add comprehensive Chemeleon documentation section.

- **Files**:
  - `docs/api/admet.model.rst` - Add Chemeleon section
- **Success**:
  - Chemeleon model and config documented
- **Research References**:
  - `src/admet/model/chemeleon/__init__.py` - Module exports
  - `src/admet/model/config.py` - ChemeleonModelParams
- **Dependencies**:
  - Task 3.2 completion

**Implementation**:

```restructuredtext
Chemeleon Subpackage
--------------------

The ``admet.model.chemeleon`` subpackage provides integration with the
Chemeleon pre-trained molecular encoder.

**Features:**

- Pre-trained encoder from large-scale molecular data
- Frozen encoder with trainable FFN head
- Supports all FFN architectures (regression, MoE, branched)
- Automatic checkpoint download from Zenodo

.. autoclass:: admet.model.chemeleon.ChemeleonModel
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
^^^^^^^^^^^^^

.. autoclass:: admet.model.config.ChemeleonModelParams
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.config.UnfreezeScheduleConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.config.ChemeleonHeadConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example
^^^^^^^

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   config = OmegaConf.create({
       "model": {
           "type": "chemeleon",
           "chemeleon": {
               "checkpoint_path": "auto",  # Download from Zenodo
               "head": {"hidden_dims": [256, 128], "dropout": 0.2},
               "unfreeze_schedule": {"freeze_encoder": True}
           }
       },
       "data": {
           "smiles_col": "SMILES",
           "target_cols": ["LogD", "Log KSOL"]
       }
   })

   model = ModelRegistry.create(config)
   model.fit(train_smiles, train_targets)
```

### Task 3.4: Document classical models API (XGBoost, LightGBM, CatBoost)

Add classical models section to admet.model.rst.

- **Files**:
  - `docs/api/admet.model.rst` - Add Classical section
- **Success**:
  - All classical model classes documented
- **Research References**:
  - `src/admet/model/classical/__init__.py` - Module exports
  - `src/admet/model/config.py` - XGBoostModelParams, LightGBMModelParams, CatBoostModelParams
- **Dependencies**:
  - Task 3.3 completion

**Implementation**:

```restructuredtext
Classical Models Subpackage
---------------------------

The ``admet.model.classical`` subpackage provides wrappers for traditional
gradient boosting models that use molecular fingerprints as input features.

**Supported Models:**

- **XGBoost**: Extreme Gradient Boosting
- **LightGBM**: Light Gradient Boosting Machine
- **CatBoost**: Categorical Boosting

All classical models require a ``FingerprintConfig`` to generate feature vectors
from SMILES strings.

.. automodule:: admet.model.classical
   :members:
   :undoc-members:
   :show-inheritance:

XGBoost Model
^^^^^^^^^^^^^

.. autoclass:: admet.model.classical.XGBoostModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.config.XGBoostModelParams
   :members:
   :noindex:

LightGBM Model
^^^^^^^^^^^^^^

.. autoclass:: admet.model.classical.LightGBMModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.config.LightGBMModelParams
   :members:
   :noindex:

CatBoost Model
^^^^^^^^^^^^^^

.. autoclass:: admet.model.classical.CatBoostModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: admet.model.config.CatBoostModelParams
   :members:
   :noindex:
```

---

## Phase 4: Configuration Documentation

### Task 4.1: Update docs/guide/configuration.rst with UnifiedModelConfig

Expand configuration guide to document the unified config system.

- **Files**:
  - `docs/guide/configuration.rst` - Major expansion
- **Success**:
  - UnifiedModelConfig pattern explained
  - Model type discriminator documented
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 60-70) - Config system
  - `src/admet/model/config.py` - UnifiedModelConfig definition
- **Dependencies**:
  - Phase 3 completion

**Implementation** - Add new section after "Configuration System":

```restructuredtext
Unified Model Configuration
---------------------------

All models use a unified configuration schema with a ``model.type`` discriminator
field that determines which model-specific parameters apply.

**Configuration Structure:**

.. code-block:: yaml

   # Data configuration (shared by all models)
   data:
     data_dir: "path/to/data"
     smiles_col: "SMILES"
     target_cols: ["LogD", "Log KSOL"]

   # Model configuration with type discriminator
   model:
     type: chemprop  # One of: chemprop, chemeleon, xgboost, lightgbm, catboost

     # Model-specific parameters (only the matching section is used)
     chemprop:
       depth: 5
       message_hidden_dim: 600
       # ... chemprop-specific params

     xgboost:
       n_estimators: 500
       max_depth: 8
       # ... xgboost-specific params

   # Optimization (shared structure, model-specific defaults)
   optimization:
     max_epochs: 150
     batch_size: 32

   # MLflow tracking (shared by all models)
   mlflow:
     enabled: true
     experiment_name: "my_experiment"

**Model Type Selection:**

The ``model.type`` field accepts:

- ``chemprop``: Chemprop MPNN (PyTorch)
- ``chemeleon``: Pre-trained encoder (PyTorch)
- ``xgboost``: XGBoost gradient boosting
- ``lightgbm``: LightGBM gradient boosting
- ``catboost``: CatBoost gradient boosting

**Loading Configuration:**

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   # Load from YAML
   config = OmegaConf.load("configs/my_model.yaml")

   # Create model via registry (uses model.type to select class)
   model = ModelRegistry.create(config)
```

### Task 4.2: Update docs/guide/config_reference.rst with complete schema

Create comprehensive config reference documentation.

- **Files**:
  - `docs/guide/config_reference.rst` - Full schema documentation
- **Success**:
  - All config fields documented with types and defaults
- **Research References**:
  - `src/admet/model/config.py` - Full config definitions
  - `src/admet/model/chemprop/config.py` - Chemprop-specific config
- **Dependencies**:
  - Task 4.1 completion

**Implementation** - Major sections to add:

```restructuredtext
Configuration Reference
=======================

This page provides a complete reference for all configuration options.

Base Configuration Classes
--------------------------

BaseDataConfig
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``data_dir``
     - str
     - REQUIRED
     - Directory containing train.csv and validation.csv
   * - ``test_file``
     - str
     - None
     - Path to test data CSV
   * - ``blind_file``
     - str
     - None
     - Path to blind test data CSV (no labels)
   * - ``smiles_col``
     - str
     - "SMILES"
     - Column name for SMILES strings
   * - ``target_cols``
     - List[str]
     - []
     - Target column names for prediction
   * - ``target_weights``
     - List[float]
     - []
     - Per-task loss weights (equal if empty)
   * - ``output_dir``
     - str
     - None
     - Output directory for checkpoints

BaseMlflowConfig
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - True
     - Enable MLflow tracking
   * - ``tracking_uri``
     - str
     - None
     - MLflow server URI
   * - ``experiment_name``
     - str
     - "admet"
     - Experiment name
   * - ``run_name``
     - str
     - None
     - Optional run name
   * - ``nested``
     - bool
     - False
     - Create nested run (for ensembles)
   * - ``log_model``
     - bool
     - True
     - Log model artifacts
   * - ``log_predictions``
     - bool
     - True
     - Log prediction CSVs
   * - ``compress_artifacts``
     - bool
     - True
     - Compress artifacts (50-80% size reduction)

Chemprop Configuration
----------------------

ChempropModelParams
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``depth``
     - int
     - 5
     - Message passing iterations
   * - ``message_hidden_dim``
     - int
     - 600
     - MPNN hidden dimension
   * - ``dropout``
     - float
     - 0.1
     - Dropout probability
   * - ``num_layers``
     - int
     - 2
     - FFN layers
   * - ``hidden_dim``
     - int
     - 600
     - FFN hidden dimension
   * - ``batch_norm``
     - bool
     - True
     - Use batch normalization
   * - ``ffn_type``
     - str
     - "regression"
     - FFN architecture: regression, mixture_of_experts, branched
   * - ``aggregation``
     - str
     - "mean"
     - Aggregation method: mean, sum, norm

[Continue with all model types...]
```

### Task 4.3: Update config loading examples across all guides

Update programmatic config loading examples to use current patterns.

- **Files**:
  - Multiple guide files
- **Success**:
  - All examples use `ModelRegistry.create()` pattern
- **Dependencies**:
  - Tasks 4.1, 4.2 completion

**Pattern to apply everywhere:**

Old pattern:

```python
from admet.model.chemprop import ChempropModel, ChempropConfig
config = OmegaConf.structured(ChempropConfig(**yaml_config))
model = ChempropModel.from_config(config)
```

New pattern:

```python
from omegaconf import OmegaConf
from admet.model import ModelRegistry

config = OmegaConf.load("configs/my_config.yaml")
model = ModelRegistry.create(config)
```

---

## Phase 5: Create Classical Models Guide

### Task 5.1: Create docs/guide/classical_models.rst

Create comprehensive guide for classical models with usage examples.

- **Files**:
  - `docs/guide/classical_models.rst` - New file
- **Success**:
  - Guide covers all three classical model types
  - Includes fingerprint configuration
  - Shows complete training workflow
- **Research References**:
  - `src/admet/model/classical/` - Implementation
  - `src/admet/features/fingerprints.py` - FingerprintGenerator
- **Dependencies**:
  - Phase 4 completion

**Implementation**:

```restructuredtext
Classical Models Guide
======================

This guide covers training classical machine learning models (XGBoost, LightGBM,
CatBoost) for ADMET property prediction using molecular fingerprints.

Overview
--------

Classical models provide fast training and inference with competitive performance
for many ADMET prediction tasks. Unlike graph neural networks, they require
fixed-length feature vectors generated from molecular fingerprints.

**Advantages:**

- Fast training (minutes vs hours)
- No GPU required
- Interpretable feature importance
- Robust to hyperparameter choices

**When to use:**

- Quick baseline models
- Limited computational resources
- Need for feature importance analysis
- Tabular data with pre-computed descriptors

Fingerprint Configuration
-------------------------

Classical models require fingerprint configuration to generate features from SMILES:

.. code-block:: yaml

   model:
     type: xgboost
     fingerprint:
       type: morgan  # morgan, rdkit, maccs, mordred
       morgan:
         radius: 2
         n_bits: 2048
         use_chirality: false

**Fingerprint Types:**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Type
     - Dimensions
     - Description
   * - morgan
     - Configurable (default 2048)
     - Circular fingerprints (ECFP-like), best general-purpose choice
   * - rdkit
     - Configurable (default 2048)
     - Path-based topological fingerprints
   * - maccs
     - 167 (fixed)
     - Structural keys, fast but lower dimensionality
   * - mordred
     - ~1800
     - Comprehensive molecular descriptors, highest dimensionality

XGBoost Training
----------------

**Configuration (configs/xgboost_example.yaml):**

.. code-block:: yaml

   data:
     data_dir: "assets/dataset/split_train_val/v3/split_0/fold_0"
     smiles_col: "SMILES"
     target_cols:
       - "LogD"
       - "Log KSOL"

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
       n_jobs: -1  # Use all cores

   mlflow:
     enabled: true
     experiment_name: "xgboost_admet"

**Training via CLI:**

.. code-block:: bash

   admet model train --config configs/xgboost_example.yaml

**Training via Python:**

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   config = OmegaConf.load("configs/xgboost_example.yaml")
   model = ModelRegistry.create(config)

   # Load data
   import pandas as pd
   train_df = pd.read_csv("train.csv")
   val_df = pd.read_csv("validation.csv")

   # Train
   model.fit(
       train_df["SMILES"].tolist(),
       train_df[["LogD", "Log KSOL"]].values,
       val_smiles=val_df["SMILES"].tolist(),
       val_y=val_df[["LogD", "Log KSOL"]].values,
   )

   # Predict
   predictions = model.predict(val_df["SMILES"].tolist())

LightGBM Training
-----------------

**Configuration:**

.. code-block:: yaml

   model:
     type: lightgbm
     fingerprint:
       type: morgan
       morgan:
         radius: 2
         n_bits: 2048
     lightgbm:
       n_estimators: 500
       max_depth: 8
       learning_rate: 0.05
       num_leaves: 31
       subsample: 0.8
       colsample_bytree: 0.8
       reg_alpha: 0.1
       reg_lambda: 1.0
       n_jobs: -1

**Key LightGBM Parameters:**

- ``num_leaves``: Controls tree complexity (default 31)
- ``min_child_samples``: Minimum data in leaf (regularization)
- ``boosting_type``: "gbdt" (default), "dart", "goss"

CatBoost Training
-----------------

**Configuration:**

.. code-block:: yaml

   model:
     type: catboost
     fingerprint:
       type: morgan
       morgan:
         radius: 2
         n_bits: 2048
     catboost:
       iterations: 500
       depth: 8
       learning_rate: 0.05
       l2_leaf_reg: 3.0
       random_strength: 1.0
       bagging_temperature: 1.0
       verbose: false

**CatBoost Advantages:**

- Native handling of categorical features
- Ordered boosting reduces overfitting
- GPU training support

Multi-Task Training
-------------------

All classical models support multi-task prediction by training separate models
per target internally:

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model import ModelRegistry

   config = OmegaConf.create({
       "model": {
           "type": "xgboost",
           "fingerprint": {"type": "morgan"},
           "xgboost": {"n_estimators": 500}
       },
       "data": {
           "target_cols": ["LogD", "Log KSOL", "Log HLM CLint"]
       }
   })

   model = ModelRegistry.create(config)

   # Train on all 3 targets (trains 3 internal XGBoost models)
   model.fit(smiles, targets)  # targets shape: (n_samples, 3)

   # Predict all targets
   predictions = model.predict(test_smiles)  # shape: (n_test, 3)

Hyperparameter Tuning
---------------------

For classical models, use scikit-learn's cross-validation utilities:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   import xgboost as xgb

   # Create XGBoost regressor directly for tuning
   param_grid = {
       "max_depth": [4, 6, 8],
       "n_estimators": [100, 300, 500],
       "learning_rate": [0.01, 0.05, 0.1],
   }

   xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
   grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="neg_mae")
   grid_search.fit(fingerprints, targets)

   print(f"Best params: {grid_search.best_params_}")

Comparison with Neural Networks
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Classical Models
     - Chemprop MPNN
   * - Training Time
     - Minutes
     - Hours
   * - GPU Required
     - No
     - Recommended
   * - Feature Engineering
     - Required (fingerprints)
     - Automatic (graph)
   * - Performance
     - Good baseline
     - State-of-the-art
   * - Interpretability
     - Feature importance
     - Attention weights
   * - Multi-task Learning
     - Separate models
     - Shared representation

See Also
--------

- :doc:`modeling` - Chemprop neural network training
- :doc:`configuration` - Configuration file format
- :doc:`/api/admet.features` - Fingerprint generation API
```

### Task 5.2: Add classical_models to index.rst toctree

Add the new guide to the documentation index.

- **Files**:
  - `docs/index.rst` - Add to Guides toctree
- **Success**:
  - Classical models guide accessible from index
- **Dependencies**:
  - Task 5.1 completion

**Implementation** - Add after `modeling`:

```restructuredtext
   guide/modeling
   guide/classical_models
   guide/hpo
```

---

## Phase 6: Update Code Examples in Guides

### Task 6.1: Update docs/guide/cli.rst examples

Update all programmatic examples to use current patterns.

- **Files**:
  - `docs/guide/cli.rst` - Update Python code blocks
- **Success**:
  - All examples use `ModelRegistry.create()`
- **Research References**:
  - #file:../research/20260105-sphinx-docs-v1-update-research.md (Lines 35-38) - CLI examples
- **Dependencies**:
  - Phase 5 completion

**Changes required:**

1. Single Model Training example (lines ~45-55):

```python
# OLD
from admet.model.chemprop import ChempropModel, ChempropConfig
config = OmegaConf.merge(
    OmegaConf.structured(ChempropConfig),
    OmegaConf.load("configs/0-experiment/chemprop.yaml")
)
model = ChempropModel.from_config(config)
model.fit()

# NEW
from omegaconf import OmegaConf
from admet.model import ModelRegistry

config = OmegaConf.load("configs/0-experiment/chemprop.yaml")
model = ModelRegistry.create(config)
model.fit(train_smiles, train_targets)
```

2. Ensemble Training example (lines ~65-75):

```python
# OLD
from admet.model.chemprop import ModelEnsemble, EnsembleConfig
config = OmegaConf.merge(
    OmegaConf.structured(EnsembleConfig),
    OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
)
ensemble = ModelEnsemble.from_config(config)

# NEW
from omegaconf import OmegaConf
from admet.model.chemprop import ModelEnsemble

config = OmegaConf.load("configs/3-production/ensemble_chemprop.yaml")
ensemble = ModelEnsemble(config)
ensemble.train_all()
```

### Task 6.2: Update docs/guide/modeling.rst examples

Update modeling guide examples.

- **Files**:
  - `docs/guide/modeling.rst` - Update code blocks
- **Success**:
  - All examples current
- **Dependencies**:
  - Task 6.1 completion

**Changes required:**

1. Single Model Training (lines ~20-30)
2. Ensemble Training (lines ~40-55)
3. HPO example (lines ~70-80)

Use same pattern as Task 6.1.

### Task 6.3: Update docs/guide/hpo.rst examples

Update HPO guide examples.

- **Files**:
  - `docs/guide/hpo.rst` - Update code blocks
- **Success**:
  - HPO examples current
- **Dependencies**:
  - Task 6.2 completion

**Key updates:**

- Quick Start programmatic example
- Configuration loading examples

### Task 6.4: Update docs/guide/architecture.rst examples

Update architecture guide examples.

- **Files**:
  - `docs/guide/architecture.rst` - Update Configuration System section
- **Success**:
  - Examples match current API
- **Dependencies**:
  - Task 6.3 completion

**Changes required (lines ~90-100):**

```python
# OLD
from admet.model.chemprop import ChempropConfig, EnsembleConfig
config = OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
cfg = OmegaConf.structured(EnsembleConfig(**config))
print(cfg.model.ffn_type)

# NEW
from omegaconf import OmegaConf
from admet.model import ModelRegistry

config = OmegaConf.load("configs/3-production/ensemble_chemprop.yaml")
print(config.model.type)  # "chemprop"
print(config.model.chemprop.ffn_type)  # "regression"

model = ModelRegistry.create(config)
```

### Task 6.5: Review and update remaining guide files

Review all remaining guide files for outdated examples.

- **Files**:
  - `docs/guide/curriculum.rst`
  - `docs/guide/task_affinity.rst`
  - `docs/guide/splitting.rst`
  - `docs/guide/mlflow_artifacts.rst`
  - `docs/guide/profiling.rst`
  - `docs/guide/performance_optimization.rst`
- **Success**:
  - All guides use current API patterns
- **Dependencies**:
  - Task 6.4 completion

---

## Phase 7: Final Validation

### Task 7.1: Clean build and verify 0 warnings

Run complete clean build.

- **Files**:
  - None (verification)
- **Success**:
  - `rm -rf docs/_build && make -C docs html` shows 0 warnings
- **Dependencies**:
  - Phase 6 completion

**Command:**

```bash
cd /home/aglisman/VSCodeProjects/OpenADMET-ExpansionRx-Blind-Challenge
source .venv/bin/activate
rm -rf docs/_build
make -C docs html 2>&1 | grep -E "(WARNING|ERROR|build succeeded)"
```

### Task 7.2: Verify all internal links resolve

Check for broken cross-references.

- **Files**:
  - None (verification)
- **Success**:
  - No broken :doc: or :ref: links
- **Dependencies**:
  - Task 7.1 completion

**Command:**

```bash
make -C docs linkcheck
```

### Task 7.3: Spot-check rendered HTML for code examples

Manual verification of rendered documentation.

- **Files**:
  - `docs/_build/html/` - Browse key pages
- **Success**:
  - Code blocks render correctly
  - Examples are syntactically valid
  - API autodoc generates content
- **Dependencies**:
  - Task 7.2 completion

**Pages to check:**

- `_build/html/api/admet.model.html`
- `_build/html/api/admet.features.html`
- `_build/html/guide/configuration.html`
- `_build/html/guide/classical_models.html`
- `_build/html/guide/cli.html`

---

## Dependencies

- Python 3.11 with virtualenv
- Sphinx 7.3.7+
- sphinx-autodoc-typehints
- myst-parser
- furo theme

## Success Criteria

- `make -C docs html` completes with 0 warnings
- All 5 API subpackages documented (data, features, model, plot, util)
- Classical models guide created with XGBoost, LightGBM, CatBoost examples
- All code examples use `ModelRegistry.create()` pattern
- UnifiedModelConfig schema documented in config_reference.rst
- Version references updated to 1.2.0
- All new pages included in toctree
