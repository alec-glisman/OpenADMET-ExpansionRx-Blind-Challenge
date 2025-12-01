Data Pipeline (``admet.data``)
==============================

The ``admet.data`` package provides utilities for data loading, chemical
processing, and dataset splitting for ADMET property prediction.

.. automodule:: admet.data
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Modules
-------

constant
^^^^^^^^

Dataset constants and transformation helpers.

.. automodule:: admet.data.constant
   :members:
   :undoc-members:
   :show-inheritance:

fingerprint
^^^^^^^^^^^

Molecular fingerprint generation utilities.

.. automodule:: admet.data.fingerprint
   :members:
   :undoc-members:
   :show-inheritance:

property
^^^^^^^^

Property calculation and feature engineering.

.. automodule:: admet.data.property
   :members:
   :undoc-members:
   :show-inheritance:

smiles
^^^^^^

SMILES processing and canonicalization.

.. automodule:: admet.data.smiles
   :members:
   :undoc-members:
   :show-inheritance:

split
^^^^^

Cluster-based data splitting with stratification support.

.. automodule:: admet.data.split
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``split`` module provides sophisticated cluster-aware cross-validation:

- **Clustering Methods**: BitBirch (default), random, scaffold, k-means, UMAP, Butina
- **Splitting Strategies**: GroupKFold, StratifiedKFold, MultilabelStratifiedKFold
- **Quality-Aware Splitting**: Stratification by quality tiers and task coverage

Example usage:

.. code-block:: python

   from admet.data.split import create_bitbirch_splits

   # Create cluster-aware splits
   splits = create_bitbirch_splits(
       df=dataset,
       smiles_col="SMILES",
       task_cols=["LogD", "Solubility"],
       quality_col="quality",
       n_splits=5,
       n_folds=5,
   )

stats
^^^^^

Statistical analysis and dataset diagnostics.

.. automodule:: admet.data.stats
   :members:
   :undoc-members:
   :show-inheritance:
