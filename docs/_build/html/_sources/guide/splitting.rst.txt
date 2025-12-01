Dataset Splitting Methodology
=============================

This guide describes how datasets are partitioned into training and validation
sets using cluster-aware cross-validation with quality stratification.

Goals
-----

- Maintain representative distributions across splits
- Avoid data leakage via cluster-based grouping (similar molecules stay together)
- Respect quality tiers for training robustness
- Ensure reproducibility via deterministic seeds

Core Module
-----------

The ``admet.data.split`` module provides the splitting functionality:

- ``pipeline()``: High-level function for end-to-end splitting
- ``get_bitbirch_clusters()``: Cluster SMILES using BitBirch fingerprints
- ``cluster_data()``: Cluster molecules using various methods
- ``cluster_multilabel_stratified_kfold()``: Multi-label stratified k-fold splitting

Clustering Methods
------------------

The module supports multiple clustering approaches:

- **BitBirch** (default): Fast hierarchical clustering on molecular fingerprints
- **Random**: Random cluster assignment
- **Scaffold**: Murcko scaffold-based grouping
- **K-Means**: K-means clustering on fingerprints
- **UMAP**: UMAP dimensionality reduction + clustering
- **Butina**: Taylor-Butina clustering

Splitting Strategies
--------------------

Three k-fold cross-validation strategies are available:

- **MultilabelStratifiedKFold**: Stratifies by task presence and quality tier (recommended)
- **StratifiedKFold**: Stratifies by quality tier only
- **GroupKFold**: Ensures cluster integrity (no cluster split across folds)

Pipeline Usage
--------------

The high-level ``pipeline()`` function handles clustering and splitting:

.. code-block:: python

   from admet.data.split import pipeline

   # Create 5 splits × 5 folds with BitBirch clustering
   df_with_splits = pipeline(
       df=dataset,
       smiles_col="SMILES",
       quality_col="Quality",
       target_cols=["LogD", "Log KSOL", "Log HLM CLint"],
       cluster_method="bitbirch",
       split_method="multilabel_stratified_kfold",
       n_splits=5,
       n_folds=5,
       fig_dir="outputs/split_diagnostics",
   )

   # Result contains columns: split_0_fold_0, split_0_fold_1, etc.
   # Values are "train" or "validation"

Lower-Level API
---------------

For more control, use the individual functions:

.. code-block:: python

   from admet.data.split import (
       get_bitbirch_clusters,
       cluster_data,
       cluster_multilabel_stratified_kfold,
   )

   # Get cluster labels
   cluster_labels = get_bitbirch_clusters(smiles_list)

   # Or use cluster_data for any method
   cluster_mapping = cluster_data(
       df,
       smiles_col="SMILES",
       cluster_method="bitbirch",
   )

   # Perform stratified splitting
   fold_assignments = cluster_multilabel_stratified_kfold(
       df=df,
       cluster_labels=cluster_labels,
       target_cols=target_cols,
       quality_col="Quality",
       n_folds=5,
       seed=42,
   )

Output Structure
----------------

Split outputs follow this directory structure:

.. code-block:: text

   assets/dataset/split_train_val/
   └── v3/
       └── quality_high/
           └── bitbirch/
               └── multilabel_stratified_kfold/
                   └── data/
                       ├── split_0/
                       │   ├── fold_0/
                       │   │   ├── train.csv
                       │   │   └── validation.csv
                       │   ├── fold_1/
                       │   └── ...
                       └── split_1/
                           └── ...

Diagnostic Figures
------------------

When ``fig_dir`` is provided, the pipeline generates:

- Cluster size histograms
- Per-endpoint value distributions by split
- Train/validation set size comparisons
- Quality tier distributions

Cross-References
----------------

- See :doc:`data_sources` for upstream curated dataset provenance
- See :doc:`architecture` for the overall pipeline placement
- See :doc:`modeling` for how splits are used in training
