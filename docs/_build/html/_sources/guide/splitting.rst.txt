Dataset Splitting Methodology
=============================

This guide describes how datasets are partitioned into training, validation,
test sets and how quality tiers interact with splits.

Goals
-----

- Maintain representative distributions across splits.
- Avoid data leakage (e.g. identical molecules appearing in both train and test).
- Respect quality tiers (e.g. high quality prioritized for training robustness).
- Ensure reproducibility via deterministic seeds.

Core Components
---------------

- `admet.data.dataset_split_pipeline`: Orchestrates end‑to‑end splitting.
- `admet.data.splitter`: Implements splitting strategies and helper utilities.

Typical Workflow
----------------

.. code-block:: text

   load curated dataset --> annotate quality --> group / stratify --> perform split
                         --> write split artifacts --> downstream training

Stratification Logic (Conceptual)
---------------------------------

(Placeholder) Strategies may include:

- Random Stratified: Maintain label distribution across train/valid/test.
- Scaffold or Cluster Based: Group molecules to reduce structural leakage.
- Quality-Aware: Allocate high quality examples proportionally or with oversampling.

Determinism & Seeds
-------------------

Pass a fixed seed to splitting functions for reproducibility:

.. code-block:: python

   from admet.data.splitter import create_splits

   train_df, valid_df, test_df = create_splits(df, seed=42)

Leakage Avoidance
-----------------

Common safeguards:

- Deduplicate identical canonical SMILES before splitting.
- (Optional) Remove highly similar molecules or group by scaffold.
- Audit cross‑split overlap after splitting (report count, %).

Output Artifacts
----------------

Split outputs are stored under directories such as:

- `assets/raw/splits/high_quality/`
- `assets/raw/splits/medium_quality/`
- Combined or ensemble split variants in `temp/xgb_artifacts/` (for model training).

Validation & QA
---------------

Recommended post‑split checks:

1. Label distribution parity across splits.
2. Overlap count of unique SMILES between splits (expect 0 ideally).
3. Summary statistics (mean, std of key numeric fields) per split.
4. Size ratio (e.g. 70/15/15) adherence.

Extensibility
-------------

To add a new strategy:

1. Implement a function/class in `splitter.py` encapsulating logic.
2. Add a selector/factory to route CLI argument to new implementation.
3. Document constraints (e.g. minimum dataset size) here and in CLI help.

Future Enhancements
-------------------

- Incorporate scaffold splitting for chemical diversity.
- Add automatic leakage audit reports to build artifacts.
- Support k‑fold cross validation mode.

Cross-References
----------------

- See :doc:`data_sources` for upstream curated dataset provenance.
- See :doc:`architecture` for the overall pipeline placement.

