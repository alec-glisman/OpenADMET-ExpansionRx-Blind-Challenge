Architecture Overview
======================

This page gives you a high‑level mental model of the `admet` package: how
modules are layered, how data flows through the system, and where you can
extend functionality.

Layered Package Map
-------------------

The code base is organized into clear functional layers:

1. Data Layer (`admet.data`): loading, curation, chemical feature extraction, splitting.
2. Modeling Layer (`admet.model`): model wrappers (e.g., XGBoost, LightGBM) with unified interfaces.
3. Training Layer (`admet.train`): trainer abstractions that orchestrate datasets, models, metrics, persistence.
4. Evaluation Layer (`admet.evaluate`): metric calculation and reporting.
5. Visualization Layer (`admet.visualize`): generation of performance figures and dataset exploration plots.
6. CLI Layer (`admet.cli`): user‑facing entry points for download, split, train, evaluate.

Data Flow Lifecycle
-------------------

The typical end‑to‑end pipeline follows these stages:

.. code-block:: text

   raw data --> standardization / cleaning --> feature engineering (fingerprints)
             --> dataset splitting (quality tiers & train/valid/test)
             --> model training (hyperparameters / config) --> evaluation (metrics)
             --> visualization (figures, summaries) --> artifacts (saved models)

Key Modules and Responsibilities
--------------------------------

- `admet.data.download`: Fetch or assemble raw and curated datasets.
- `admet.data.chem` / `fingerprinting`: Molecular feature generation (e.g. fingerprints).
- `admet.data.splitter`: Implements split strategies and quality stratification.
- `admet.model.*_wrapper`: Adapters around external libraries to present a unified interface.
- `admet.train.*`: Trainers coordinating model fitting and artifact output.
- `admet.evaluate.metrics`: Metric definitions for classification/regression.
- `admet.visualize.*`: Plotting utilities for dataset and performance inspection.

Extensibility Points
--------------------

You can add new functionality without modifying core internals by:

- New Data Source: Implement a loader in `admet.data.download` and register it in the CLI.
- Custom Feature Representation: Add a module in `admet.data.fingerprinting` and expose a factory.
- New Model Type: Create a wrapper subclass in `admet.model` following existing wrapper API.
- Alternative Trainer: Implement a trainer in `admet.train` leveraging shared utilities.
- Additional Metrics: Define functions in `admet.evaluate.metrics` and reference them in evaluation.
- New Visualizations: Add plotting functions in `admet.visualize` and link them in docs/examples.

Design Principles
-----------------

- Separation of Concerns: Keep data preparation, training, and evaluation isolated.
- Minimal Coupling: Model wrappers do not assume specific dataset implementations beyond required interface.
- Declarative Configuration: Hyperparameters and run settings externalized (e.g., YAML config files).
- Reproducibility: Deterministic seeds for splitting and training when feasible.

Cross-References
----------------

- See :doc:`data_sources` for input provenance and curation.
- See :doc:`splitting` for dataset partitioning methodology.
- See :doc:`modeling` (planned) for details on adding new model wrappers.

