.. _mlflow_artifacts:

=======================
MLflow Artifacts Guide
=======================

This document describes the complete MLflow artifacts structure generated during ensemble training for the OpenADMET ExpansionRx Challenge.

Overview
========

The ensemble training creates a hierarchical MLflow run structure:

* **1 Parent Run**: Ensemble-level aggregated results
* **25 Nested Runs**: Individual model results (5 splits × 5 folds)

Artifact Organization
=====================

Parent Run (Ensemble Level)
----------------------------

The parent run contains aggregated predictions and metrics across all 25 models.

.. code-block:: text

   Parent Run: production_ensemble_chemprop_hpo_topk/rank_001
   │
   ├── predictions/
   │   ├── test_ensemble_predictions.csv         # Detailed test predictions
   │   └── blind_ensemble_predictions.csv        # Detailed blind predictions
   │
   ├── submissions/
   │   ├── test_ensemble_submissions.csv         # Test submission format
   │   └── blind_ensemble_submissions.csv        # Blind submission format (FINAL)
   │
   ├── plots/
   │   ├── test/
   │   │   ├── prediction_distributions.png      # Test prediction distributions
   │   │   ├── uncertainty_distributions.png     # Test uncertainty distributions
   │   │   ├── parity_LogD.png                   # Parity plots for each endpoint
   │   │   ├── parity_Log_KSOL.png
   │   │   └── ... (one per endpoint)
   │   │
   │   └── blind/
   │       ├── prediction_distributions.png      # Blind prediction distributions
   │       └── uncertainty_distributions.png     # Blind uncertainty distributions
   │
   └── metrics/
       └── (ensemble-level metrics logged as MLflow metrics)

Nested Runs (Individual Models)
--------------------------------

Each of the 25 models (5 splits × 5 folds) has its own nested run.

.. code-block:: text

   Nested Run: split_0_fold_0
   │
   ├── predictions/
   │   ├── train_predictions.csv                 # Training set predictions
   │   ├── train_submissions.csv                 # Training submissions
   │   ├── train_distribution_stats.csv          # Training distribution stats
   │   │
   │   ├── val_predictions.csv                   # Validation set predictions
   │   ├── val_submissions.csv                   # Validation submissions
   │   ├── val_distribution_stats.csv            # Validation distribution stats
   │   │
   │   ├── test_predictions.csv                  # Test set predictions
   │   ├── test_submissions.csv                  # Test submissions
   │   ├── test_distribution_stats.csv           # Test distribution stats
   │   │
   │   ├── blind_predictions.csv                 # Blind set predictions
   │   ├── blind_submissions.csv                 # Blind submissions
   │   └── blind_distribution_stats.csv          # Blind distribution stats
   │
   ├── plots/
   │   ├── train/
   │   │   ├── parity_LogD.png                   # Parity plots per endpoint
   │   │   └── ... (one per endpoint)
   │   │
   │   ├── val/
   │   │   ├── parity_LogD.png
   │   │   └── ... (one per endpoint)
   │   │
   │   └── test/
   │       ├── parity_LogD.png
   │       └── ... (one per endpoint)
   │
   ├── checkpoints/
   │   └── best_model.ckpt                       # Best model checkpoint
   │
   └── metrics/
       └── (per-model metrics logged as MLflow metrics)

File Formats
============

Predictions CSV (Detailed Format)
----------------------------------

Ensemble Level
~~~~~~~~~~~~~~

File: ``*_ensemble_predictions.csv``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``SMILES``
     - Canonical SMILES string
   * - ``Molecule Name``
     - Molecule identifier (if present in input)
   * - ``{endpoint}_mean``
     - Mean prediction across 25 models
   * - ``{endpoint}_std``
     - Standard deviation across models
   * - ``{endpoint}_stderr``
     - Standard error (std / √25)
   * - ``{endpoint}_transformed_mean``
     - 10^mean for "Log " columns
   * - ``{endpoint}_transformed_stderr``
     - Propagated stderr through 10^x
   * - ``{endpoint}_actual``
     - Ground truth (test set only)

Individual Model Level
~~~~~~~~~~~~~~~~~~~~~~

File: ``*_predictions.csv``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``SMILES``
     - Canonical SMILES string
   * - ``{endpoint}``
     - Raw prediction in log space

Submissions CSV (Challenge Format)
-----------------------------------

Ensemble Level
~~~~~~~~~~~~~~

File: ``*_ensemble_submissions.csv``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Column
     - Description
   * - ``SMILES``
     - Canonical SMILES string
   * - ``Molecule Name``
     - Molecule identifier (if present)
   * - ``LogD``
     - Direct mean prediction
   * - ``KSOL``
     - 10^(Log KSOL mean) - transformed to μM
   * - ``HLM CLint``
     - 10^(Log HLM CLint mean) - transformed
   * - ``MLM CLint``
     - 10^(Log MLM CLint mean) - transformed
   * - ``Caco-2 Permeability Papp A>B``
     - 10^(Log Caco-2 Papp mean)
   * - ``Caco-2 Permeability Efflux``
     - 10^(Log Caco-2 Efflux mean)
   * - ``MPPB``
     - 10^(Log MPPB mean) - % unbound
   * - ``MBPB``
     - 10^(Log MBPB mean) - % unbound
   * - ``MGMB``
     - 10^(Log MGMB mean) - % unbound

Individual Model Level
~~~~~~~~~~~~~~~~~~~~~~

Same format but contains single model predictions instead of ensemble mean.

Distribution Stats CSV
----------------------

Statistical summaries for each endpoint.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Description
   * - ``split``
     - Dataset split (train/val/test/blind)
   * - ``target``
     - Endpoint name
   * - ``transform``
     - "raw" (log space) or "10^x" (linear space)
   * - ``statistic``
     - Stat name (mean, median, std, min, max, etc.)
   * - ``value``
     - Statistic value

Accessing Artifacts
===================

Via MLflow UI
-------------

.. code-block:: bash

   # Start MLflow server
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8084

   # Navigate to http://localhost:8084
   # Select experiment: production_ensemble_chemprop_hpo_topk
   # Select run: rank_001
   # Click "Artifacts" tab

Via MLflow Client (Python)
---------------------------

.. code-block:: python

   import mlflow

   # Connect to tracking server
   mlflow.set_tracking_uri("http://127.0.0.1:8084")

   # Get run
   run = mlflow.get_run(run_id)

   # Download artifact
   mlflow.artifacts.download_artifacts(
       run_id=run_id,
       artifact_path="submissions/blind_ensemble_submissions.csv",
       dst_path="./downloads"
   )

Via Command Line
----------------

.. code-block:: bash

   # Download all artifacts from a run
   mlflow artifacts download \
     --run-id <run_id> \
     --dst-path ./artifacts

   # Download specific artifact
   mlflow artifacts download \
     --run-id <run_id> \
     --artifact-path submissions/blind_ensemble_submissions.csv \
     --dst-path ./submissions

Key Files for Challenge Submission
===================================

Final Submission File
---------------------

**Primary:** ``submissions/blind_ensemble_submissions.csv`` from the **parent run**

This file contains:

* Mean predictions aggregated across all 25 models
* Transformed values (10^x for Log columns)
* Proper units for all endpoints
* Molecule Name column (if present in blind_test.csv)

Validation
----------

**For local validation:** ``submissions/test_ensemble_submissions.csv`` from the **parent run**

Compare this against the held-out test set to verify model performance.

Uncertainty Estimates
---------------------

**For confidence intervals:** ``predictions/blind_ensemble_predictions.csv`` from the **parent run**

This file includes:

* ``{endpoint}_mean``: Point estimate
* ``{endpoint}_stderr``: Standard error for confidence intervals
* Example: 95% CI = mean ± 1.96 × stderr

Workflow Example
================

1. Train Ensemble
-----------------

.. code-block:: bash

   python -m admet.model.chemprop.ensemble \
     --config configs/3-production/ensemble_chemprop_hpo_001.yaml

2. Locate Parent Run
---------------------

Check MLflow UI for the parent run ID, or find it in logs:

.. code-block:: text

   INFO - Parent MLflow run: 7a3f8e9c1d2b4a5e6f7g8h9i0j1k2l3m

3. Download Final Submission
-----------------------------

.. code-block:: bash

   mlflow artifacts download \
     --run-id 7a3f8e9c1d2b4a5e6f7g8h9i0j1k2l3m \
     --artifact-path submissions/blind_ensemble_submissions.csv \
     --dst-path ./final_submission

4. Verify Format
----------------

.. code-block:: python

   import pandas as pd

   # Load submission
   df = pd.read_csv("final_submission/blind_ensemble_submissions.csv")

   # Check required columns
   required = ["SMILES", "LogD", "KSOL", "HLM CLint", "MLM CLint",
               "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
               "MPPB", "MBPB", "MGMB"]

   assert all(col in df.columns for col in required), "Missing required columns"
   assert len(df) > 0, "Submission file is empty"
   print(f"✓ Submission file valid with {len(df)} molecules")

Notes
=====

Molecule Name Preservation
---------------------------

The ``Molecule Name`` column is automatically preserved if present in:

* ``test_file`` → appears in test predictions
* ``blind_file`` → appears in blind predictions

This column is included in both ``*_predictions.csv`` and ``*_submissions.csv`` files.

Missing Values
--------------

* Individual model predictions may contain NaN for specific endpoints
* Ensemble predictions handle NaN by computing statistics over available (non-NaN) values
* If all 25 models predict NaN for a molecule/endpoint, the ensemble will also be NaN

Temporary Files
---------------

The ensemble creates a temporary directory (``/tmp/ensemble_*``) during training. Artifacts are:

1. Saved to temp directory
2. Logged to MLflow
3. Temp directory is cleaned up

.. important::
   Always retrieve artifacts from MLflow, not the temp directory.

Parallelization
---------------

Artifacts are logged from Ray workers to MLflow. The parent run aggregates all results after all workers complete. Ensure the MLflow tracking server is accessible from all Ray worker nodes.

Troubleshooting
===============

Missing Blind Predictions
--------------------------

**Symptom:** Only test predictions appear, no blind predictions

**Solution:** Ensure ``blind_file`` is specified in the ensemble config:

.. code-block:: yaml

   data:
     blind_file: "assets/dataset/set/blind_test.csv"

**Verification:** Check that line 551 in ``ensemble.py`` assigns ``blind_preds``:

.. code-block:: python

   blind_preds = pred_df.copy()  # This line must be present

Empty Submission Files
----------------------

**Symptom:** Submission CSV files have no data rows

**Causes:**

1. All molecules failed SMILES canonicalization
2. All predictions returned NaN
3. Input file path incorrect

**Debug:**

* Check logs for SMILES parsing errors
* Verify input file exists and is readable
* Check individual model predictions for NaN patterns

MLflow Artifact Upload Failures
--------------------------------

**Symptom:** ``MlflowException: Failed to log artifact``

**Solutions:**

1. Verify MLflow server is running: ``curl http://127.0.0.1:8084/health``
2. Check disk space on MLflow artifact store
3. Ensure Ray workers can reach MLflow server
4. Verify MLflow version compatibility (≥2.0)

See Also
========

* :doc:`modeling` - Model training and evaluation guide
* :doc:`hpo` - Hyperparameter optimization guide
* :doc:`config_reference` - Configuration file reference
* :ref:`genindex` - Complete API index

References
==========

* `MLflow Tracking Documentation <https://mlflow.org/docs/latest/tracking.html>`_
* :doc:`modeling` - Ensemble training guide
* :doc:`hpo` - Hyperparameter optimization guide
