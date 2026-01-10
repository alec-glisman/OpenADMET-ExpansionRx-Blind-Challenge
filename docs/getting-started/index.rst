Getting Started
===============

Build and evaluate ADMET prediction models with state-of-the-art graph neural networks, systematic hyperparameter optimization, and robust ensemble training.

Whether you're a machine learning practitioner or drug discovery expert, this guide will help you get started quickly.

Quickstart (5 Minutes)
----------------------

Train your first model in minutes:

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: 5-Minute Tutorial
      :link: quickstart
      :link-type: doc

      Complete end-to-end example from configuration to MLflow results.

Core Concepts
-------------

Before diving in, understand the key components:

**For ML Practitioners:**

1. **Architecture** (:doc:`/guide/architecture`) - System design and model choices
2. **Training Pipeline** (:doc:`/guide/modeling`) - Single model and ensemble training
3. **HPO** (:doc:`/guide/hpo`) - Hyperparameter optimization with Ray Tune
4. **Configuration** (:doc:`/guide/configuration`) - OmegaConf-based YAML configs

**For Drug Discovery Experts:**

1. **Endpoints** (:doc:`/guide/endpoints`) - The 9 ADMET properties we predict
2. **Data Sources** (:doc:`/guide/data_sources`) - Dataset details and quality tiers
3. **Splitting Strategy** (:doc:`/guide/splitting`) - Cluster-aware molecule splitting
4. **Leaderboard** (:doc:`/guide/leaderboard`) - Challenge submission and ranking

Learning Paths
--------------

Choose your path based on your goal:

.. tab-set::

   .. tab-item:: Train a Single Model

      1. :doc:`quickstart` - Run first training
      2. :doc:`/guide/configuration` - Understand config structure
      3. :doc:`/guide/modeling` - Explore training options
      4. :doc:`/guide/mlflow_artifacts` - Track experiments

   .. tab-item:: Run HPO

      1. :doc:`quickstart` - Basic training first
      2. :doc:`/guide/hpo` - HPO fundamentals and warmstarting
      3. :doc:`/reference/scripts` - Shell scripts for automation
      4. :doc:`/guide/profiling` - Performance optimization

   .. tab-item:: Deploy Ensemble

      1. :doc:`/guide/modeling` - Understand ensemble training
      2. :doc:`/guide/hpo` - Generate optimized configs
      3. :doc:`/reference/scripts` - Production scripts
      4. :doc:`/guide/troubleshooting` - Debug common issues

   .. tab-item:: Contribute

      1. :doc:`/guide/development` - Setup dev environment
      2. :doc:`/guide/architecture` - Understand codebase
      3. :doc:`/api/admet` - Explore API reference
      4. :doc:`/dev/planning` - Review planning docs

Installation
------------

See :doc:`installation` for detailed setup instructions.

**Quick install:**

.. code-block:: bash

   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev,docs]"
   uv run pre-commit install

**With pip:**

.. code-block:: bash

   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev,docs]"
   pre-commit install

**Verify installation:**

.. code-block:: bash

   admet --help
   pytest -q

Common Workflows
----------------

**Train single model:**

.. code-block:: bash

   admet model train -c configs/0-experiment/0-single-fold/chemprop.yaml

**Run HPO:**

.. code-block:: bash

   admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50

**Train ensemble:**

.. code-block:: bash

   admet model ensemble -c configs/3-hpo-ensemble-production/0_chemprop_v1/ensemble_chemprop_hpo_001.yaml --max-parallel 4

**Generate data splits:**

.. code-block:: bash

   admet data split data.csv --cluster-method bitbirch --n-splits 5 --n-folds 5

**Scrape leaderboard:**

.. code-block:: bash

   admet leaderboard scrape --user <username>

Next Steps
----------

- :doc:`quickstart` - Train your first model
- :doc:`/guide/cli` - Complete CLI reference
- :doc:`/reference/scripts` - Shell scripts for automation
- :doc:`/guide/architecture` - Understand the system design
