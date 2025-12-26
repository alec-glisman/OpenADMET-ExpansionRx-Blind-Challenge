CLI Usage
=========

The ``admet`` package provides command-line tools for leaderboard analysis
and model training workflows.

Leaderboard Commands
--------------------

The ``admet leaderboard`` command provides tools for scraping and analyzing
competition leaderboards.

**Basic Usage**

.. code-block:: bash

   # Scrape leaderboard for a user
   admet leaderboard scrape --user your_username

   # Custom output directory
   admet leaderboard scrape --user your_username --output ./results

   # Skip plot generation (faster)
   admet leaderboard scrape --user your_username --no-plots

   # Different HuggingFace Space
   admet leaderboard scrape --user your_username --space owner/space-name

For detailed leaderboard documentation, see :doc:`leaderboard`.

Model Training
--------------

Training Chemprop models is done via configuration files and Python scripts.

Training Chemprop Models
------------------------

**Single Model Training**

Train a single Chemprop model using a configuration file:

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.chemprop import ChempropModel, ChempropConfig

   # Load configuration
   config = OmegaConf.merge(
       OmegaConf.structured(ChempropConfig),
       OmegaConf.load("configs/0-experiment/chemprop.yaml")
   )

   # Train model
   model = ChempropModel.from_config(config)
   model.fit()

**Ensemble Training**

Train an ensemble of models across multiple data splits:

.. code-block:: bash

   # Using the admet CLI
   admet model ensemble --config configs/0-experiment/ensemble_chemprop_production.yaml

Or programmatically:

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.chemprop import ModelEnsemble, EnsembleConfig

   config = OmegaConf.merge(
       OmegaConf.structured(EnsembleConfig),
       OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
   )

   ensemble = ModelEnsemble.from_config(config)
   ensemble.train_all()

**Hyperparameter Optimization**

Run HPO with Ray Tune and ASHA scheduler:

.. code-block:: bash

   # Using the admet CLI
   admet model hpo --config configs/1-hpo-single/hpo_chemprop.yaml

**List Available Models**

View all registered model types:

.. code-block:: bash

   admet model list

This shows the available model backends: ``chemprop``, ``chemeleon``, ``xgboost``,
``lightgbm``, ``catboost``.

Configuration Files
-------------------

Data Commands
-------------

Split datasets and generate cluster-aware train/validation assignments using the `admet` CLI.

.. code-block:: bash

   # Run the high-level split pipeline and save augmented dataframe
   admet data split data/admet_train.csv --output outputs/ --smiles-col SMILES

Testing the CLI
---------------

The Typer app used by the CLI is available programmatically as ``admet.cli.app``.
When writing unit tests, prefer invoking the top-level app (``admet``) so subcommand
parsing behaves the same as when the CLI is installed as a console script. Use
Typer's ``CliRunner`` to exercise commands in tests, for example:

.. code-block:: python

   from typer.testing import CliRunner
   from admet.cli import app as main_app

   runner = CliRunner()
   result = runner.invoke(main_app, ["data", "split", "--output", "./out", "data.csv"])
   assert result.exit_code == 0

Avoid invoking sub-``Typer`` instances (for example ``data_app``) directly when
testing command-line parsing; doing so can lead to different argument parsing
behavior and unexpected errors.


Configuration files are located in ``configs/``:

- ``0-experiment/chemprop.yaml``: Single model training configuration
- ``0-experiment/ensemble_chemprop_production.yaml``: Ensemble training across splits/folds
- ``0-experiment/ensemble_joint_sampling_example.yaml``: Joint sampling example
- ``1-hpo-single/hpo_chemprop.yaml``: Hyperparameter optimization settings
- ``curriculum/``: Curriculum learning configurations
- ``task-affinity/``: Task affinity configurations
- ``3-production/``: Production ensemble configurations

MLflow Tracking
---------------

All training runs log to MLflow automatically. Configure tracking in your YAML:

.. code-block:: yaml

   mlflow:
     tracking: true
     tracking_uri: "http://127.0.0.1:8084"
     experiment_name: "chemprop_admet"

Start the MLflow server:

.. code-block:: bash

   mlflow server --host 127.0.0.1 --port 8080

See also
--------

* :doc:`overview` for a high-level introduction
* :doc:`modeling` for detailed training workflows
* :doc:`configuration` for configuration file structure
