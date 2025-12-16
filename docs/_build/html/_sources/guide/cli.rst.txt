CLI Usage
=========

The ``admet`` package provides Python modules for training Chemprop models.
Training is typically done via configuration files and Python scripts rather
than a CLI interface.

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

   python -m admet.model.chemprop.ensemble --config configs/0-experiment/ensemble_chemprop_production.yaml

Or programmatically:

.. code-block:: python

   from omegaconf import OmegaConf
   from admet.model.chemprop import ChempropEnsemble, EnsembleConfig

   config = OmegaConf.merge(
       OmegaConf.structured(EnsembleConfig),
       OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
   )

   ensemble = ChempropEnsemble.from_config(config)
   ensemble.train_all()

**Hyperparameter Optimization**

Run HPO with Ray Tune and ASHA scheduler:

.. code-block:: bash

   python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml

Configuration Files
-------------------

Configuration files are located in ``configs/``:

- ``single_chemprop.yaml``: Single model training configuration
- ``ensemble_chemprop.yaml``: Ensemble training across splits/folds
- ``hpo_chemprop.yaml``: Hyperparameter optimization settings
- ``chemprop_curriculum.yaml``: Curriculum learning configuration

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
