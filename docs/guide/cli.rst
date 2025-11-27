CLI Usage
=========

The `admet` command-line interface (CLI) is your primary tool for interacting with the framework. It streamlines common tasks like downloading datasets, creating reproducible splits, and training models.

You can access the help message for any command by appending `--help`.

Examples
--------

**Splitting Datasets**

To create reproducible splits using random and scaffold cluster methods:

.. code-block:: bash

   admet split datasets assets/dataset/eda/data/set \
       --log-level INFO \
       --output assets/dataset/splits \
       --overwrite

**Training a Single Model**

To train a single XGBoost model on a specific fold (dataset path is taken from
``data.root`` inside the config):

.. code-block:: bash

   admet --log-level INFO \
       train xgb \
       --config configs/xgb_train_single.yaml

**Distributed Training with Ray**

To train an ensemble of models across multiple folds using Ray for parallel execution:

.. code-block:: bash

   admet --log-level INFO \
       train xgb \
       --config configs/xgb_train_ensemble.yaml

Provide ``--data-root path/to/dataset`` to override the directory specified in the YAML at runtime.

MLflow logging is enabled by default. Set ``training.experiment_name`` (and optionally
``training.tracking_uri``) in the YAML to control where runs are recorded. Ensemble
training starts a parent run and logs each fold as a child run so metrics and artifacts
are grouped together. All YAML config values and CLI overrides are logged as MLflow
parameters for reproducibility.

See also
--------

* :doc:`overview` for a high-level introduction.
* :doc:`development` for setting up your environment.
