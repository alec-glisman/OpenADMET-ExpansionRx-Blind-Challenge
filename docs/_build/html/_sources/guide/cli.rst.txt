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

To train a single XGBoost model on a specific fold:

.. code-block:: bash

   admet train xgb \
       'assets/dataset/splits/high_quality/random_cluster/split_0/fold_0/hf_dataset' \
       --config configs/xgb.yaml \
       --output-dir temp/xgb_artifacts/single_model \
       --seed 123

**Distributed Training with Ray**

To train an ensemble of models across multiple folds using Ray for parallel execution:

.. code-block:: bash

   admet train xgb \
       'assets/dataset/splits/high_quality/random_cluster' \
       --config configs/xgb.yaml \
       --output-dir temp/xgb_artifacts/ensemble \
       --seed 123 \
       --multi \
       --ray-address "local"

See also
--------

* :doc:`overview` for a high-level introduction.
* :doc:`development` for setting up your environment.
