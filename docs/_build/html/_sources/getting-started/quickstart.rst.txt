=======================================
Quickstart: Train Your First Model
=======================================

Train a Chemprop MPNN model on ADMET data in under 5 minutes.

.. mermaid::

   flowchart LR
      A[📄 Config<br/>YAML] --> B[📊 Data<br/>Load]
      B --> C[🧠 Train<br/>Epochs]
      C --> D[📈 MLflow<br/>Metrics]

      style A fill:#e1f5fe
      style B fill:#fff9c4
      style C fill:#f3e5f5
      style D fill:#c8e6c9

Prerequisites
-------------

- Python 3.11 environment
- Package installed with development extras:

.. code-block:: bash

   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"

Step 1: Prepare Configuration
------------------------------

Create a minimal configuration file or use an existing one. Here's a simple example for training a Chemprop model:

.. code-block:: yaml

   # config.yaml
   data:
     data_dir: assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0
     test_file: assets/dataset/set/local_test.csv
     smiles_col: SMILES
     target_cols:
       - LogD
       - Log KSOL
       - Log HLM CLint
       - Log MLM CLint
       - Log Caco-2 Permeability Papp A>B
       - Log Caco-2 Permeability Efflux
       - Log MPPB
       - Log MBPB
       - Log MGMB

   model:
     type: chemprop
     chemprop:
       depth: 5
       message_hidden_dim: 600
       num_layers: 2
       hidden_dim: 600
       dropout: 0.1
       ffn_type: regression

   optimization:
     criterion: MSE
     max_epochs: 50
     batch_size: 32
     init_lr: 0.0001
     max_lr: 0.001
     patience: 10

   mlflow:
     tracking: true
     tracking_uri: http://127.0.0.1:8084
     experiment_name: Quickstart

Alternatively, use an example configuration from the repository:

.. code-block:: bash

   cp configs/0-experiment/0-single-fold/chemprop.yaml my_config.yaml

Step 2: Train Model
-------------------

**Using CLI:**

.. code-block:: bash

   admet model train -c my_config.yaml

**Expected output:**

.. code-block:: text

   ✓ Configuration validated
   ✓ Data loaded: 1200 training, 300 validation samples
   ✓ Model initialized: Chemprop MPNN (1.2M parameters)

   Training progress:
   Epoch 1/50: train_loss=1.234, val_loss=1.567
   Epoch 5/50: train_loss=0.892, val_loss=1.234
   ...
   Epoch 32/50: Early stopping triggered

   ✓ Training complete: 32 epochs, 8.5 minutes
   ✓ Best validation loss: 0.987 at epoch 17
   ✓ Test metrics logged to MLflow

**Using Python API:**

.. code-block:: python

   from admet.model.chemprop import ChempropModel
   from admet.model.config import UnifiedModelConfig
   from omegaconf import OmegaConf

   # Load configuration
   config = OmegaConf.load("my_config.yaml")
   model_config = UnifiedModelConfig(**config)

   # Initialize and train
   model = ChempropModel(model_config)
   model.train()

   # Results automatically logged to MLflow
   print(f"Best validation loss: {model.best_val_loss:.3f}")

Step 3: View Results in MLflow
-------------------------------

Start the MLflow UI to explore your results:

.. code-block:: bash

   mlflow ui --backend-store-uri http://127.0.0.1:8084

Navigate to ``http://127.0.0.1:8084`` in your browser to see:

- Training and validation metrics over time
- Hyperparameter values
- Model artifacts and predictions
- Performance plots (parity plots, residuals)

**Key metrics to check:**

- ``val_loss``: Validation loss (lower is better)
- ``test_rmse``: Test set root mean squared error
- Per-endpoint metrics: ``test_rmse_LogD``, ``test_rmse_Log_KSOL``, etc.

Next Steps
----------

Now that you've trained your first model, explore advanced features:

- **Ensemble training**: Combine multiple models for better performance → :doc:`/guide/modeling`
- **Hyperparameter optimization**: Systematically tune model settings → :doc:`/guide/hpo`
- **Curriculum learning**: Progressive training on quality-filtered data → :doc:`/guide/curriculum`
- **Task affinity analysis**: Understand multi-task relationships → :doc:`/guide/task_affinity`
- **Production deployment**: Scale to production ensembles → :doc:`/guide/config_reference`

For detailed configuration options, see :doc:`/guide/configuration` and :doc:`/guide/config_reference`.

Troubleshooting
---------------

**MLflow connection error:**

If you see ``ConnectionRefusedError``, start the MLflow server:

.. code-block:: bash

   mlflow server --host 127.0.0.1 --port 8084 --backend-store-uri sqlite:///assets/models/mlflow_postgres/mlflow.db

**Data not found:**

Verify the data directory paths exist or update them in your config:

.. code-block:: bash

   ls -la assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0

**CUDA out of memory:**

Reduce batch size in your config:

.. code-block:: yaml

   optimization:
     batch_size: 16  # Reduced from 32
