Modeling Guide
==============

Model training uses Chemprop message-passing neural networks, classical ML methods
(XGBoost, LightGBM, CatBoost), and foundation models (Chemeleon). Train single models
or ensembles across 5 splits × 5 folds with Ray parallelization.

Training Workflow
-----------------

The typical training workflow progresses from single model to optimized ensemble:

.. mermaid::

   flowchart LR
      subgraph "Data Preparation"
         A[Raw Data] --> B[Cluster<br/>BitBirch]
         B --> C[5×5 Splits]
      end

      subgraph "Model Development"
         D[Single Model] --> E[Validate]
         E --> F[HPO Search]
         F --> G[Top K Configs]
      end

      subgraph "Production"
         H[Ensemble<br/>25 Models] --> I[Average<br/>Predictions]
         I --> J[Submit]
      end

      C --> D
      G --> H

      style A fill:#e1f5fe
      style F fill:#fff9c4
      style J fill:#c8e6c9

**Workflow Steps:**

1. **Data Split**: Generate cluster-aware splits with :doc:`splitting`
2. **Single Model**: Train baseline model to validate data and config
3. **HPO**: Run hyperparameter search with :doc:`hpo` to find optimal configs
4. **Ensemble**: Train multiple models across splits/folds for robustness
5. **Submit**: Average predictions and evaluate on test set

Chemprop Models
---------------

The primary modeling approach uses Chemprop message-passing neural networks
via the ``admet.model.chemprop`` subpackage. Key classes include:

- **ChempropModel**: Single model training with configurable FFN architectures
- **ModelEnsemble**: Ensemble training across multiple splits/folds
- **ChempropHPO**: Hyperparameter optimization with Ray Tune

Single Model Training
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from admet.model.registry import ModelRegistry
   from admet.model.config import UnifiedModelConfig
   from omegaconf import OmegaConf

   # Load configuration from YAML
   config = OmegaConf.merge(
       OmegaConf.structured(UnifiedModelConfig),
       OmegaConf.load("configs/0-experiment/0-single-fold/chemprop.yaml")
   )

   # Create and train model
   model = ModelRegistry.create(config)
   model.fit()  # Data is loaded from config.data

   # Make predictions on test SMILES
   test_smiles = ["CCO", "CCCO", "c1ccccc1"]
   predictions = model.predict(test_smiles)

Ensemble Training
^^^^^^^^^^^^^^^^^

For production use, train multiple models across different data splits using
the ``ModelEnsemble`` class:

.. code-block:: python

   from admet.model.chemprop.ensemble import ModelEnsemble
   from omegaconf import OmegaConf

   # Load ensemble configuration
   config = OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")

   # Create ensemble and train across splits/folds (parallelized with Ray)
   ensemble = ModelEnsemble.from_config(config)
   ensemble.discover_splits_folds()
   ensemble.train_all(max_parallel=4)

   # Make ensemble predictions with uncertainty quantification
   test_smiles = ["CCO", "CCCO", "c1ccccc1"]
   predictions = ensemble.predict_ensemble(test_smiles)

FFN Architecture Options
^^^^^^^^^^^^^^^^^^^^^^^^

The ``ffn_type`` parameter controls the prediction head:

- ``regression``: Standard multi-layer perceptron (default, best overall performance)
- ``mixture_of_experts``: Mixture of experts for heterogeneous data (MoE)
- ``branched``: Branched architecture with shared trunk and task-specific branches

.. note::

   In HPO search spaces, shorthand values ``mlp``, ``moe``, and ``branched`` are
   automatically mapped to the full config values.

Configuration example:

.. code-block:: yaml

   model:
     ffn_type: mixture_of_experts
     hidden_dim: 300
     num_layers: 3
     # MoE-specific
     n_experts: 4

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use Ray Tune with ASHA scheduler for HPO:

.. code-block:: python

   from admet.model.chemprop.hpo import ChempropHPO
   from omegaconf import OmegaConf

   # Load HPO configuration
   config = OmegaConf.load("configs/hpo_chemprop.yaml")

   # Run hyperparameter search
   hpo = ChempropHPO(config)
   best_config, results = hpo.run()

   print(f"Best validation MAE: {results.best_result['val_mae']:.4f}")

For comprehensive HPO documentation including search space configuration,
ASHA scheduler tuning, and best practices, see :doc:`hpo`.

Joint Sampling
--------------

The ``JointSampler`` provides unified two-stage sampling with task-aware oversampling:

**Task Oversampling Algorithm:**

Sample task ``t`` with probability ``p_t ∝ count_t^(-α)`` where ``α ∈ [0, 1]``
controls rebalancing strength (0 = uniform, 1 = fully inverse-weighted by task size).

**Configuration via YAML:**

.. code-block:: yaml

   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.5  # [0, 1] task rebalancing strength

.. note::
   **Curriculum Learning (Abandoned):** The curriculum learning feature that was
   part of joint sampling has been abandoned after a comprehensive ablation study
   showed it degraded performance by 15-30% when using external datasets. The
   extreme data sparsity (>90% missing) and distribution shift caused catastrophic
   forgetting on multiple endpoints. See :doc:`curriculum` for details.

MLflow Integration
------------------

All training runs log to MLflow automatically:

.. code-block:: python

   # Configuration specifies MLflow settings
   mlflow:
     tracking_uri: "mlruns"
     experiment_name: "chemprop_admet"
     log_model: true
     log_predictions: true

Logged artifacts include:

- Model checkpoints
- Training metrics (per epoch)
- Validation predictions
- Configuration YAML
- Learning curves

.. seealso::
   For detailed information about MLflow artifact organization, file formats,
   and how to access predictions and submission files, see :doc:`mlflow_artifacts`.

CheMeleon Models
----------------

CheMeleon is a pre-trained molecular encoder that learns transferable representations
from large-scale chemical datasets. The encoder is frozen during training, while
a trainable FFN head performs task-specific predictions.

**Key Features:**

- **Pre-trained Encoder:** Frozen message passing network trained on millions of molecules
- **Flexible FFN Head:** Supports all three FFN architecture types (regression, mixture_of_experts, branched)
- **Fast Training:** Only the FFN head parameters are optimized, enabling rapid adaptation

**Basic Training:**

.. code-block:: python

   from admet.model.chemeleon import ChemeleonModel
   from admet.model.config import ChemeleonModelParams

   params = ChemeleonModelParams(
       smiles_col="SMILES",
       target_cols=["LogD"],
       ffn_type="regression",  # or "mixture_of_experts", "branched"
       ffn_hidden_dim=128,
       ffn_n_layers=2,
       dropout=0.1,
   )
   model = ChemeleonModel(params)
   model.fit(train_df)

**FFN Architecture Options:**

CheMeleon supports the same FFN architectures as Chemprop:

- ``regression``: Standard feedforward network (default)
- ``mixture_of_experts``: Multiple specialized sub-networks with learned gating
- ``branched``: Shared trunk with task-specific branches for multi-task learning

.. code-block:: yaml

   # Example: Mixture of Experts for multi-task
   ffn_type: mixture_of_experts
   n_experts: 4
   ffn_hidden_dim: 256
   ffn_n_layers: 3

**HPO Support:**

CheMeleon includes hyperparameter optimization via Ray Tune:

.. code-block:: python

   from admet.model.chemeleon import ChemeleonHPO
   from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig

   config = ChemeleonHPOConfig(
       train_data_path="data/train.csv",
       num_samples=50,
       max_epochs=100,
   )
   hpo = ChemeleonHPO(config)
   best_config, results = hpo.run()

For FFN-specific parameters during HPO, set ``tune_ffn_type=True`` in the search
space configuration to explore all architecture variants.

Classical ML Models
-------------------

For baseline comparisons, classical models are available in
``admet.model.classical``:

.. code-block:: python

   from admet.model.classical import XGBoostModel

   # Train XGBoost baseline
   model = XGBoostModel(params={"max_depth": 6, "learning_rate": 0.05})
   model.fit(X_train, y_train)
   preds = model.predict(X_test)

Model Persistence
-----------------

Models are saved as PyTorch Lightning checkpoints:

.. code-block:: python

   # Save model
   model.save("models/chemprop_logd.ckpt")

   # Load model
   model = ChempropModel.load("models/chemprop_logd.ckpt")

Cross-References
----------------

- See :doc:`hpo` for hyperparameter optimization guide
- See :doc:`profiling` for performance profiling and optimization
- See :doc:`configuration` for detailed configuration options
- See :doc:`splitting` for dataset partitioning methodology
- See :doc:`mlflow_artifacts` for MLflow artifact structure and accessing predictions
