Modeling Guide
==============

This guide describes the model implementations and how to train models for
ADMET property prediction using the ``admet`` package.

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

   from admet.model.chemprop import ChempropModel, ChempropConfig
   from omegaconf import OmegaConf

   # Load configuration from YAML
   config = OmegaConf.load("configs/0-experiment/chemprop.yaml")
   cfg = OmegaConf.structured(ChempropConfig(**config))

   # Create and train model
   model = ChempropModel.from_config(cfg)
   model.fit(train_df, val_df)

   # Make predictions
   predictions = model.predict(test_df)

Ensemble Training
^^^^^^^^^^^^^^^^^

For production use, train multiple models across different data splits:

.. code-block:: python

   from admet.model.chemprop import ModelEnsemble, EnsembleConfig
   from omegaconf import OmegaConf

   # Load ensemble configuration
   config = OmegaConf.load("configs/0-experiment/ensemble_chemprop_production.yaml")
   cfg = OmegaConf.structured(EnsembleConfig(**config))

   # Train ensemble (parallelized with Ray)
   ensemble = ModelEnsemble.from_config(cfg)
   ensemble.train_all()

   # Make ensemble predictions with uncertainty
   predictions = ensemble.predict_ensemble(test_df)

FFN Architecture Options
^^^^^^^^^^^^^^^^^^^^^^^^

The ``ffn_type`` parameter controls the prediction head:

- ``regression``: Standard multi-layer perceptron (default)
- ``mixture_of_experts``: Mixture of experts for heterogeneous data
- ``branched``: Branched architecture with shared and task-specific layers

Configuration example:

.. code-block:: yaml

   model:
     ffn_type: mixture_of_experts
     ffn_hidden_dim: 300
     ffn_num_layers: 3

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

Curriculum Learning
-------------------

Quality-aware curriculum learning progressively incorporates data based on
quality tiers with count-normalized sampling:

.. code-block:: python

   from admet.model.chemprop import CurriculumState, CurriculumCallback, CurriculumPhaseConfig

   # Configure curriculum with count normalization (recommended)
   config = CurriculumPhaseConfig(
       count_normalize=True,  # Adjust for dataset size imbalance
       min_high_quality_proportion=0.25,  # Safety floor
   )

   curriculum = CurriculumState(
       qualities=["high", "medium", "low"],
       patience=5,  # Epochs without improvement before advancing
       config=config,
   )

   # Add callback to model training
   callback = CurriculumCallback(curr_state=curriculum)

Curriculum phases with conservative default proportions:

1. **Warmup**: 80% high, 15% medium, 5% low - learn core patterns
2. **Expand**: 60% high, 30% medium, 10% low - incorporate more data
3. **Robust**: 50% high, 35% medium, 15% low - build robustness
4. **Polish**: 70% high, 20% medium, 10% low - fine-tune while maintaining diversity

For detailed configuration and count normalization explanation, see :doc:`curriculum`.

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
- See :doc:`configuration` for detailed configuration options
- See :doc:`splitting` for dataset partitioning methodology
- See :doc:`mlflow_artifacts` for MLflow artifact structure and accessing predictions
