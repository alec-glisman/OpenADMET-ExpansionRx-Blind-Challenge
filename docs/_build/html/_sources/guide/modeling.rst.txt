Modeling Guide
==============

This guide describes the model implementations and how to train models for
ADMET property prediction using the ``admet`` package.

Chemprop Models
---------------

The primary modeling approach uses Chemprop message-passing neural networks
via the ``admet.model.chemprop`` subpackage. Key classes include:

- **ChempropModel**: Single model training with configurable FFN architectures
- **ChempropEnsemble**: Ensemble training across multiple splits/folds
- **ChempropHPO**: Hyperparameter optimization with Ray Tune

Single Model Training
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from admet.model.chemprop import ChempropModel, ChempropConfig
   from omegaconf import OmegaConf

   # Load configuration from YAML
   config = OmegaConf.load("configs/single_chemprop.yaml")
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

   from admet.model.chemprop import ChempropEnsemble, EnsembleConfig
   from omegaconf import OmegaConf

   # Load ensemble configuration
   config = OmegaConf.load("configs/ensemble_chemprop.yaml")
   cfg = OmegaConf.structured(EnsembleConfig(**config))

   # Train ensemble (parallelized with Ray)
   ensemble = ChempropEnsemble.from_config(cfg)
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

   from admet.model.chemprop import ChempropHPO, HPOConfig
   from omegaconf import OmegaConf

   # Load HPO configuration
   config = OmegaConf.load("configs/hpo_chemprop.yaml")
   cfg = OmegaConf.structured(HPOConfig(**config))

   # Run hyperparameter search
   hpo = ChempropHPO.from_config(cfg)
   best_config = hpo.run()

Curriculum Learning
-------------------

Quality-aware curriculum learning progressively incorporates data based on
quality tiers:

.. code-block:: python

   from admet.model.chemprop import CurriculumState, CurriculumCallback

   # Configure curriculum phases
   curriculum = CurriculumState(
       warmup_epochs=5,      # Use only high-quality data
       expand_epochs=10,     # Gradually add medium-quality data
       robust_epochs=15,     # Include all data
       polish_epochs=5,      # Fine-tune on high-quality data
   )

   # Add callback to model training
   callback = CurriculumCallback(curriculum_state=curriculum)

Curriculum phases:

1. **Warmup**: Train only on highest-quality data
2. **Expand**: Gradually incorporate lower-quality data
3. **Robust**: Use all available data with quality-based weighting
4. **Polish**: Fine-tune on high-quality data

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

- See :doc:`configuration` for detailed configuration options
- See :doc:`splitting` for dataset partitioning methodology
