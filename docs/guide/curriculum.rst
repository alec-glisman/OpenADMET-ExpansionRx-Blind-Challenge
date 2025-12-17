Curriculum Learning Guide
=========================

This guide covers quality-aware curriculum learning for Chemprop models. Curriculum
learning progressively adjusts sampling from high-quality to lower-quality data
based on validation loss improvements, enabling more robust model training.

Overview
--------

Curriculum learning is inspired by how humans learn—starting with easier or
more reliable examples before moving to harder or noisier ones. In ADMET
prediction, data quality varies significantly:

- **High quality**: Well-characterized compounds with reliable measurements
- **Medium quality**: Mixed reliability or older data
- **Low quality**: Noisy or less reliable measurements

The curriculum progressively exposes the model to this data in phases.

Curriculum Phases
-----------------

The curriculum proceeds through four phases:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Phase
     - Weights (3-quality)
     - Description
   * - **Warmup**
     - 90% high, 10% medium, 0% low
     - Focus on high-quality data to learn core patterns
   * - **Expand**
     - 60% high, 35% medium, 5% low
     - Gradually incorporate medium-quality data
   * - **Robust**
     - 40% high, 40% medium, 20% low
     - Include low-quality data for robustness
   * - **Polish**
     - 100% high, 0% medium, 0% low
     - Fine-tune on high-quality data only

Phase transitions occur automatically when the overall validation loss stops
improving for ``patience`` epochs.

Quick Start
-----------

Enable curriculum learning via the ``joint_sampling`` configuration:

.. code-block:: yaml

   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.5  # Optional: enable task-aware oversampling
     curriculum:
       enabled: true
       quality_col: "Quality"
       qualities:
         - "high"
         - "medium"
         - "low"
       patience: 5
       strategy: "sampled"
       log_per_quality_metrics: true
     seed: 42

Then train normally:

.. code-block:: python

   from admet.model.chemprop import ChempropModel
   from omegaconf import OmegaConf

   config = OmegaConf.load("configs/curriculum/chemprop_curriculum.yaml")
   model = ChempropModel.from_config(config)
   model.fit()

Configuration
-------------

Curriculum Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``joint_sampling`` configuration, which combines curriculum learning
with task-aware oversampling:

.. code-block:: yaml

   joint_sampling:
     enabled: true

     # Task-aware oversampling (optional)
     task_oversampling:
       alpha: 0.5  # 0=uniform, 1=fully inverse-weighted by task size

     # Curriculum learning settings
     curriculum:
       enabled: true
       quality_col: "Quality"
       qualities:
         - "high"
         - "medium"
         - "low"
       patience: 5
       strategy: "sampled"  # or "deterministic"
       reset_early_stopping_on_phase_change: false
       log_per_quality_metrics: true

     # Global sampling settings
     num_samples: null  # null = use full dataset size per epoch
     seed: 42
     increment_seed_per_epoch: true
     log_to_mlflow: true

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Whether to enable curriculum learning
   * - ``quality_col``
     - ``"Quality"``
     - Column name containing quality labels
   * - ``qualities``
     - ``["high", "medium", "low"]``
     - Ordered list of quality levels (highest to lowest)
   * - ``patience``
     - ``5``
     - Epochs without validation loss improvement before advancing phase
   * - ``seed``
     - ``42``
     - Random seed for reproducible curriculum sampling

Adaptive Quality Levels
^^^^^^^^^^^^^^^^^^^^^^^

The curriculum adapts to the number of quality levels provided:

- **1 quality**: warmup → polish (effectively no curriculum)
- **2 qualities**: warmup → expand → polish
- **3+ qualities**: warmup → expand → robust → polish

Example with two quality levels:

.. code-block:: yaml

   curriculum:
     enabled: true
     quality_col: "Quality"
     qualities:
       - "reliable"
       - "uncertain"
     patience: 5

Data Preparation
----------------

Your training data must include a quality column with values matching the
configured ``qualities`` list:

.. code-block:: python

   import pandas as pd

   # Example training data with quality labels
   df_train = pd.DataFrame({
       "SMILES": ["CCO", "CCN", "CCC", "CCCC"],
       "LogD": [1.2, 0.8, 2.1, 1.5],
       "Quality": ["high", "high", "medium", "low"]
   })

Quality labels should be lowercase strings matching exactly the values in
``curriculum.qualities``.

How It Works
------------

Sampling-Based Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^

The curriculum uses weighted random sampling to control which data points
appear in each training batch:

1. **WeightedRandomSampler**: Each sample gets a weight based on its quality
   and the current curriculum phase
2. **Sampling with replacement**: Samples are drawn according to weights,
   so high-quality samples appear more frequently in early phases
3. **Phase-dependent weights**: Weights are updated when phases change

.. code-block:: python

   from admet.model.chemprop.curriculum_sampler import build_curriculum_sampler
   from admet.model.chemprop.curriculum import CurriculumState

   # Create curriculum state
   state = CurriculumState(qualities=["high", "medium", "low"], patience=5)

   # Build sampler for training data
   quality_labels = df_train["Quality"].tolist()
   sampler = build_curriculum_sampler(
       quality_labels=quality_labels,
       curriculum_state=state,
       seed=42,
   )

   # Use with DataLoader
   loader = DataLoader(dataset, sampler=sampler, batch_size=32)

Phase Transition Logic
^^^^^^^^^^^^^^^^^^^^^^

Phase transitions are triggered by the ``CurriculumCallback``:

1. Monitor overall validation loss (``val_loss``)
2. Track best validation loss and epoch
3. If no improvement for ``patience`` epochs, advance to next phase
4. Update sampling weights based on new phase

.. code-block:: python

   from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState

   # Create curriculum state and callback
   state = CurriculumState(qualities=["high", "medium", "low"], patience=5)
   callback = CurriculumCallback(curr_state=state, monitor_metric="val_loss")

   # The callback is automatically added to the trainer when curriculum is enabled

Monitoring and Logging
----------------------

Phase Transitions
^^^^^^^^^^^^^^^^^

Phase transitions are logged to both the console and MLflow:

.. code-block:: text

   INFO - Curriculum phase transition: warmup -> expand at epoch 12 (step 1440),
          val_loss=0.4523, weights={'high': 0.6, 'medium': 0.35, 'low': 0.05}

MLflow Metrics
^^^^^^^^^^^^^^

When MLflow tracking is enabled, the following metrics are logged:

- ``curriculum_phase``: Numeric phase indicator (0=warmup, 1=expand, 2=robust, 3=polish)
- ``curriculum_phase_epoch``: Epoch when phase transition occurred

These metrics enable visualization of curriculum progression in MLflow UI.

Best Practices
--------------

Data Quality Assignment
^^^^^^^^^^^^^^^^^^^^^^^

Consider these factors when assigning quality labels:

1. **Measurement reliability**: Standard deviation of repeated measurements
2. **Data source**: Primary measurements vs. literature aggregation
3. **Assay conditions**: Standardized protocols vs. varied conditions
4. **Temporal factors**: Recent data may be higher quality

Quality Distribution
^^^^^^^^^^^^^^^^^^^^

Aim for a reasonable distribution across quality levels:

.. code-block:: python

   # Check quality distribution
   print(df_train["Quality"].value_counts(normalize=True))
   # high      0.40
   # medium    0.35
   # low       0.25

If one quality dominates, curriculum learning may have limited effect.

Patience Tuning
^^^^^^^^^^^^^^^

The ``patience`` parameter balances exploration vs. exploitation:

- **Lower patience (3-5)**: Faster phase transitions, less time in each phase
- **Higher patience (8-15)**: More thorough learning in each phase

Start with ``patience=5`` and adjust based on:

- Total training epochs
- Validation loss convergence behavior
- Dataset size and complexity

Integration with Other Features
-------------------------------

With Ensemble Training
^^^^^^^^^^^^^^^^^^^^^^

Curriculum learning works with ensemble training. Each model in the ensemble
follows its own curriculum progression:

.. code-block:: yaml

   # configs/ensemble_curriculum.yaml
   data:
     data_dir: "assets/dataset/split_train_val/..."
     target_cols: ["LogD", "Log KSOL"]

   curriculum:
     enabled: true
     quality_col: "Quality"
     qualities: ["high", "medium", "low"]
     patience: 5

With HPO
^^^^^^^^

Curriculum learning can be enabled during HPO, though shorter trial durations
may limit its effectiveness:

.. code-block:: yaml

   # In HPO config
   curriculum:
     enabled: true
     patience: 3  # Lower patience for shorter trials
     qualities: ["high", "medium", "low"]

Consider whether curriculum benefits outweigh the overhead for short HPO trials.

Troubleshooting
---------------

No Phase Transitions
^^^^^^^^^^^^^^^^^^^^

If the curriculum stays in the warmup phase:

1. **Check patience**: May be too high relative to training epochs
2. **Verify quality labels**: Ensure labels match ``qualities`` list exactly
3. **Check validation loss**: If constantly improving, phases won't advance

.. code-block:: python

   # Debug: print phase and weights during training
   print(f"Phase: {model.curriculum_state.phase}")
   print(f"Weights: {model.curriculum_state.weights}")
   print(f"Best epoch: {model.curriculum_state.best_epoch}")

Unknown Quality Labels
^^^^^^^^^^^^^^^^^^^^^^

Warning: "Quality labels {'unknown'} not in curriculum qualities"

This means your data contains quality labels not in the ``qualities`` list.
Samples with unknown labels get zero weight (never sampled).

Solution: Add missing quality levels to the config or clean your data.

All Zero Weights
^^^^^^^^^^^^^^^^

Warning: "All sample weights are zero. Falling back to uniform sampling."

This happens when no samples match the current phase's non-zero quality levels.
The sampler falls back to uniform sampling to avoid empty batches.

API Reference
-------------

See the API documentation for detailed class and function references:

- :mod:`admet.model.chemprop.curriculum` - CurriculumState and CurriculumCallback
- :mod:`admet.model.chemprop.curriculum_sampler` - Weighted sampler utilities
- :mod:`admet.model.chemprop.config` - CurriculumConfig dataclass

Cross-References
----------------

- See :doc:`modeling` for general modeling guide
- See :doc:`hpo` for hyperparameter optimization
- See :doc:`configuration` for configuration file format

