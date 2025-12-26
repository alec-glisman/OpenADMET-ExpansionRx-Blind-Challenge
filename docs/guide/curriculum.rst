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

The curriculum proceeds through four phases with conservative default proportions:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Phase
     - Target Proportions (3-quality)
     - Description
   * - **Warmup**
     - 80% high, 15% medium, 5% low
     - Focus on high-quality data to learn core patterns
   * - **Expand**
     - 60% high, 30% medium, 10% low
     - Gradually incorporate medium-quality data
   * - **Robust**
     - 50% high, 35% medium, 15% low
     - Include low-quality data for robustness
   * - **Polish**
     - 70% high, 20% medium, 10% low
     - Fine-tune on high-quality data while maintaining diversity

The **polish phase** maintains 30% non-high-quality data (20% medium + 10% low)
to prevent overfitting to high-quality examples and preserve learned robustness.

Phase transitions occur automatically when the overall validation loss stops
improving for ``patience`` epochs.

Count Normalization
^^^^^^^^^^^^^^^^^^^

**Why count normalization matters**: In ADMET datasets, quality levels often have
vastly different sizes. For example:

- High quality: 5,000 samples (4%)
- Medium quality: 100,000 samples (83%)
- Low quality: 15,000 samples (13%)

Without count normalization, setting weights ``[0.8, 0.15, 0.05]`` does NOT mean
80% of your batches will contain high-quality samples. The medium-quality dataset
has 20× more samples, so it dominates training regardless of weights.

With ``count_normalize=True`` (default), the target proportions are achieved by
adjusting per-sample weights:

.. math::

   weight_{sample} = \frac{target\_proportion}{count}

This ensures that specifying ``[0.8, 0.15, 0.05]`` actually results in 80% of
training batches containing high-quality samples.

Metric Alignment
^^^^^^^^^^^^^^^^

**Why metric alignment matters**: By default, both curriculum phase transitions
and early stopping monitor ``val_loss`` (overall validation loss). However, if
your **test data is entirely high-quality**, this creates metric misalignment:

- With High=5k, Medium=100k: overall ``val_loss`` is dominated by medium-quality
- Model checkpoints optimize for medium-quality predictions, not high-quality test

With ``monitor_metric: "val/mae/high"``, you align optimization with your actual
evaluation metric:

.. code-block:: yaml

   curriculum:
     monitor_metric: "val/mae/high"  # Monitor high-quality validation MAE
     early_stopping_metric: null     # If null, uses monitor_metric

Available metric patterns:

- ``val_loss`` - Overall validation loss (default)
- ``val/mae/high`` - High-quality validation MAE
- ``val/rmse/high`` - High-quality validation RMSE
- ``val/mae/medium``, ``val/mae/low`` - Other quality levels

Adaptive Curriculum
^^^^^^^^^^^^^^^^^^^

**What is adaptive curriculum?** Instead of using fixed phase proportions, the
adaptive curriculum automatically adjusts sampling weights based on per-quality
validation performance trends.

**How it works:**

1. Track per-quality metrics (e.g., ``val/mae/high``, ``val/mae/medium``) over time
2. Compute relative improvement rates for each quality level
3. If high-quality improves faster than others → increase high-quality proportion
4. If high-quality lags behind → decrease high-quality proportion (bounded by floor)

.. code-block:: yaml

   curriculum:
     adaptive_enabled: true
     adaptive_improvement_threshold: 0.02  # 2% relative improvement triggers adjustment
     adaptive_max_adjustment: 0.1          # Max 10% weight change per epoch
     adaptive_lookback_epochs: 5           # Compare current vs 5 epochs ago

**Parameters:**

- ``adaptive_improvement_threshold``: Minimum relative improvement gap required
- ``adaptive_max_adjustment``: Maximum proportion change per adjustment
- ``adaptive_lookback_epochs``: How far back to look for trend computation
- ``min_high_quality_proportion``: Safety floor (default 0.25)

Loss Weighting
^^^^^^^^^^^^^^

**Complementing sampling with loss weighting**: While curriculum sampling controls
*which* samples appear in batches, loss weighting controls *how much* each sample's
gradient contributes to learning.

.. code-block:: yaml

   curriculum:
     loss_weighting_enabled: true
     loss_weights:
       high: 1.0     # Full gradient weight
       medium: 0.5   # Half gradient weight
       low: 0.3      # 30% gradient weight

**When to use loss weighting:**

- Combined with sampling for stronger high-quality emphasis
- When you want gradients from all data but prioritize high-quality
- Alternative to aggressive sampling proportions

**Note:** Loss weighting applies per-sample weights during the forward pass via
Chemprop's built-in sample weight mechanism.

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

       # Count normalization (recommended for imbalanced datasets)
       count_normalize: true
       min_high_quality_proportion: 0.25

       # Metric alignment (NEW): monitor high-quality metrics
       monitor_metric: "val/mae/high"
       early_stopping_metric: null  # Uses monitor_metric if null

       # Adaptive curriculum (NEW): auto-adjust proportions
       adaptive_enabled: false

       # Loss weighting (NEW): scale gradients by quality
       loss_weighting_enabled: false
       loss_weights:
         high: 1.0
         medium: 0.5
         low: 0.3

       # Optional: customize phase proportions for HPO
       # warmup_proportions: [0.80, 0.15, 0.05]
       # expand_proportions: [0.60, 0.30, 0.10]
       # robust_proportions: [0.50, 0.35, 0.15]
       # polish_proportions: [0.70, 0.20, 0.10]
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

       # Count normalization for imbalanced datasets (default: true)
       count_normalize: true
       min_high_quality_proportion: 0.25

       # Metric alignment: monitor high-quality metrics
       monitor_metric: "val/mae/high"
       early_stopping_metric: null  # Uses monitor_metric if null

       # Adaptive curriculum (auto-adjust proportions)
       adaptive_enabled: false
       adaptive_improvement_threshold: 0.02
       adaptive_max_adjustment: 0.1
       adaptive_lookback_epochs: 5

       # Loss weighting (scale gradients by quality)
       loss_weighting_enabled: false
       loss_weights:
         high: 1.0
         medium: 0.5
         low: 0.3

       # Optional: HPO-friendly phase proportions
       # warmup_proportions: [0.80, 0.15, 0.05]
       # expand_proportions: [0.60, 0.30, 0.10]
       # robust_proportions: [0.50, 0.35, 0.15]
       # polish_proportions: [0.70, 0.20, 0.10]

     # Global sampling settings
     num_samples: null  # null = use full dataset size per epoch
     seed: 42
     increment_seed_per_epoch: true
     log_to_mlflow: true

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

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
   * - ``count_normalize``
     - ``true``
     - Adjust weights for dataset size imbalance (recommended)
   * - ``min_high_quality_proportion``
     - ``0.25``
     - Safety floor: minimum high-quality proportion in any phase
   * - ``monitor_metric``
     - ``"val_loss"``
     - Metric for curriculum phase transitions (e.g., ``val/mae/high``)
   * - ``early_stopping_metric``
     - ``null``
     - Metric for early stopping; uses ``monitor_metric`` if null
   * - ``adaptive_enabled``
     - ``false``
     - Enable adaptive proportion adjustment based on per-quality trends
   * - ``adaptive_improvement_threshold``
     - ``0.02``
     - Minimum relative improvement gap to trigger adjustment (2%)
   * - ``adaptive_max_adjustment``
     - ``0.1``
     - Maximum proportion change per adjustment (10%)
   * - ``adaptive_lookback_epochs``
     - ``5``
     - Epochs to look back for trend computation
   * - ``loss_weighting_enabled``
     - ``false``
     - Enable per-quality loss weights for gradient scaling
   * - ``loss_weights``
     - ``null``
     - Quality-to-weight mapping (e.g., ``{high: 1.0, medium: 0.5}``)
   * - ``warmup_proportions``
     - ``[0.80, 0.15, 0.05]``
     - Target proportions [high, medium, low] for warmup phase
   * - ``expand_proportions``
     - ``[0.60, 0.30, 0.10]``
     - Target proportions for expand phase
   * - ``robust_proportions``
     - ``[0.50, 0.35, 0.15]``
     - Target proportions for robust phase
   * - ``polish_proportions``
     - ``[0.70, 0.20, 0.10]``
     - Target proportions for polish phase (maintains diversity)

Adaptive Quality Levels
^^^^^^^^^^^^^^^^^^^^^^^

The curriculum adapts to the number of quality levels provided:

- **1 quality**: warmup → polish (effectively no curriculum)
- **2 qualities**: warmup → expand → polish (defaults: [0.85, 0.15] → [0.65, 0.35] → [0.75, 0.25])
- **3+ qualities**: warmup → expand → robust → polish (full curriculum)

Example with two quality levels:

.. code-block:: yaml

   curriculum:
     enabled: true
     quality_col: "Quality"
     qualities:
       - "reliable"
       - "uncertain"
     patience: 5
     count_normalize: true

     # Optional: customize 2-quality proportions
     # warmup_proportions: [0.85, 0.15]
     # expand_proportions: [0.65, 0.35]
     # polish_proportions: [0.75, 0.25]

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

Two-Stage Sampling with JointSampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``JointSampler`` provides unified sampling that combines task-aware oversampling
with curriculum learning via a two-stage algorithm:

**Stage 1: Task Selection**

Sample a task ``t`` according to inverse-power probabilities:

.. math::

   p_t \propto count_t^{-\alpha}

Where:

- ``count_t`` = number of samples with labels for task ``t``
- ``α ∈ [0, 1]`` controls rebalancing strength:
  - ``α=0``: Uniform task sampling (no rebalancing)
  - ``α=0.5``: Moderate rebalancing (default, recommended)
  - ``α=1.0``: Full inverse-proportional (rare tasks heavily favored)

**Stage 2: Within-Task Sampling**

Sample a molecule from task ``t``'s valid indices, weighted by curriculum:

.. math::

   p_i \propto w_{curriculum}[i]

Where ``w_curriculum[i]`` is determined by the sample's quality label and
the current curriculum phase proportions.

**Weight Composition**

The joint weight for sample ``i`` is computed as:

.. math::

   w_{joint}[i] = w_{task}[i] \times w_{curriculum}[i]

This preserves backward compatibility:

- When curriculum is disabled: reduces to task oversampling only
- When ``task_alpha=0``: curriculum weights control all sampling
- When both enabled: multiplicative composition balances both strategies

**Important: num_workers Limitation**

.. warning::

   When using ``JointSampler`` with ``num_workers > 0`` in DataLoader, each
   worker gets its own copy of the sampler. The internal ``_current_epoch``
   counter and curriculum phase state will NOT be synchronized across workers,
   potentially causing inconsistent sampling behavior.

   **Recommendation**: Use ``num_workers=0`` for reliable curriculum learning.

Sampling-Based Curriculum (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The legacy ``DynamicCurriculumSampler`` uses weighted random sampling to control
which data points appear in each training batch:

1. **WeightedRandomSampler**: Each sample gets a weight based on its quality
   and the current curriculum phase
2. **Sampling with replacement**: Samples are drawn according to weights,
   so high-quality samples appear more frequently in early phases
3. **Phase-dependent weights**: Weights are updated when phases change
4. **Count normalization**: Weights are adjusted for dataset size imbalance

The ``JointSampler`` is the preferred approach as it unifies task and curriculum
sampling in a single, well-tested implementation.

Count-Normalized Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^

When ``count_normalize=True`` (default), the sampler converts target proportions
to per-sample weights that achieve those proportions regardless of dataset size:

.. code-block:: python

   # Example: High=5k, Medium=100k, Low=15k samples
   # Target proportions for warmup: [0.80, 0.15, 0.05]

   # Without count normalization (legacy):
   #   Raw weights: [0.80, 0.15, 0.05] applied to each sample
   #   Result: ~4% high (dominated by medium's 20x size!)

   # With count normalization:
   #   Normalized weights:
   #     high:   0.80 / 5000   = 0.00016 per sample
   #     medium: 0.15 / 100000 = 0.0000015 per sample
   #     low:    0.05 / 15000  = 0.0000033 per sample
   #   Result: ~80% high in batches (as intended!)

   from admet.model.chemprop.curriculum_sampler import DynamicCurriculumSampler
   from admet.model.chemprop.curriculum import CurriculumState, CurriculumPhaseConfig

   # Create curriculum state with count normalization
   config = CurriculumPhaseConfig(count_normalize=True)
   state = CurriculumState(qualities=["high", "medium", "low"], config=config)

   # Build sampler for training data
   quality_labels = df_train["Quality"].tolist()
   sampler = DynamicCurriculumSampler(
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

Weight Statistics Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``JointSampler`` computes and logs comprehensive weight statistics each epoch
to help monitor sampling behavior:

.. code-block:: python

   def get_weight_statistics(weights: np.ndarray) -> dict[str, float]:
       return {
           "min": float,           # Minimum weight
           "max": float,           # Maximum weight
           "mean": float,          # Mean weight
           "entropy": float,       # Distribution uniformity (H = -sum(p*log(p)))
           "effective_samples": float,  # 1/sum(weights^2) - higher = more uniform
       }

**Interpretation:**

- **entropy**: Higher values indicate more uniform sampling; lower values indicate
  concentration on specific samples
- **effective_samples**: The "effective" number of samples being used. A value close
  to the dataset size indicates uniform sampling; a low value indicates sampling is
  concentrated on a small subset

These statistics are logged as:

.. code-block:: text

   Weight stats: min=0.000001, max=0.000160, mean=0.000008, entropy=11.234, effective_samples=4523.1

MLflow Metrics
^^^^^^^^^^^^^^

When MLflow tracking is enabled, the following metrics are logged with hierarchical naming:

**Curriculum State Metrics:**

- ``curriculum/phase``: Numeric phase indicator (0=warmup, 1=expand, 2=robust, 3=polish)
- ``curriculum/phase_epoch``: Epoch when phase transition occurred
- ``curriculum/val_loss_at_transition``: Validation loss at the time of phase transition
- ``curriculum/transition``: Phase transition marker (logged at global_step for training curves)
- ``curriculum/weight/<quality>``: Sampling weight for each quality level (e.g., ``curriculum/weight/high``)

**Per-Quality Metrics (during training):**

When ``log_per_quality_metrics: true`` is enabled, per-quality metrics are computed
and logged on each epoch for both training and validation, enabling training curve visualization:

*Validation metrics:*

- ``val/<metric>/<quality>``: Metric for a specific quality level

  Examples:

  - ``val/mae/high``: Mean absolute error for high-quality validation samples
  - ``val/mse/medium``: Mean squared error for medium-quality validation samples
  - ``val/rmse/low``: Root mean squared error for low-quality validation samples
  - ``val/loss/high``: Loss for high-quality samples (same as MSE)
  - ``val/count/high``: Number of validation samples for high-quality

*Training metrics:*

- ``train/<metric>/<quality>``: Metric for a specific quality level during training

  Examples:

  - ``train/mae/high``: Mean absolute error for high-quality training samples
  - ``train/mse/medium``: Mean squared error for medium-quality training samples
  - ``train/rmse/low``: Root mean squared error for low-quality training samples

*Per-target metrics (when multiple targets):*

- ``<split>/<metric>/<quality>/<target>``: Metric for a specific quality level and target

  Examples:

  - ``val/mae/high/LogD``: MAE for high-quality validation samples on LogD target
  - ``train/rmse/medium/KSOL``: RMSE for medium-quality training samples on KSOL target

**MLflow Tags:**

- ``curriculum_transition_epoch_<N>``: Tag marking the epoch and transition type (e.g., ``warmup_to_expand``)

These metrics enable visualization of curriculum progression in MLflow UI. The hierarchical
naming groups related metrics together for easier navigation.

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

With count normalization enabled (default), the curriculum works effectively
even with highly imbalanced quality distributions:

.. code-block:: python

   # Check quality distribution
   print(df_train["Quality"].value_counts(normalize=True))
   # high      0.04   # Only 4% high-quality is fine with count_normalize=True
   # medium    0.83   # Large medium-quality dataset
   # low       0.13   # Some low-quality data

   # The curriculum will still achieve target proportions:
   # - warmup: 80% high, 15% medium, 5% low in actual batches
   # - polish: 70% high, 20% medium, 10% low in actual batches

Without count normalization, aim for a reasonable distribution across quality levels.

Patience Tuning
^^^^^^^^^^^^^^^

The ``patience`` parameter balances exploration vs. exploitation:

- **Lower patience (3-5)**: Faster phase transitions, less time in each phase
- **Higher patience (8-15)**: More thorough learning in each phase

Start with ``patience=5`` and adjust based on:

- Total training epochs
- Validation loss convergence behavior
- Dataset size and complexity

Phase Proportion Tuning
^^^^^^^^^^^^^^^^^^^^^^^

The default proportions are conservative and designed for molecular property
prediction with imbalanced datasets:

.. code-block:: yaml

   # Conservative defaults (good starting point)
   warmup_proportions: [0.80, 0.15, 0.05]   # 80% high
   expand_proportions: [0.60, 0.30, 0.10]   # 60% high
   robust_proportions: [0.50, 0.35, 0.15]   # 50% high (minimum with safety floor)
   polish_proportions: [0.70, 0.20, 0.10]   # 70% high (maintains diversity)

   # More aggressive (favor high-quality more)
   warmup_proportions: [0.90, 0.08, 0.02]
   expand_proportions: [0.75, 0.20, 0.05]
   robust_proportions: [0.60, 0.30, 0.10]
   polish_proportions: [0.85, 0.10, 0.05]

   # More inclusive (use more low-quality data)
   warmup_proportions: [0.70, 0.20, 0.10]
   expand_proportions: [0.50, 0.35, 0.15]
   robust_proportions: [0.40, 0.35, 0.25]
   polish_proportions: [0.60, 0.25, 0.15]

The ``min_high_quality_proportion`` (default: 0.25) ensures high-quality data
never drops below 25% in any phase, preventing catastrophic forgetting.

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

- :mod:`admet.model.chemprop.joint_sampler` - JointSampler (unified two-stage sampling)
- :mod:`admet.model.chemprop.curriculum` - CurriculumState and CurriculumCallback
- :mod:`admet.model.chemprop.curriculum_sampler` - DynamicCurriculumSampler (legacy)
- :mod:`admet.model.chemprop.config` - JointSamplingConfig, CurriculumConfig dataclasses

Cross-References
----------------

- See :doc:`modeling` for general modeling guide
- See :doc:`hpo` for hyperparameter optimization
- See :doc:`configuration` for configuration file format
