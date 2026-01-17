Migration Guide
===============

This guide helps you upgrade your configuration files and code when moving between major versions of the OpenADMET Challenge package.

.. contents:: On this page
   :local:
   :depth: 2

Version 1.2 → 1.3
-----------------

**Release Date:** 2026-01-11

Performance Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.3 introduces a new ``performance_optimization`` configuration section. Existing configurations continue to work unchanged (all optimizations are disabled by default).

**New optional fields:**

.. code-block:: yaml

   # configs/example.yaml
   performance_optimization:
     use_mixed_precision: false              # Enable AMP for 40-60% faster training
     async_checkpoint_upload: false          # Background MLflow uploads
     checkpoint_save_interval_seconds: 0.0   # Throttle checkpoint saves

   optimization:
     accumulate_grad_batches: 1              # Gradient accumulation steps

   post_training:
     use_gpu_metrics: auto                   # "auto", "true", or "false"

GPU Metrics Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``post_training.use_gpu_metrics`` field now supports three modes:

- ``"auto"`` (default): Automatically detect GPU availability
- ``"true"``: Force GPU metrics computation
- ``"false"``: Force CPU metrics computation

**Backward compatibility:** Boolean values (``true``/``false``) continue to work.

Version 1.1 → 1.2
-----------------

**Release Date:** 2025-12-15

Joint Sampling Schema
~~~~~~~~~~~~~~~~~~~~~

The curriculum learning and task oversampling configurations were consolidated under a unified ``joint_sampling`` section.

**Old configuration (deprecated):**

.. code-block:: yaml

   # OLD (v1.1)
   optimization:
     task_sampling_alpha: 0.5

   curriculum:
     enabled: true
     warmup_epochs: 5
     quality_threshold: 0.8

**New configuration:**

.. code-block:: yaml

   # NEW (v1.2+)
   joint_sampling:
     enabled: true
     task_oversampling:
       alpha: 0.5
     curriculum:
       enabled: true
       warmup_epochs: 5
       quality_threshold: 0.8

**Migration script:** ``legacy/migrate_sampling_configs.py`` (for reference only)

Version 1.0 → 1.1
-----------------

**Release Date:** 2025-11-01

Unified Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration schema was restructured to use a discriminator pattern for multi-model support.

**Old configuration (deprecated):**

.. code-block:: yaml

   # OLD (v1.0)
   model:
     depth: 5
     message_hidden_dim: 600
     ffn_hidden_dim: 512
     ffn_num_layers: 3

**New configuration:**

.. code-block:: yaml

   # NEW (v1.1+)
   model:
     type: chemprop  # Discriminator: chemprop, chemeleon, xgboost, lightgbm, catboost
     chemprop:
       depth: 5
       message_hidden_dim: 600
       ffn_hidden_dim: 512
       ffn_num_layers: 3

This change enables support for multiple model types within the same configuration schema.

**Migration script:** ``legacy/config_migration.py`` (for reference only)

Model Registry Pattern
~~~~~~~~~~~~~~~~~~~~~~

The ``create_model_from_config()`` function was deprecated in favor of the ``ModelRegistry`` pattern:

**Old code (deprecated):**

.. code-block:: python

   # OLD
   from admet.model.chemprop import create_model_from_config

   model = create_model_from_config(config)

**New code:**

.. code-block:: python

   # NEW
   from admet.model import ModelRegistry

   model = ModelRegistry.create(config)

Deprecated Features
-------------------

Curriculum Learning
~~~~~~~~~~~~~~~~~~~

.. warning::

   Curriculum learning is deprecated and will be removed in a future release.
   Use task oversampling with ``joint_sampling.task_oversampling`` instead.

**Reason:** Experiments showed that curriculum learning provided minimal benefit
compared to simpler task oversampling strategies, while adding significant
complexity to the training pipeline.

**Migration:** Set ``joint_sampling.curriculum.enabled: false`` and use
``joint_sampling.task_oversampling.alpha`` for task weighting.

``admet model train-chemprop`` CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. deprecated:: 1.2.0

   Use ``admet model train`` instead.

The ``train-chemprop`` subcommand was consolidated into the generic ``train`` command:

.. code-block:: bash

   # OLD
   admet model train-chemprop -c config.yaml

   # NEW
   admet model train -c config.yaml

Legacy Scripts
--------------

Migration scripts are preserved in the ``legacy/`` folder for historical reference.
See ``legacy/README.md`` for details on each script's purpose.

.. warning::

   Do not run legacy migration scripts on current configurations.
   They are retained for historical reference only.
