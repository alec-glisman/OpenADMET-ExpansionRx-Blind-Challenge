Changelog
=========

All notable changes to the OpenADMET project are documented here. For upgrade
instructions, see the :doc:`migration`.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. contents:: Versions
   :local:
   :depth: 1

Version 1.3.0 (2026-01-11)
--------------------------

Performance Optimization Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release focuses on training speed and resource efficiency without changing
model quality.

Added
^^^^^

**Core Optimizations**

- **Mixed Precision Training (AMP)**: Automatic mixed precision using PyTorch
  Lightning's native FP16 mode. Enable with ``performance_optimization.use_mixed_precision: true``.
  Expected speedup: 40-60% faster training with 30% lower GPU memory.

- **Asynchronous Checkpoint Uploads**: Background thread for non-blocking MLflow
  artifact uploads. Enable with ``performance_optimization.async_checkpoint_upload: true``.
  Expected speedup: 5-10% reduction in I/O wait time.

- **Checkpoint Save Throttling**: Configurable minimum interval between saves.
  Set ``performance_optimization.checkpoint_save_interval_seconds`` to prevent
  excessive checkpoint I/O during rapid model improvement.

**GPU Acceleration**

- **GPU Metrics Computation**: Automatic GPU detection for post-training metrics.
  Configure with ``post_training.use_gpu_metrics`` (values: ``"auto"``, ``"true"``, ``"false"``).
  Expected speedup: 2-5× faster metrics computation when GPU available.

**Training Configuration**

- **Gradient Accumulation**: Simulate larger batch sizes with
  ``optimization.accumulate_grad_batches``. Allows larger effective batch sizes
  without OOM errors.

**Documentation**

- Added profiling guide for ensemble training: :doc:`profiling`
- Added migration guide: :doc:`migration`

Testing
^^^^^^^

- 23 new unit tests for performance optimization features
- ≥90% coverage for all new code paths

Backward Compatibility
^^^^^^^^^^^^^^^^^^^^^^

- **Zero breaking changes**: All existing configurations work unchanged
- All optimizations disabled by default (conservative defaults)
- Automatic fallbacks: GPU metrics fall back to CPU if unavailable

Version 1.2.0 (2025-12-15)
--------------------------

Joint Sampling & Task Optimization Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

- **Joint Sampling Configuration**: Unified ``joint_sampling`` section consolidating
  curriculum learning and task oversampling settings.

- **Task Oversampling**: New ``joint_sampling.task_oversampling.alpha`` parameter
  for task-weighted sampling during training.

- **Enhanced Logging**: Structured JSON logging with configurable verbosity.

Changed
^^^^^^^

- Migrated from flat ``optimization.task_sampling_alpha`` and ``curriculum``
  sections to nested ``joint_sampling`` structure.

Deprecated
^^^^^^^^^^

- ``admet model train-chemprop`` CLI command. Use ``admet model train`` instead.
- Flat curriculum configuration. Use ``joint_sampling.curriculum`` instead.

Version 1.1.0 (2025-11-01)
--------------------------

Unified Model Configuration Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

- **UnifiedModelConfig**: Single configuration schema supporting multiple model
  types (Chemprop, Chemeleon, XGBoost, LightGBM, CatBoost).

- **ModelRegistry**: Factory pattern for model instantiation. Use
  ``ModelRegistry.create(config)`` instead of model-specific factories.

- **Multi-Model Support**: Train and compare different model architectures using
  the same configuration structure with ``model.type`` discriminator.

Changed
^^^^^^^

- Configuration schema restructured to use discriminator pattern. Model parameters
  now nested under ``model.chemprop``, ``model.xgboost``, etc.

Deprecated
^^^^^^^^^^

- ``create_model_from_config()`` function. Use ``ModelRegistry.create()`` instead.

Version 1.0.0 (2025-10-01)
--------------------------

Initial Release
~~~~~~~~~~~~~~~

- Chemprop MPNN model for multi-task ADMET prediction
- 5×5 cross-validation with BitBirch cluster-aware splitting
- Ray Tune hyperparameter optimization with ASHA scheduler
- MLflow experiment tracking
- CLI interface with train, ensemble, and HPO commands
- Support for 9 ADMET endpoints

See Also
--------

- :doc:`migration` - Upgrade instructions between versions
- `GitHub Releases <https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge/releases>`_ - Full release notes
