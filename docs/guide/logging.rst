.. _ray_tune_logging:

=====================================
Ray Tune Output Logging to MLflow
=====================================

This guide explains the Ray Tune logging infrastructure for the OpenADMET Challenge, which automatically collects Ray trial logs and uploads them to MLflow artifacts for storage and inspection.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The Ray Tune logging system provides:

- **Automatic Log Collection**: Captures all Ray trial output logs and aggregates them
- **Log Compression**: Compresses logs into tar.gz archives to reduce storage requirements
- **MLflow Integration**: Uploads compressed logs to MLflow artifacts for centralized access
- **Progress Tracking**: Real-time progress reporting with ETA calculations during ensemble training
- **Signal Handling**: Gracefully handles SIGINT/SIGTERM to ensure logs are uploaded even on interruption
- **Configurable Verbosity**: Control logging detail level (quiet, normal, verbose)
- **Size Enforcement**: Enforces maximum total log size limits to prevent disk overflow

Quick Start
===========

Enable logging in your config file:

.. code-block:: yaml

   logging:
     enabled: true
     verbose: 1                    # 0=quiet, 1=normal, 2=verbose
     max_total_logs_gb: 1.0        # Max total size (GB) for compressed logs
     fail_on_upload_error: true    # Fail if MLflow upload fails

Then run HPO or Ensemble training as normal:

.. code-block:: bash

   # HPO with logging (verbose)
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml --logging-verbose 2

   # Ensemble with logging (normal)
   admet model ensemble -c configs/3-production/ensemble_chemprop_hpo_001.yaml

   # Disable logging for a run
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml --no-logging

Configuration
==============

Logging Configuration Schema
-----------------------------

The logging configuration is defined by the ``RayLoggingConfig`` dataclass:

.. code-block:: python

   @dataclass
   class RayLoggingConfig:
       """Ray Tune logging configuration."""
       enabled: bool = True
       verbose: int = 0              # 0=quiet, 1=normal, 2=verbose
       max_total_logs_gb: float = 1.0
       fail_on_upload_error: bool = True

**Parameters**:

- **enabled** (bool, default=True): Enable/disable Ray Tune logging
- **verbose** (int, default=0): Logging verbosity level

  - 0 = quiet (minimal output)
  - 1 = normal (standard progress reporting)
  - 2 = verbose (detailed debug information)

- **max_total_logs_gb** (float, default=1.0): Maximum total size (in GB) for compressed logs. Logs larger than this limit are truncated.
- **fail_on_upload_error** (bool, default=True): Whether to fail the entire run if MLflow artifact upload fails

Configuration File Example
---------------------------

Add the logging section to your YAML config:

.. code-block:: yaml

   # model/hpo config sections above...

   logging:
     enabled: true
     verbose: 1
     max_total_logs_gb: 2.0
     fail_on_upload_error: false    # Don't fail if MLflow is unavailable

   # rest of config below...

All 131 YAML configuration files in the ``configs/`` directory have been automatically updated with default logging sections.

CLI Usage
=========

HPO Command
-----------

Train HPO with logging control:

.. code-block:: bash

   # Default logging from config file
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml

   # Override verbosity level
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml --logging-verbose 2

   # Disable logging for this run
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml --no-logging

Ensemble Command
----------------

Train ensemble with logging control:

.. code-block:: bash

   # Default logging from config file
   admet model ensemble -c configs/3-production/ensemble_chemprop_hpo_001.yaml

   # Override verbosity level
   admet model ensemble -c configs/3-production/ensemble_chemprop_hpo_001.yaml --logging-verbose 1

   # Disable logging for this run
   admet model ensemble -c configs/3-production/ensemble_chemprop_hpo_001.yaml --no-logging

Available Flags
~~~~~~~~~~~~~~~

- ``--logging-verbose N``: Override verbosity level (0-2). Omit to use config value.
- ``--no-logging``: Disable logging for this run. Overrides config file.

API Reference
=============

RayLogManager
-------------

Context manager for Ray Tune logging with log collection, compression, and MLflow upload.

.. code-block:: python

   class RayLogManager:
       def __init__(
           self,
           mlflow_run_id: str,
           output_dir: Path,
           verbose: int = 0,
           max_total_logs_gb: float = 1.0,
           fail_on_upload_error: bool = True,
       ) -> None:
           """
           Initialize RayLogManager.

           Parameters
           ----------
           mlflow_run_id : str
               MLflow run ID for artifact logging
           output_dir : Path
               Output directory containing trial logs
           verbose : int, default 0
               Logging verbosity level (0-2)
           max_total_logs_gb : float, default 1.0
               Maximum total log size in GB
           fail_on_upload_error : bool, default True
               Raise exception if MLflow upload fails
           """

       def __enter__(self) -> "RayLogManager":
           """Enter context manager."""

       def __exit__(self, exc_type, exc_val, exc_tb) -> None:
           """Exit context manager, triggering log upload."""

**Example**:

.. code-block:: python

   from pathlib import Path
   from admet.util.ray_logging import RayLogManager

   with RayLogManager(
       mlflow_run_id="abc123",
       output_dir=Path("/tmp/ray_results"),
       verbose=1,
       max_total_logs_gb=2.0,
       fail_on_upload_error=False,
   ):
       # Run Ray Tune HPO or Ensemble training here
       # Logs will be automatically collected and uploaded on exit

QuietProgressReporter
---------------------

Custom Ray Tune progress reporter with minimal output:

.. code-block:: python

   class QuietProgressReporter(CLIReporter):
       """
       Ray Tune progress reporter with minimal output.

       Updates every 5 seconds with: [Completed / Total | Running | Errored]
       """

       def report_progress(self, trials, done=False) -> None:
           """Report progress at regular intervals."""

**Features**:

- Periodic updates (5-second intervals)
- Compact format: ``[5/10 | 2 running | 1 errored]``
- Minimal terminal output compared to default CLIReporter

EnsembleProgressTracker
-----------------------

Progress tracker with ETA calculation for ensemble training:

.. code-block:: python

   class EnsembleProgressTracker:
       def __init__(self, total_tasks: int, verbose: int = 0) -> None:
           """
           Initialize progress tracker.

           Parameters
           ----------
           total_tasks : int
               Total number of ensemble models to train
           verbose : int, default 0
               Logging verbosity level
           """

       def update(self, completed: int) -> None:
           """
           Update progress.

           Parameters
           ----------
           completed : int
               Number of completed training tasks
           """

**Example**:

.. code-block:: python

   from admet.util.ray_logging import EnsembleProgressTracker

   tracker = EnsembleProgressTracker(total_tasks=25, verbose=1)

   for i in range(1, 26):
       # Train model...
       tracker.update(completed=i)

LogArtifactCallback
-------------------

Ray Tune callback for integration with the logging system:

.. code-block:: python

   class LogArtifactCallback(Callback):
       """Ray Tune callback for log collection on experiment completion."""

       def on_experiment_end(self, algorithm: Any, **info) -> None:
           """Trigger log collection when experiment completes."""

Troubleshooting
===============

Logs Not Appearing in MLflow
----------------------------

**Problem**: Logs are collected but not visible in MLflow UI

**Solutions**:

1. Check MLflow is initialized and tracking URI is set:

   .. code-block:: bash

      mlflow ui  # Check MLflow server is running

2. Verify MLflow run ID is correctly passed to logging context manager
3. Check log file size hasn't exceeded ``max_total_logs_gb`` limit
4. Review log compression output in console for errors

Memory Issues During Training
------------------------------

**Problem**: High memory usage during log collection

**Solutions**:

1. Reduce ``max_total_logs_gb`` limit (trade-off: older logs may be truncated)
2. Increase number of parallel tasks to reduce single-trial log volume
3. Enable verbose logging (``verbose: 2``) to see memory-related debug messages

MLflow Upload Failures
----------------------

**Problem**: Training stops due to failed MLflow upload

**Solutions**:

1. Set ``fail_on_upload_error: false`` to continue training even if upload fails
2. Check network connectivity to MLflow server
3. Verify MLflow tracking URI is correct:

   .. code-block:: bash

      echo $MLFLOW_TRACKING_URI

4. Check available disk space on MLflow server

Ray Cluster Interruption
------------------------

**Problem**: Training interrupted by SIGINT/SIGTERM

**Solution**: RayLogManager automatically handles these signals and uploads logs before shutdown. Check MLflow for partial logs from the interrupted run.

Ray Trial Logs Not Collected
-----------------------------

**Problem**: Expected trial log files not found in output directory

**Solutions**:

1. Verify Ray is configured to write logs to expected directory (check ``RAY_LOG_TO_DRIVER``)
2. Check trial directories exist in output folder:

   .. code-block:: bash

      ls -la <output_dir>/trial_*/logs/

3. Enable verbose logging (``verbose: 2``) to see collection debug messages

Performance Impact
==================

Logging Overhead
----------------

**Log Collection**: Minimal impact (background process, non-blocking)

- Typical overhead: 1-2% CPU
- Memory peak during compression: ~100MB for 1GB uncompressed logs

**Log Compression**: Time varies by size and compression level

- 100MB logs → 20-30MB compressed: ~2-5 seconds
- 1GB logs → 200-300MB compressed: ~20-40 seconds

**MLflow Upload**: Network-dependent

- Same network: 1-10 seconds for typical logs
- Remote network: 10-60 seconds depending on bandwidth

Disk Space Requirements
-----------------------

**Uncompressed Logs**: Typically 10-20 MB per trial

- 50 trials × 15 MB = 750 MB per HPO experiment
- 25 models × 15 MB = 375 MB per Ensemble training run

**Compressed Logs**: 20-25% of original size

- 750 MB uncompressed → ~150-200 MB compressed
- 375 MB uncompressed → ~75-100 MB compressed

**MLflow Artifacts**: Same as compressed size

- Artifacts stored on MLflow backend (disk, S3, etc.)
- Compression reduces artifact storage by 75-80%

Best Practices
==============

1. **Enable Logging by Default**: Keep ``enabled: true`` in production configs for better observability

2. **Set Appropriate Verbosity**:
   - Use ``verbose: 0`` (quiet) for production runs
   - Use ``verbose: 1`` (normal) for development
   - Use ``verbose: 2`` (verbose) for debugging issues

3. **Configure Size Limits**:
   - Set ``max_total_logs_gb: 1.0`` for typical runs
   - Increase to ``2.0`` or ``3.0`` for long-running experiments
   - Monitor disk usage to prevent overflow

4. **Handle Upload Failures Gracefully**:
   - Set ``fail_on_upload_error: false`` in production
   - Train continues even if MLflow is temporarily unavailable
   - Manual upload can be done later if needed

5. **Monitor Disk Space**:
   - Keep 2-3 GB free for log collection
   - Archive old logs to external storage
   - Use ``df -h`` to monitor available space

6. **Review Logs Regularly**:
   - Check MLflow UI for trial metrics and logs
   - Investigate failed trials early
   - Delete old artifacts to free space

Advanced Usage
==============

Custom Verbosity Levels
-----------------------

Override config verbosity at runtime:

.. code-block:: bash

   # Always use verbose logging for debugging
   admet model hpo -c config.yaml --logging-verbose 2

   # Disable logging for quick test run
   admet model hpo -c config.yaml --no-logging

Batch Config Updates
--------------------

The ``scripts/add_logging_to_configs.py`` script was used to add logging sections to all 131 configuration files:

.. code-block:: bash

   # Show what would be updated (dry-run)
   python scripts/add_logging_to_configs.py --dry-run

   # Actually update configs
   python scripts/add_logging_to_configs.py

   # Update only specific directory
   python scripts/add_logging_to_configs.py --config-dir ./configs/1-hpo-single

Programmatic Usage
------------------

Use RayLogManager directly in custom scripts:

.. code-block:: python

   from pathlib import Path
   from ray import tune
   from admet.util.ray_logging import RayLogManager, QuietProgressReporter

   output_dir = Path("/tmp/ray_results")

   with RayLogManager(
       mlflow_run_id="my_experiment_001",
       output_dir=output_dir,
       verbose=1,
       max_total_logs_gb=2.0,
   ):
       tuner = tune.Tuner(
           "MyTrainable",
           param_space=search_space,
           run_config=tune.RunConfig(
               progress_reporter=QuietProgressReporter(),
           ),
       )
       results = tuner.fit()

Log Inspection
--------------

Access compressed logs from MLflow:

1. Open MLflow UI: ``mlflow ui``
2. Navigate to your run
3. Click "Artifacts" tab
4. Download ``ray_trial_logs_TIMESTAMP.tar.gz``
5. Extract and review logs:

   .. code-block:: bash

      tar -xzf ray_trial_logs_TIMESTAMP.tar.gz
      less trial_0/logs/train.log

Testing
=======

Run the comprehensive test suite:

.. code-block:: bash

   # All ray_logging tests
   pytest tests/test_ray_logging.py -v

   # Specific test class
   pytest tests/test_ray_logging.py::TestRayLogManager -v

   # Only unit tests (skip slow integration tests)
   pytest tests/test_ray_logging.py -v -m "not slow"

   # Benchmark tests
   pytest tests/test_ray_logging.py::TestLoggingPerformance -v --benchmark-only

Test Coverage
~~~~~~~~~~~~~

The test suite includes:

- **Unit Tests** (25+): RayLogManager, compression, upload, signal handling
- **Integration Tests** (5+): HPO with logging, Ensemble with logging
- **Performance Benchmarks** (3): Compression speed, log collection, memory overhead
- **Edge Cases** (5+): Empty logs, corrupted files, large files, size limits

Related Documentation
=====================

- :ref:`configuration`: Complete configuration reference
- :ref:`cli`: Command-line interface documentation
- :ref:`hpo`: Hyperparameter optimization guide
- :ref:`ensemble`: Ensemble training guide

See Also
========

- `Ray Tune Documentation <https://docs.ray.io/en/latest/tune/>`_
- `MLflow Artifacts <https://www.mlflow.org/docs/latest/artifacts.html>`_
- `Python logging module <https://docs.python.org/3/library/logging.html>`_

Glossary
========

.. glossary::

   MLflow Run
      A single experiment execution tracked by MLflow, with associated metrics, parameters, and artifacts

   MLflow Artifact
      A file (log, model checkpoint, plot) associated with an MLflow run

   Ray Trial
      A single training task in a Ray Tune hyperparameter optimization or ensemble experiment

   Compression
      Reduction of log file size using gzip compression (tar.gz format)

   Verbosity Level
      Control over how much logging output is displayed (0=quiet, 1=normal, 2=verbose)

Changelog
=========

**Version 1.0** (2025-01-04):

- Initial release of Ray Tune logging infrastructure
- Support for HPO and Ensemble training
- Automatic log collection, compression, and MLflow upload
- Progress tracking with ETA calculation
- Signal handling for graceful shutdown
- Comprehensive test suite
- Batch configuration update script

License
=======

This logging infrastructure is part of the OpenADMET Challenge project and follows the same license terms.

.. toctree::
   :hidden:

   ../index
