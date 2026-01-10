Profiling Guide
===============

The profiling system identifies performance bottlenecks in ensemble training with
minimal overhead. Three profiling modes provide progressively detailed insights:
phase-level (~1%), function-level (~5-10%), and full profiling (~15-25% overhead).

Overview
--------

The profiling system provides three levels of detail to help you optimize training performance:

1. **Phase-level profiling** (default): Tracks major phases like training, prediction, plot generation, and artifact logging (~1% overhead)
2. **Function-level profiling**: Uses cProfile to track individual function calls within your code (~5-10% overhead)
3. **Full profiling**: Extended function tracking with detailed statistics (~15-25% overhead)

Quick Start
-----------

1. Enable Profiling (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiling is enabled by default in phase mode. Your config already includes:

.. code-block:: yaml

    profiling:
      enabled: true
      mode: phase  # Minimal overhead, tracks major phases
      print_summary: true
      log_to_mlflow: true

2. Run Ensemble Training
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    admet model ensemble -c configs/3-hpo-production/ensemble_chemprop_hpo_001.yaml --max-parallel 5

3. Review Profiling Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training completes, you'll see a detailed profiling summary:

.. code-block:: text

    ================================================================================
     PROFILING SUMMARY: ensemble_chemprop
    ================================================================================
     Total Duration: 15m 32.4s
    ================================================================================
    Phase                                    Count      Total         Mean       %
    --------------------------------------------------------------------------------
    ensemble_model_train                         1    14m 28.1s    14m 28.1s   93.1%
    plot_generation                              1      1m 2.3s      1m 2.3s    6.7%
    artifact_logging                             1        2.1s         2.1s    0.2%
    ================================================================================

    ================================================================================
     PER-MODEL TRAINING BREAKDOWN
    ================================================================================
    Model                     Total      Training      Predict      Metrics        Plots    Artifacts
    ------------------------------------------------------------------------------------------------------------------------
    split_0_fold_0          3m 12.5s      2m 45.2s       15.3s        3.2s        7.8s         1.0s
    split_0_fold_1          3m 8.1s       2m 42.1s       14.9s        3.1s        7.2s         0.8s
    ...
    ------------------------------------------------------------------------------------------------------------------------
    Min                     3m 8.1s
    Max                     3m 12.5s
    Mean                    3m 10.2s
    Sum (serial)           15m 51.0s
    Parallel speedup          1.10x (22.0% efficiency)
    ================================================================================

    ================================================================================
     BOTTLENECK ANALYSIS (Aggregated across all models)
    ================================================================================
    Phase                            Total Time      Per Model    % of Total         Optimization Potential
    ------------------------------------------------------------------------------------------------------------------------
    Training (PyTorch)                 13m 46.5s       2m 45.3s        86.9%                              -
    Plot Generation                      36.0s           7.2s         3.8%     Set post_training.generate_plots=false
    Prediction                           1m 15.0s       15.0s         7.9%     Ensure cache_predictions=true
    Metrics Computation                  15.5s           3.1s         1.6%     Disable compute_train_metrics
    Artifact Logging                      4.0s           0.8s         0.4%                              -
    ================================================================================

Profiling Modes
---------------

Phase Mode (Recommended for Production)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overhead**: ~1%

**Use case**: Default profiling for all training runs

.. code-block:: yaml

    profiling:
      enabled: true
      mode: phase

**Tracks**:

- Data loading and preprocessing
- Training epochs
- Validation
- Prediction
- Metrics computation
- Plot generation
- Artifact logging to MLflow

Function Mode (Debugging Bottlenecks)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overhead**: ~5-10%

**Use case**: Identifying slow functions within phases

.. code-block:: yaml

    profiling:
      enabled: true
      mode: function
      function_top_n: 100  # Track top 100 slowest functions

**Additional output**:

.. code-block:: text

    ================================================================================
     AGGREGATED FUNCTION HOTSPOTS (across all models)
    ================================================================================
    Function                                                    Cum(s)       Calls   Models
    ------------------------------------------------------------------------------------------------------------------------
    admet.model.chemprop.model.ChempropModel.fit               825.3        5        5
    admet.plot.parity.plot_parity                              36.2       45        5
    matplotlib.pyplot.savefig                                  28.1       45        5
    admet.model.chemprop.model._generate_evaluation_plots      24.5       10        5
    mlflow.log_artifact                                        12.3       50        5
    ...
    ================================================================================

Full Mode (Deep Performance Analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overhead**: ~15-25%

**Use case**: Comprehensive profiling for optimization work

.. code-block:: yaml

    profiling:
      enabled: true
      mode: full
      function_top_n: 200

Includes extended function tracking and per-batch timing.

Disabled Mode (Maximum Speed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Overhead**: 0%

**Use case**: Final production runs where every second counts

.. code-block:: yaml

    profiling:
      enabled: false
      # or
      mode: disabled

Optimizing Based on Profiling Results
--------------------------------------

1. Plot Generation is Slow (>5% of total time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Generating parity plots for each model is taking too long.

**Solutions**:

.. code-block:: yaml

    # Option A: Disable plots entirely (fastest)
    post_training:
      generate_plots: false

    # Option B: Lower plot quality (faster rendering)
    post_training:
      generate_plots: true
      plot_dpi: 100  # vs default 150

    # Option C: Only generate ensemble-level plots (disable per-model plots)
    # This requires code changes to skip plots in individual model training

**Expected speedup**: 3-10% reduction in total training time.

2. MLflow Artifact Logging is Slow (>3% of total time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Uploading artifacts to MLflow is blocking model training.

**Solutions**:

.. code-block:: yaml

    post_training:
      async_artifact_upload: true  # Experimental: upload in background
      log_model_to_mlflow: false   # Skip PyTorch model logging (saves time)

**Expected speedup**: 2-5% reduction in total training time.

3. Metrics Computation is Slow (>30% of total time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Computing correlation metrics (especially Spearman and Kendall rank correlations) is taking too long.

**Symptoms**:

.. code-block:: text

    Phase                                     Count        Total         Mean        %
    --------------------------------------------------------------------------------
    metrics_computation                           1     8m 47.3s     8m 47.3s    93.6%
    training_total                                1       20.49s       20.49s     3.6%

**Solutions**:

.. code-block:: yaml

    # Option A: Disable expensive rank correlations (30-50% speedup)
    post_training:
      compute_rank_correlations: false  # Disables Spearman and Kendall tau
      # You'll still get: MAE, RMSE, R2, Pearson correlation

    # Option B: Disable per-quality metrics if using curriculum learning
    post_training:
      compute_per_quality_metrics: false  # Skip quality-level breakdown
      # Metrics still computed for overall validation/test

    # Option C: Disable train set metrics
    post_training:
      compute_train_metrics: false  # Skip training set metrics entirely

    # Option D: Combine all optimizations for maximum speed
    post_training:
      compute_rank_correlations: false
      compute_per_quality_metrics: false
      compute_train_metrics: false
      cache_predictions: true  # Essential for avoiding redundant predictions

**Expected speedup**:

- **Automatic (v2.0+)**: Vectorized batch computation provides 10-30x speedup automatically
- Disabling rank correlations: Additional 30-50% reduction in metrics computation time
- Disabling per-quality metrics: Additional 40-70% reduction (if many quality levels)
- Disabling train metrics: Additional 30-50% reduction
- GPU acceleration: Additional 2-5x speedup for large datasets (requires CuPy)
- **Combined: Can reduce 8+ minutes to under 10 seconds** (~50-100x total speedup)

**Advanced GPU Acceleration** (Optional):

For datasets with >10k samples, enable GPU-accelerated metrics:

.. code-block:: yaml

    post_training:
      use_gpu_metrics: true  # Requires: pip install cupy-cuda11x or cupy-cuda12x
      compute_rank_correlations: false  # Recommended with GPU for maximum speed

.. note::

    GPU acceleration automatically falls back to CPU if CuPy is unavailable or errors occur.

**What you keep**:

Even with all optimizations enabled, you still get:

- MAE, RAE, MAPE, RMSE, R², Pearson correlation for validation/test sets
- Parity plots (if enabled)
- All training metrics from PyTorch Lightning

4. Prediction is Slow (>5% of total time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Multiple redundant ``predict()`` calls on the same data.

**Solutions**:

.. code-block:: yaml

    post_training:
      cache_predictions: true  # Should already be enabled (default)
      compute_train_metrics: false  # Disable train set predictions

**Expected speedup**: 5-15% reduction in total training time.

5. Low Parallel Efficiency (<50%)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Models are waiting for resources instead of running in parallel.

**Diagnosis**:

.. code-block:: text

    Parallel speedup: 1.10x (22.0% efficiency)

This means you're only getting 22% of the ideal 5x speedup with 5 parallel workers.

**Solutions**:

.. code-block:: yaml

    # Option A: Reduce parallelism to match available resources
    ray:
      max_parallel: 2  # vs 5

    # Option B: Add more GPUs
    ray:
      max_parallel: 5
      num_gpus: 1.0  # Full GPU per model
      gpu_ids: [0, 1, 2, 3, 4]  # 5 GPUs

    # Option C: Reduce per-model GPU allocation
    ray:
      max_parallel: 5
      num_gpus: 0.2  # 5 models share 1 GPU

**Expected improvement**: Efficiency >80% indicates good parallelization.

Advanced Usage
--------------

Profiling a Specific Training Run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable function-level profiling for a single diagnostic run:

.. code-block:: bash

    # Edit your config to enable function mode
    vim configs/3-hpo-production/ensemble_chemprop_hpo_001.yaml

    # Change:
    # profiling:
    #   mode: function

    # Run with fewer models for faster profiling
    admet model ensemble -c configs/3-hpo-production/ensemble_chemprop_hpo_001.yaml

Accessing Profiling Metrics in MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiling metrics are logged to MLflow at two levels:

**1. Individual Model Runs (Nested Runs)**

Each split/fold model logs its own profiling metrics under ``profiling.*``:

- ``profiling.total_seconds``: Total time for this specific model
- ``profiling.training_total.total_seconds``: PyTorch training time
- ``profiling.ensemble_prediction.total_seconds``: Prediction time
- ``profiling.plot_generation.total_seconds``: Plot generation time
- ``profiling.artifact_logging.total_seconds``: Artifact upload time
- ``profiling.metrics_computation.total_seconds``: Metrics computation time

**2. Parent Run (Aggregated Statistics)**

The parent run contains aggregated statistics across all models under ``profiling.ensemble.*``:

- ``profiling.ensemble.n_models``: Number of models trained
- ``profiling.ensemble.mean_seconds``: Average time per model
- ``profiling.ensemble.sum_seconds``: Total serial time
- ``profiling.ensemble.parallel_speedup``: Actual speedup achieved
- ``profiling.ensemble.parallel_efficiency_pct``: Parallelization efficiency
- ``profiling.ensemble.training_total_seconds``: Aggregated training time
- ``profiling.ensemble.training_pct``: Percentage of time spent in training
- ``profiling.ensemble.plots_total_seconds``: Aggregated plot generation time
- ``profiling.ensemble.plots_pct``: Percentage of time spent generating plots
- ``profiling.ensemble.artifacts_total_seconds``: Aggregated artifact logging time
- ``profiling.ensemble.artifacts_pct``: Percentage of time spent logging artifacts

You can query these metrics to track performance across experiments:

.. code-block:: python

    import mlflow

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["1"])

    for run in runs:
        metrics = run.data.metrics

        # Check if this is a parent run (has ensemble metrics)
        if "profiling.ensemble.n_models" in metrics:
            print(f"\nEnsemble Run: {run.info.run_name}")
            print(f"  Models trained: {metrics.get('profiling.ensemble.n_models', 0):.0f}")
            print(f"  Parallel speedup: {metrics.get('profiling.ensemble.parallel_speedup', 0):.2f}x")
            print(f"  Efficiency: {metrics.get('profiling.ensemble.parallel_efficiency_pct', 0):.1f}%")
            print(f"  Plot generation: {metrics.get('profiling.ensemble.plots_pct', 0):.1f}%")
            print(f"  Training time: {metrics.get('profiling.ensemble.training_pct', 0):.1f}%")

Disabling Profiling Summary Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're running many experiments and don't want console output:

.. code-block:: yaml

    profiling:
      enabled: true
      mode: phase
      print_summary: false  # Disable console output
      log_to_mlflow: true   # Still log metrics to MLflow

Profiling Configuration Reference
----------------------------------

.. code-block:: yaml

    profiling:
      # Whether to enable profiling at all
      enabled: true

      # Profiling detail level: "disabled", "phase", "function", "full"
      # - disabled: No profiling (0% overhead)
      # - phase: Phase-level timing only (~1% overhead)
      # - function: cProfile function tracking (~5-10% overhead)
      # - full: Extended function tracking (~15-25% overhead)
      mode: phase

      # Number of top functions to track (function/full mode only)
      function_top_n: 50

      # Only track functions from modules containing this string
      function_filter_module: admet

      # Whether to log profiling metrics to MLflow
      log_to_mlflow: true

      # Whether to print profiling summary to console
      print_summary: true

      # Whether to aggregate profiling data across ensemble models
      ensemble_aggregation: true

Common Issues
-------------

Issue: Profiling overhead is too high
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution**: Switch to phase mode or disable profiling:

.. code-block:: yaml

    profiling:
      mode: phase  # or "disabled"

Issue: No profiling output shown
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Check**:

1. Is ``profiling.enabled: true``?
2. Is ``profiling.print_summary: true``?
3. Did training complete successfully?

Issue: MLflow profiling metrics not logged
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Check**:

1. Is ``profiling.log_to_mlflow: true``?
2. Is ``mlflow.tracking: true``?
3. Check MLflow UI under the run's metrics tab

Best Practices
--------------

1. **Always use phase mode in production** - Minimal overhead, useful insights
2. **Use function mode for optimization** - Temporarily enable when investigating performance
3. **Review profiling after every major config change** - Ensure your changes improved performance
4. **Compare profiling across experiments** - Use MLflow metrics to track performance trends
5. **Profile with representative data** - Use realistic dataset sizes for accurate bottleneck identification

Example Optimization Workflow
------------------------------

1. **Baseline**: Run with ``mode: phase`` to establish baseline timing
2. **Identify**: Look at bottleneck analysis for phases >5% of total time
3. **Investigate**: If needed, run with ``mode: function`` to dig deeper
4. **Optimize**: Apply suggested optimizations from bottleneck analysis
5. **Verify**: Run again with ``mode: phase`` to measure improvement
6. **Production**: Keep ``mode: phase`` enabled for ongoing monitoring

Error Handling and Robustness
------------------------------

The profiling system is designed to always collect data, even when things go wrong:

Keyboard Interrupts (Ctrl+C)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you interrupt ensemble training, profiling data will still be:

- ✅ Displayed in the terminal summary
- ✅ Logged to MLflow for all completed models
- ✅ Aggregated across whatever models finished

.. code-block:: bash

    # Even if you Ctrl+C during training...
    ^C Ensemble training interrupted by user. Saving profiling info...

    ================================================================================
    PROFILING SUMMARY (collected even on error/interrupt)
    ================================================================================
    # Full profiling output still appears

Training Failures
^^^^^^^^^^^^^^^^^

If individual models fail during training:

- ✅ Profiling data is still collected for that model
- ✅ Partial timing information is logged to MLflow
- ✅ Aggregated statistics include all models (failed + successful)
- ✅ Terminal summary shows which models failed

MLflow Logging Errors
^^^^^^^^^^^^^^^^^^^^^^

If MLflow logging fails:

- ✅ Terminal output is still generated
- ✅ Warning logged but training continues
- ✅ Profiling data preserved in memory

Summary
-------

The profiling system gives you visibility into where ensemble training time is spent, helping you:

- **Identify bottlenecks**: See which phases are slowest
- **Optimize selectively**: Focus on high-impact improvements
- **Track performance**: Monitor training speed across experiments
- **Debug issues**: Drill down to function-level when needed
- **Always available**: Profiling data collected even on errors/interrupts

Key Features
^^^^^^^^^^^^

✅ **Per-model profiling**: Each split/fold logs detailed timing to its own MLflow run

✅ **Aggregated statistics**: Parent run contains ensemble-wide metrics

✅ **Robust to failures**: Profiling data always collected, even on errors

✅ **Terminal output**: Summary always printed, even if MLflow fails

✅ **Automatic bottleneck detection**: Suggestions for optimization

For most users, the default ``mode: phase`` provides the best balance of insight and performance.
