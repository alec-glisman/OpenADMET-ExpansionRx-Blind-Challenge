Visualization Tools (``admet.plot``)
====================================

The ``admet.plot`` package provides plotting utilities for ADMET model
evaluation and dataset exploration. It configures consistent styling using
``scienceplots`` (when available) or Seaborn defaults.

.. automodule:: admet.plot
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Global Configuration
--------------------

The package automatically configures Matplotlib with:

- ``GLASBEY_PALETTE``: A colorblind-friendly categorical color palette
- Scientific plotting style (via ``scienceplots`` if available)
- Agg backend for headless environments

Modules
-------

density
^^^^^^^

Distribution and density plots for dataset exploration.

.. automodule:: admet.plot.density
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

- ``plot_endpoint_distributions()``: Plot histograms for all target columns
- ``plot_property_distributions()``: Plot property distributions with KDE

heatmap
^^^^^^^

Correlation heatmaps and matrix visualizations.

.. automodule:: admet.plot.heatmap
   :members:
   :undoc-members:
   :show-inheritance:

latex
^^^^^

LaTeX-safe string formatting for publication-quality figures.

.. automodule:: admet.plot.latex
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

- ``latex_sanitize()``: Escape special LaTeX characters
- ``text_correlation()``: Format correlation statistics for LaTeX
- ``text_distribution()``: Format distribution statistics for LaTeX

metrics
^^^^^^^

Metric computation and bar chart visualizations.

.. automodule:: admet.plot.metrics
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

- ``compute_metrics_df()``: Compute regression metrics (RMSE, MAE, R²)
- ``compute_metrics_by_split()``: Compute metrics grouped by data split
- ``plot_metric_bar()``: Bar chart of metric values
- ``plot_all_metrics()``: Multi-panel metric visualization
- ``metrics_to_latex_table()``: Export metrics as LaTeX table

parity
^^^^^^

Parity plots for regression model evaluation.

.. automodule:: admet.plot.parity
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

- ``plot_parity()``: Single parity plot (predicted vs actual)
- ``plot_parity_by_split()``: Parity plots faceted by data split
- ``plot_parity_grid()``: Grid of parity plots for multi-task models
- ``save_parity_plots()``: Save parity plots to disk

split
^^^^^

Dataset split visualization and diagnostics.

.. automodule:: admet.plot.split
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

- ``plot_cluster_size_histogram()``: Histogram of cluster sizes
- ``plot_endpoint_finite_value_counts()``: Bar chart of non-NaN counts per endpoint
- ``plot_train_val_dataset_sizes()``: Compare train/validation set sizes
- ``plot_train_cluster_size_boxplots()``: Box plots of cluster sizes per fold

Example Usage
-------------

.. code-block:: python

   import pandas as pd
   from admet.plot import (
       plot_parity,
       plot_metric_bar,
       compute_metrics_df,
       GLASBEY_PALETTE,
   )

   # Load predictions
   df = pd.DataFrame({
       "y_true": [1.0, 2.0, 3.0, 4.0],
       "y_pred": [1.1, 1.9, 3.2, 3.8],
   })

   # Plot parity
   fig, ax = plot_parity(df["y_true"], df["y_pred"], title="LogD Predictions")
   fig.savefig("parity_logd.png", dpi=300)

   # Compute and plot metrics
   metrics = compute_metrics_df(df["y_true"], df["y_pred"])
   fig, ax = plot_metric_bar(metrics, title="Model Performance")
   fig.savefig("metrics.png", dpi=300)
