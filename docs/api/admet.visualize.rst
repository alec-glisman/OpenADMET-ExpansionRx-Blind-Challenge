Visualization Tools (`admet.visualize`)
=======================================

The ``admet.visualize`` package provides plotting utilities for
multi-endpoint regression models. It focuses on:

* Parity plots and metric bar charts for ``train``/``validation``/``test``
  splits via :func:`admet.visualize.model_performance.plot_parity_grid` and
  :func:`admet.visualize.model_performance.plot_metric_bars`.
* High-level helpers to configure Matplotlib/Seaborn styles and color
  palettes via the :mod:`admet.visualize` top-level package, including the
  ``GLASBEY_PALETTE`` constant.
* Ensemble-evaluation specific visualizations in
  :mod:`admet.visualize.ensemble_eval`, which operate directly on the
  DataFrames produced by :mod:`admet.evaluate.ensemble`.

Core submodules
---------------

.. automodule:: admet.visualize.model_performance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: admet.visualize.ensemble_eval
   :members:
   :undoc-members:
   :show-inheritance:

Top-level configuration
-----------------------

.. automodule:: admet.visualize
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
