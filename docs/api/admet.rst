ADMET Package API
=================

Top-level API index for the ``admet`` package. This package provides tools for
training and evaluating machine learning models for ADMET (Absorption,
Distribution, Metabolism, Excretion, Toxicity) property prediction.

The package is organized into the following subpackages:

- :doc:`admet.data` - Data loading, chemistry utilities, and dataset splitting
- :doc:`admet.features` - Molecular fingerprints and descriptor generation
- :doc:`admet.model` - Model implementations (Chemprop, Chemeleon, Classical ML)
- :doc:`admet.plot` - Visualization utilities for plots and figures
- :doc:`admet.util` - Utility functions, profiling, and logging configuration

.. toctree::
   :maxdepth: 1

   admet.data
   admet.features
   admet.model
   admet.plot
   admet.util

Package Version
---------------

.. code-block:: python

   import admet
   print(admet.__version__)  # "1.3.0"
