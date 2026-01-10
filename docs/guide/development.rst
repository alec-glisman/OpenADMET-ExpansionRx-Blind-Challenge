Development Guide
=================

This guide will help you set up your development environment and contribute to the OpenADMET project.

Environment Setup
-----------------

To get started, you'll need a Python 3.11 environment. We recommend using a virtual environment to keep your dependencies isolated.

1. Create and activate a virtual environment:

   .. code-block:: bash

      uv venv
      source .venv/bin/activate

2. Install the project in editable mode with development and documentation dependencies:

   .. code-block:: bash

      uv pip install -e ".[dev,docs]"

Run Tests
---------

We use `pytest` for testing. You can run the full test suite with:

.. code-block:: bash

   pytest -q

Code Style
----------

We follow strict code style guidelines to ensure consistency:

* **Line Length**: 109 characters (Black default).
* **Imports**: Managed by `isort` (profile "black").
* **Docstrings**: NumPy style with RST module headers.

To automatically format your code:

.. code-block:: bash

   black . && isort .

Type Checking
-------------

We use `mypy` for static type checking. If you have `mypy` installed (included in dev extras), you can run:

.. code-block:: bash

   mypy src/admet

Docs Build
----------

To build the documentation locally, ensure you have installed the documentation dependencies:

.. code-block:: bash

   uv pip install -e ".[docs]"

Then build the HTML documentation:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

You can view the generated documentation by opening `docs/_build/html/index.html` in your web browser.

Performance Profiling
---------------------

When debugging slow training or optimizing ensemble workflows, use the built-in profiling system:

.. code-block:: yaml

   # Enable in your config YAML
   profiling:
     enabled: true
     mode: phase  # Options: phase, function, full
     print_summary: true
     log_to_mlflow: true

Profiling modes:

* **phase**: Tracks major phases (training, prediction, plots) with ~1% overhead
* **function**: Uses cProfile for function-level tracking (~5-10% overhead)
* **full**: Extended function tracking with detailed statistics (~15-25% overhead)

After training, you'll see a bottleneck analysis showing where time is spent:

.. code-block:: bash

   # Example output
   Phase                     Total Time    % of Total    Optimization Potential
   Training (PyTorch)        13m 46.5s     86.9%         -
   Plot Generation           36.0s         3.8%          Set generate_plots=false
   Prediction                1m 15.0s      7.9%          Ensure cache_predictions=true

For detailed usage and optimization strategies, see the :doc:`profiling` guide.

Continuous Improvement
----------------------

We welcome contributions! If you find missing documentation, unclear type hints, or have ideas for improving the data pipelines, training orchestration, or model protocols, please open an issue or submit a pull request.

See also
--------

* :doc:`architecture`
* :doc:`cli`
* :doc:`profiling`
