============
Installation
============

Get OpenADMET Challenge running on your system in under 5 minutes.

.. mermaid::

   flowchart LR
      A[Clone Repo] --> B[Create venv]
      B --> C[Install Package]
      C --> D[Setup Hooks]
      D --> E[✅ Ready!]

      style A fill:#e1f5fe
      style E fill:#c8e6c9

Requirements
============

- **Python**: 3.11 (required)
- **Package Manager**: `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip
- **GPU**: Optional but recommended for neural network training
- **RAM**: 16GB+ recommended for ensemble training

Quick Install
=============

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         # Clone repository
         git clone https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge.git
         cd OpenADMET-ExpansionRx-Blind-Challenge

         # Create virtual environment and install
         uv venv && source .venv/bin/activate
         uv pip install -e ".[dev,docs]"

         # Setup pre-commit hooks
         uv run pre-commit install
         uv run pre-commit install --hook-type commit-msg

   .. tab-item:: pip

      .. code-block:: bash

         # Clone repository
         git clone https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge.git
         cd OpenADMET-ExpansionRx-Blind-Challenge

         # Create virtual environment
         python3.11 -m venv .venv
         source .venv/bin/activate

         # Install package
         pip install -e ".[dev,docs]"

         # Setup pre-commit hooks
         pre-commit install
         pre-commit install --hook-type commit-msg

   .. tab-item:: conda

      .. code-block:: bash

         # Clone repository
         git clone https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge.git
         cd OpenADMET-ExpansionRx-Blind-Challenge

         # Create conda environment with Python 3.11
         conda create -n admet python=3.11 -y
         conda activate admet

         # Install package
         pip install -e ".[dev,docs]"

Verify Installation
===================

Check that the CLI is available:

.. code-block:: bash

   admet --help

Expected output:

.. code-block:: text

   Usage: admet [OPTIONS] COMMAND [ARGS]...

   OpenADMET Challenge CLI - ADMET prediction with graph neural networks.

   Options:
     --version  Show version and exit.
     --help     Show this message and exit.

   Commands:
     data         Data processing commands.
     leaderboard  Leaderboard commands.
     model        Model training commands.

GPU Support
===========

For GPU acceleration, ensure PyTorch is installed with CUDA support:

.. code-block:: bash

   # Check CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

If CUDA is not detected, reinstall PyTorch with CUDA:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

MLflow Setup
============

Start the MLflow tracking server:

.. code-block:: bash

   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Access the UI at http://localhost:5000.

Optional Dependencies
=====================

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   make -C docs html
   # View at docs/_build/html/index.html

Development Tools
-----------------

Run linting and type checking:

.. code-block:: bash

   # Format code
   black src/ tests/ && isort src/ tests/

   # Type check
   mypy src/admet/

   # Lint
   pylint src/admet/ --fail-under=9.0

Troubleshooting
===============

Common installation issues and solutions:

**ImportError: bitbirch.pruning**
   This module requires compilation. If unavailable, clustering features are disabled.
   The core training functionality works without it.

**CUDA out of memory**
   Reduce batch size in configuration or use ``gpus_per_trial: 0.5`` to share GPUs.

**Ray initialization fails**
   Set ``RAY_ADDRESS=local`` or ensure no stale Ray processes are running:

   .. code-block:: bash

      ray stop --force

**Permission denied on pre-commit**
   Ensure hooks are executable:

   .. code-block:: bash

      chmod +x .git/hooks/*

Next Steps
==========

- :doc:`quickstart` — Train your first model in 5 minutes
- :doc:`/guide/configuration` — Understand the YAML configuration system
- :doc:`/guide/architecture` — Learn the system architecture
