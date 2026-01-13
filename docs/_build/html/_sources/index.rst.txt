.. raw:: html

   <style>
   .hero-section { text-align: center; padding: 2rem 0 3rem; }
   .hero-section h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
   .hero-tagline { font-size: 1.25rem; color: #666; margin-bottom: 2rem; max-width: 700px; margin-left: auto; margin-right: auto; }
   .stat-number { font-size: 2rem; font-weight: 700; color: #007A73; }
   .stat-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
   .section-header { text-align: center; margin: 3rem 0 1.5rem; }
   .section-header h2 { font-size: 1.75rem; margin-bottom: 0.25rem; }
   .section-subtitle { color: #666; font-size: 1rem; }
   </style>

.. image:: _static/images/logo.svg
   :alt: OpenADMET Challenge
   :align: center
   :width: 280px

.. raw:: html

   <div class="hero-section">
   <h1>OpenADMET Challenge</h1>
   <p class="hero-tagline">
   Production-ready ADMET prediction with graph neural networks,
   hyperparameter optimization, and ensemble training.
   </p>
   </div>

.. grid:: 3
   :gutter: 4
   :class-container: sd-text-center

   .. grid-item::

      .. raw:: html

         <div class="stat-number">9</div>
         <div class="stat-label">ADMET Endpoints</div>

   .. grid-item::

      .. raw:: html

         <div class="stat-number">25</div>
         <div class="stat-label">Ensemble Models</div>

   .. grid-item::

      .. raw:: html

         <div class="stat-number">5×5</div>
         <div class="stat-label">Split × Fold CV</div>

----

.. raw:: html

   <div class="section-header">
   <h2>Competition Solution Highlights</h2>
   <p class="section-subtitle">Competitive approach in the ExpansionRx Blind Challenge</p>
   </div>

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🏆 Competition Results
      :class-card: sd-border-0 sd-shadow-sm

      Ranked **11th out of 321** participants (Top 3.4%) in the OpenADMET ExpansionRx
      Blind Challenge with MA-RAE of 0.57 ± 0.02 — within error bar of 2nd place
      (as of 2026-01-12).

   .. grid-item-card:: 🔬 Robust Methodology
      :class-card: sd-border-0 sd-shadow-sm

      5×5 cross-validation with :term:`BitBirch` cluster-aware splitting
      prevents data leakage and ensures robust performance estimates.

   .. grid-item-card:: 🧠 Advanced Techniques
      :class-card: sd-border-0 sd-shadow-sm

      :term:`Chemprop` :term:`MPNN` with 25-model :term:`ensemble`
      and task-weighted loss optimization for multi-task ADMET prediction.

----

.. raw:: html

   <div class="section-header">
   <h2>Quick Commands</h2>
   <p class="section-subtitle">Get running in seconds</p>
   </div>

.. tab-set::

   .. tab-item:: Install

      .. code-block:: bash

         git clone https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge.git
         cd OpenADMET-ExpansionRx-Blind-Challenge
         uv venv && source .venv/bin/activate
         uv pip install -e ".[dev]"

   .. tab-item:: Train

      .. code-block:: bash

         admet model train -c configs/0-experiment/0-single-fold/chemprop.yaml

   .. tab-item:: HPO

      .. code-block:: bash

         admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50

   .. tab-item:: Ensemble

      .. code-block:: bash

         admet model ensemble -c configs/0-experiment/1-ensemble/chemprop_ensemble.yaml

----

.. raw:: html

   <div class="section-header">
   <h2>End-to-End Pipeline</h2>
   <p class="section-subtitle">From raw molecules to robust predictions</p>
   </div>

.. mermaid::

   flowchart LR
      subgraph "📥 Data"
         A[SMILES] --> B[BitBirch<br/>Clustering]
         B --> C[5×5 Splits]
      end

      subgraph "🧠 Training"
         D[Chemprop<br/>MPNN] --> E[HPO<br/>Ray Tune]
         E --> F[Top K<br/>Configs]
      end

      subgraph "🎯 Production"
         G[25-Model<br/>Ensemble] --> H[Predictions<br/>+ Uncertainty]
      end

      C --> D
      F --> G

      style A fill:#e1f5fe
      style E fill:#fff9c4
      style H fill:#c8e6c9

----

.. raw:: html

   <div class="section-header">
   <h2>Get Started</h2>
   <p class="section-subtitle">From first model to production ensemble in minutes</p>
   </div>

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🚀 Quickstart
      :link: getting-started/quickstart
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Train your first Chemprop model in 5 minutes with a complete end-to-end example.

   .. grid-item-card:: ⚙️ Configuration
      :link: guide/configuration
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Master the YAML configuration system for models, training, and experiments.

   .. grid-item-card:: 📊 MLflow Tracking
      :link: guide/mlflow_artifacts
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Track experiments, compare runs, and analyze results with MLflow integration.

----

.. raw:: html

   <div class="section-header">
   <h2>Core Workflows</h2>
   <p class="section-subtitle">Everything you need for ADMET modeling</p>
   </div>

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Data Pipeline
      :class-card: sd-border-0 sd-shadow-sm

      .. grid:: 1
         :gutter: 1

         .. grid-item::

            :doc:`guide/data_sources` — Load and preprocess molecular data

         .. grid-item::

            :doc:`guide/splitting` — Cluster-aware train/val/test splits

         .. grid-item::

            :doc:`guide/debugging_per_quality_metrics` — Debug quality-stratified metrics

   .. grid-item-card:: Model Training
      :class-card: sd-border-0 sd-shadow-sm

      .. grid:: 1
         :gutter: 1

         .. grid-item::

            :doc:`guide/modeling` — Chemprop MPNN and Chemeleon training

         .. grid-item::

            :doc:`guide/classical_models` — XGBoost, LightGBM, CatBoost baselines

         .. grid-item::

            :doc:`guide/hpo` — Ray Tune hyperparameter optimization

.. grid:: 2
   :gutter: 3
   :margin: 3 0 0 0

   .. grid-item-card:: Advanced Techniques
      :class-card: sd-border-0 sd-shadow-sm

      .. grid:: 1
         :gutter: 1

         .. grid-item::

            :doc:`guide/curriculum` — Quality-aware curriculum learning *(experimental, abandoned)*

         .. grid-item::

            :doc:`guide/task_affinity` — Multi-task gradient optimization

         .. grid-item::

            :doc:`guide/endpoints` — ADMET endpoint reference

   .. grid-item-card:: Operations
      :class-card: sd-border-0 sd-shadow-sm

      .. grid:: 1
         :gutter: 1

         .. grid-item::

            :doc:`guide/profiling` — Performance profiling and optimization

         .. grid-item::

            :doc:`guide/logging` — Ray Tune logging and debugging

         .. grid-item::

            :doc:`reference/scripts` — Shell scripts for automation

----

.. raw:: html

   <div class="section-header">
   <h2>Reference</h2>
   <p class="section-subtitle">API documentation and configuration specs</p>
   </div>

.. grid:: 4
   :gutter: 3

   .. grid-item-card:: 🐍 Python API
      :link: api/admet
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

   .. grid-item-card:: 💻 CLI Reference
      :link: guide/cli
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

   .. grid-item-card:: 📋 Config Spec
      :link: guide/config_reference
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

   .. grid-item-card:: 🏆 Leaderboard
      :link: guide/leaderboard
      :link-type: doc
      :text-align: center
      :class-card: sd-border-0 sd-shadow-sm

----

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting-started/index
   getting-started/installation
   getting-started/quickstart
   guide/architecture

.. toctree::
   :maxdepth: 1
   :caption: Data Pipeline

   guide/endpoints
   guide/data_sources
   guide/splitting
   guide/debugging_per_quality_metrics

.. toctree::
   :maxdepth: 1
   :caption: Model Training

   guide/modeling
   guide/classical_models
   guide/hpo

.. toctree::
   :maxdepth: 1
   :caption: Advanced Training

   guide/curriculum
   guide/task_affinity

.. toctree::
   :maxdepth: 1
   :caption: Operations

   guide/mlflow_artifacts
   guide/profiling
   guide/logging
   guide/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Reference

   guide/cli
   guide/configuration
   guide/config_reference
   reference/scripts
   glossary
   api/admet
   api/leaderboard

.. toctree::
   :maxdepth: 1
   :caption: Community

   guide/leaderboard
   guide/development

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   dev/planning

----

.. grid:: 2
   :gutter: 4

   .. grid-item::

      **Build Docs Locally**

      .. tab-set::

         .. tab-item:: Make

            .. code-block:: bash

               make -C docs html

         .. tab-item:: Auto-reload

            .. code-block:: bash

               sphinx-autobuild docs docs/_build/html

   .. grid-item::

      **Contributing**

      We welcome contributions! See :doc:`guide/development` for setup
      instructions, coding standards, and how to submit pull requests.

      Found a bug? `Open an issue <https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge/issues>`_

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
