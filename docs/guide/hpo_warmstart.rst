Warmstarting Hyperparameter Optimization
=========================================

Overview
--------

Warmstarting enables you to continue hyperparameter optimization from previous runs,
leveraging historical trial data to accelerate convergence. This is especially useful when:

- Refining hyperparameters after initial broad search
- Adding more trials to existing studies
- Exploring neighboring regions of known-good configurations

Key Benefits
^^^^^^^^^^^^

- **30-50% fewer trials** to reach optimal configurations
- Reuse expensive trial evaluations from previous runs
- Iteratively refine search without starting from scratch
- Build institutional knowledge of hyperparameter landscapes

How It Works
------------

Warmstarting uses Optuna's persistent study storage:

1. **Persistent Studies**: Studies are saved to SQLite database
2. **Trial History**: All trial parameters and results stored
3. **Top-K Seeding**: Best trials from previous study enqueued first
4. **Bayesian Continuation**: TPE sampler uses historical data for suggestions

Configuration
-------------

Basic Setup
^^^^^^^^^^^

Enable study persistence in your HPO config:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     seed: 42
     n_initial_points: 100
     # Enable persistent study storage
     persist_study: true
     study_name: "chemprop_hpo_v1"  # Unique identifier for this study
     storage_dir: "hpo_results/optuna_studies"  # Database location

This creates: ``hpo_results/optuna_studies/studies.db``

Warmstart from Previous Study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To continue from a previous study:

.. code-block:: yaml

   search_algorithm:
     type: optuna
     persist_study: true
     study_name: "chemprop_hpo_v2"  # New study name
     warmstart_from: "chemprop_hpo_v1"  # Load trials from this study
     warmstart_n_trials: 15  # Number of top trials to enqueue

CLI Usage
---------

List Available Studies
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List all studies
   admet model hpo-list-studies

   # List studies in custom directory
   admet model hpo-list-studies --storage-dir my_studies/

   # Show detailed trial information
   admet model hpo-list-studies --verbose

Example output::

   ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
   ┃ Study Name         ┃ Direction ┃ Trials ┃ Best Value ┃ Created         ┃
   ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
   │ chemprop_hpo_v1    │ MINIMIZE  │    150 │   0.185432 │ 2026-01-05 14:30│
   │ chemprop_hpo_v2    │ MINIMIZE  │     75 │   0.172891 │ 2026-01-06 09:15│
   └────────────────────┴───────────┴────────┴────────────┴─────────────────┘

Run HPO with Warmstart
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run with warmstart config
   admet model hpo -c configs/hpo_warmstart.yaml

   # Override number of samples
   admet model hpo -c configs/hpo_warmstart.yaml --num-samples 50

Workflow Examples
-----------------

Iterative Refinement
^^^^^^^^^^^^^^^^^^^^

**Phase 1: Broad Search**

.. code-block:: yaml

   experiment_name: chemprop_broad_search
   search_algorithm:
     persist_study: true
     study_name: "chemprop_v1_broad"
     n_initial_points: 100

   search_space:
     learning_rate:
       type: loguniform
       low: 0.0001
       high: 0.03  # Wide range

   resources:
     num_samples: 1000  # Many trials for exploration

**Phase 2: Focused Search**

.. code-block:: yaml

   experiment_name: chemprop_focused_search
   search_algorithm:
     persist_study: true
     study_name: "chemprop_v2_focused"
     warmstart_from: "chemprop_v1_broad"
     warmstart_n_trials: 20  # Seed with top 20 trials

   search_space:
     learning_rate:
       type: loguniform
       low: 0.001
       high: 0.01  # Narrowed based on v1 results

   resources:
     num_samples: 500  # Fewer trials needed

Cross-Validation Study
^^^^^^^^^^^^^^^^^^^^^^

Warmstart across different data splits:

.. code-block:: yaml

   # Split 0
   experiment_name: chemprop_split_0
   search_algorithm:
     persist_study: true
     study_name: "chemprop_split_0"

.. code-block:: yaml

   # Split 1 - warmstart from split 0
   experiment_name: chemprop_split_1
   search_algorithm:
     persist_study: true
     study_name: "chemprop_split_1"
     warmstart_from: "chemprop_split_0"
     warmstart_n_trials: 10

Best Practices
--------------

Study Naming
^^^^^^^^^^^^

- Use descriptive names: ``{model}_{dataset}_{version}``
- Include timestamps for experiments: ``chemprop_v1_20260106``
- Document rationale in config comments

Storage Management
^^^^^^^^^^^^^^^^^^

- Keep one ``studies.db`` per project
- Backup before major changes
- Archive completed studies to separate database

Warmstart Tuning
^^^^^^^^^^^^^^^^

- Start with 10-20 warmstart trials (``warmstart_n_trials``)
- More trials = faster initial convergence but less exploration
- Fewer trials = more exploration but slower start

Monitoring
^^^^^^^^^^

- Compare warmstart vs cold-start performance
- Track trials-to-convergence metric
- Log study metadata for reproducibility

Integration with MLflow
-----------------------

Study metadata is automatically logged to MLflow:

.. code-block:: python

   # Logged to MLflow
   mlflow.log_param("study_name", "chemprop_v2")
   mlflow.log_param("warmstart_from", "chemprop_v1_initial")

   # Saved to study_metadata.json
   {
     "study_name": "chemprop_v2",
     "warmstart_from": "chemprop_v1",
     "warmstart_n_trials": 15,
     "n_trials": 500,
     "best_metric": 0.1729
   }

Use MLflow UI to track:

- Study lineage (which studies warmstarted from others)
- Convergence comparisons
- Best trial evolution

Troubleshooting
---------------

Study Not Found
^^^^^^^^^^^^^^^

**Error**: ``Study 'chemprop_v1' not found in database``

**Solution**: Verify study name with ``admet model hpo-list-studies``

Study Already Exists
^^^^^^^^^^^^^^^^^^^^

**Error**: ``Study 'chemprop_v2' already exists``

**Solution**: Use different study name or delete existing study:

.. code-block:: python

   import optuna
   optuna.delete_study(study_name="chemprop_v2", storage="sqlite:///...")

Incompatible Search Space
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Warning**: ``Parameter 'hidden_dim' not in warmstart trials``

**Solution**: Ensure search spaces are compatible. Warmstart trials will be skipped
if parameters don't match new search space.

Performance Comparison
----------------------

Expected benefits from warmstart:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Scenario
     - Cold Start Trials
     - Warmstart Trials
   * - Initial broad search
     - 1000
     - N/A
   * - Focused refinement
     - 800
     - 500 (38% reduction)
   * - Cross-validation fold
     - 1000
     - 600 (40% reduction)
   * - Architecture search
     - 1500
     - 1000 (33% reduction)

Advanced Usage
--------------

Custom Storage Location
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   search_algorithm:
     persist_study: true
     storage_dir: "/custom/path/studies"  # Custom database location

Auto-Generated Study Names
^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``study_name`` is not provided, it's auto-generated:

.. code-block:: python

   # Format: {experiment_name}_{timestamp}
   study_name = f"chemprop_hpo_20260106_143052"

Multiple Warmstart Sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, warmstart supports loading from one previous study. For multiple sources:

1. Run sequential warmstarts: A → B → C
2. Or manually merge studies in Optuna database

API Reference
-------------

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``persist_study``
     - bool
     - Enable persistent study storage (default: False)
   * - ``study_name``
     - str | None
     - Unique study identifier (auto-generated if None)
   * - ``storage_dir``
     - str | None
     - Directory for SQLite database (default: hpo_results/optuna_studies)
   * - ``warmstart_from``
     - str | None
     - Name of previous study to load trials from (default: None)
   * - ``warmstart_n_trials``
     - int
     - Number of top trials to enqueue (default: 10)

CLI Commands
^^^^^^^^^^^^

.. code-block:: bash

   # List studies
   admet model hpo-list-studies [OPTIONS]

   Options:
     --storage-dir PATH    Directory containing studies.db [default: hpo_results/optuna_studies]
     --verbose, -v         Show detailed trial information

Related Documentation
---------------------

- :doc:`hpo` - Main HPO guide
- :doc:`configuration` - Configuration file structure
- `Optuna Documentation <https://optuna.readthedocs.io/>`_ - Underlying framework
- `Ray Tune Optuna Integration <https://docs.ray.io/en/latest/tune/api/integration.html#optuna>`_

Examples Repository
-------------------

Complete examples are available in ``configs/1-hpo-single/``:

- ``hpo_chemprop_warmstart_example.yaml`` - Full warmstart configuration
- ``hpo_chemprop.yaml`` - Base configuration with persistence options
- ``hpo_chemeleon.yaml`` - Chemeleon model with persistence options
