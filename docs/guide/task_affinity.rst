===============================
Task Affinity Grouping
===============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Task Affinity Grouping (TAG) is an algorithm for efficiently identifying which tasks benefit from being trained together in multi-task learning. The method, introduced in `Fifty et al. (NeurIPS 2021) <https://arxiv.org/abs/2109.04617>`_, computes gradient-based affinity scores between tasks and uses these to cluster tasks into groups.

Key Benefits
------------

* **Improved Performance**: Groups tasks that positively influence each other
* **Efficiency**: Uses a short training run to compute affinities
* **Scalability**: Works with large numbers of tasks
* **Interpretability**: Provides quantitative affinity scores between tasks

Algorithm Overview
------------------

1. **Affinity Computation**: Run a short joint training phase and compute per-task gradients with respect to shared encoder parameters
2. **Affinity Scoring**: Measure cosine similarity between gradient vectors to quantify task affinity
3. **Task Clustering**: Group tasks using hierarchical or spectral clustering
4. **Multi-Head Training**: Train a single model with separate prediction heads for each task group

Configuration Parameters
========================

The ``TaskAffinityConfig`` class controls all aspects of task affinity computation and grouping.

Basic Parameters
----------------

enabled
^^^^^^^

:Type: ``bool``
:Default: ``False``
:Description:
    Whether to enable task affinity grouping. When ``False``, standard multi-task learning is used.
    When ``True``, the algorithm computes task affinities and groups tasks before training.

:Example:

.. code-block:: yaml

    task_affinity:
      enabled: true

n_groups
^^^^^^^^

:Type: ``int``
:Default: ``3``
:Description:
    Number of task groups to create via clustering. This controls how many separate prediction
    heads will be created in the final model. Each group will have its own decoder head that
    predicts all tasks in that group.

:Guidance:
    * Fewer groups: More sharing, potential for negative transfer
    * More groups: Less sharing, less potential for positive transfer
    * Typical range: 2-5 groups for 5-20 tasks
    * For N tasks, must have ``1 ≤ n_groups ≤ N``

:Example:

.. code-block:: yaml

    task_affinity:
      enabled: true
      n_groups: 3  # Create 3 task groups

Affinity Computation Parameters
--------------------------------

affinity_epochs
^^^^^^^^^^^^^^^

:Type: ``int``
:Default: ``1``
:Description:
    Number of epochs to run during the affinity computation phase. The algorithm performs a
    short training run to observe gradient patterns. More epochs provide more stable gradient
    statistics but increase computation time.

:Guidance:
    * Small datasets: 1-2 epochs usually sufficient
    * Large datasets: 1 epoch often enough
    * Noisy data: 2-3 epochs for more stable estimates
    * Each epoch computes gradients for all batches and accumulates affinity scores

:Example:

.. code-block:: yaml

    task_affinity:
      affinity_epochs: 2

affinity_batch_size
^^^^^^^^^^^^^^^^^^^

:Type: ``int``
:Default: ``64``
:Description:
    Batch size used during affinity computation. Affects gradient variance and computation speed.
    Larger batches provide more stable gradient estimates but use more memory.

:Guidance:
    * Small datasets: Use smaller batches (16-32)
    * Large datasets: Use larger batches (64-128)
    * Memory constraints: Reduce if running out of GPU memory
    * Trade-off: Larger batches = more stable but fewer gradient samples

:Example:

.. code-block:: yaml

    task_affinity:
      affinity_batch_size: 64

affinity_lr
^^^^^^^^^^^

:Type: ``float``
:Default: ``1e-3`` (0.001)
:Description:
    Learning rate used during the affinity computation phase. This controls how much the model
    parameters change during the short training run. The goal is to compute meaningful gradients
    without over-training.

:Guidance:
    * Standard value: ``1e-3`` works well in most cases
    * Unstable gradients: Try smaller values (``5e-4``, ``1e-4``)
    * Very small models: May use slightly larger (``2e-3``)
    * The learning rate affects gradient magnitudes, but affinity uses normalized gradients

:Example:

.. code-block:: yaml

    task_affinity:
      affinity_lr: 0.001

Affinity Scoring Parameters
----------------------------

affinity_type
^^^^^^^^^^^^^

:Type: ``str``
:Default: ``"cosine"``
:Options: ``"cosine"``, ``"dot_product"``
:Description:
    Type of affinity measure to compute between task gradients.

:Options:
    * ``"cosine"``: Cosine similarity between gradient vectors (normalized dot product).
      Range: [-1, 1]. Focuses on gradient direction rather than magnitude.
    * ``"dot_product"``: Raw dot product between gradient vectors. Considers both direction
      and magnitude. Automatically normalized to cosine-like values at the end.

:Guidance:
    * **Recommended**: ``"cosine"`` for most cases
    * ``"cosine"``: Better when tasks have different loss scales
    * ``"dot_product"``: May be useful when gradient magnitude is meaningful

:Example:

.. code-block:: yaml

    task_affinity:
      affinity_type: "cosine"

normalize_gradients
^^^^^^^^^^^^^^^^^^^

:Type: ``bool``
:Default: ``True``
:Description:
    Whether to normalize gradient vectors before computing affinity. When ``True``, each
    gradient vector is normalized to unit length before computing dot products.

:Guidance:
    * **Recommended**: ``True`` for most cases
    * ``True``: Focuses on gradient direction, robust to different loss scales
    * ``False``: Considers gradient magnitude, useful in special cases
    * Usually kept at ``True`` when ``affinity_type="cosine"``

:Example:

.. code-block:: yaml

    task_affinity:
      normalize_gradients: true

Clustering Parameters
---------------------

clustering_method
^^^^^^^^^^^^^^^^^

:Type: ``str``
:Default: ``"agglomerative"``
:Options: ``"agglomerative"``, ``"spectral"``
:Description:
    Clustering algorithm used to group tasks based on affinity scores.

:Options:
    * ``"agglomerative"``: Hierarchical agglomerative clustering. Builds a tree of task
      relationships and cuts at the specified number of groups. More interpretable and
      deterministic.
    * ``"spectral"``: Spectral clustering using graph theory. Can find non-convex clusters
      but requires more computation and is sensitive to parameters.

:Guidance:
    * **Recommended**: ``"agglomerative"`` for most cases
    * ``"agglomerative"``: Better for small-medium number of tasks (< 50)
    * ``"spectral"``: May help with complex affinity patterns
    * ``"agglomerative"``: More stable and reproducible

:Example:

.. code-block:: yaml

    task_affinity:
      clustering_method: "agglomerative"

Advanced Parameters
-------------------

encoder_param_patterns
^^^^^^^^^^^^^^^^^^^^^^

:Type: ``List[str]``
:Default: ``[]`` (uses default exclusion patterns)
:Description:
    List of string patterns to identify shared encoder parameters. Gradients are computed
    only with respect to these parameters. If empty, uses default patterns that exclude
    predictor/FFN/output layers.

:Default Exclusion Patterns:
    * ``"predictor"``
    * ``"predict"``
    * ``"ffn"``
    * ``"output"``
    * ``"readout"``
    * ``"head"``

:Guidance:
    * **Recommended**: Leave empty (``[]``) to use defaults
    * Custom patterns: Only specify if you have custom architecture
    * Parameters matching these patterns are **included** if list is non-empty
    * Parameters matching default exclusions are **excluded** if list is empty

:Example:

.. code-block:: yaml

    task_affinity:
      encoder_param_patterns: []  # Use defaults

.. code-block:: yaml

    task_affinity:
      encoder_param_patterns: ["encoder", "mpnn", "bond_message"]  # Custom

device
^^^^^^

:Type: ``str``
:Default: ``"auto"``
:Options: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
:Description:
    Device for computation during affinity computation phase.

:Options:
    * ``"auto"``: Automatically selects CUDA if available, otherwise CPU
    * ``"cpu"``: Force CPU computation
    * ``"cuda"``: Use default CUDA device
    * ``"cuda:N"``: Use specific CUDA device N

:Guidance:
    * **Recommended**: ``"auto"`` for most cases
    * GPU recommended for large models or datasets
    * CPU acceptable for small affinity computations (1-2 epochs)

:Example:

.. code-block:: yaml

    task_affinity:
      device: "auto"

seed
^^^^

:Type: ``int``
:Default: ``42``
:Description:
    Random seed for reproducibility. Affects data shuffling during affinity computation
    and spectral clustering (if used).

:Guidance:
    * Use consistent seed for reproducible experiments
    * Different seeds may produce slightly different groupings
    * Affinity scores themselves are deterministic given the same data order

:Example:

.. code-block:: yaml

    task_affinity:
      device: "cuda"
      seed: 42

log_affinity_matrix
^^^^^^^^^^^^^^^^^^^

:Type: ``bool``
:Default: ``True``
:Description:
    Whether to log the computed affinity matrix to the console/log file. When ``True``,
    prints a formatted matrix showing affinity scores between all task pairs.

:Example Output:

.. code-block:: text

    Task affinity matrix computed:
      LogD: [1.000 0.872 0.234 -0.156]
      KSOL: [0.872 1.000 0.189 -0.201]
      PAMPA: [0.234 0.189 1.000 0.445]
      hERG: [-0.156 -0.201 0.445 1.000]

:Example:

.. code-block:: yaml

    task_affinity:
      log_affinity_matrix: true

Complete Configuration Example
===============================

Basic Configuration
-------------------

Minimal configuration to enable task affinity grouping:

.. code-block:: yaml

    task_affinity:
      enabled: true
      n_groups: 3

Production Configuration
------------------------

Recommended configuration for production use:

.. code-block:: yaml

    task_affinity:
      # Core settings
      enabled: true
      n_groups: 3
      
      # Affinity computation
      affinity_epochs: 1
      affinity_batch_size: 64
      affinity_lr: 0.001
      
      # Affinity scoring
      affinity_type: "cosine"
      normalize_gradients: true
      
      # Clustering
      clustering_method: "agglomerative"
      
      # System
      device: "auto"
      seed: 42
      log_affinity_matrix: true

Advanced Configuration
----------------------

Configuration with all parameters explicitly set:

.. code-block:: yaml

    task_affinity:
      # Enable/disable
      enabled: true
      
      # Grouping
      n_groups: 4
      
      # Affinity computation phase
      affinity_epochs: 2
      affinity_batch_size: 32
      affinity_lr: 0.0005
      
      # Affinity scoring
      affinity_type: "cosine"
      normalize_gradients: true
      
      # Clustering
      clustering_method: "agglomerative"
      
      # Advanced
      encoder_param_patterns: []
      device: "cuda:0"
      seed: 123
      log_affinity_matrix: true

Usage Examples
==============

YAML Configuration Files
-------------------------

Basic Configuration (Recommended Starting Point)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``configs/chemprop_task_affinity.yaml``:

.. code-block:: yaml

    # Basic Chemprop configuration with task affinity
    model:
      type: "chemprop"
      message_passing:
        depth: 3
        hidden_size: 300
      ffn:
        num_layers: 2
        hidden_size: 300
    
    # Enable task affinity grouping
    task_affinity:
      enabled: true
      n_groups: 3
      affinity_epochs: 1
      affinity_batch_size: 64
      log_affinity_matrix: true
    
    # Training configuration
    training:
      epochs: 50
      batch_size: 64
      learning_rate: 0.001
      warmup_epochs: 2
    
    # Data configuration
    data:
      smiles_column: "SMILES"
      target_columns:
        - "LogD"
        - "KSOL"
        - "PAMPA"
        - "hERG"
        - "CLint"
      split_type: "scaffold"
      split_ratios: [0.8, 0.1, 0.1]

Advanced Configuration with All Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``configs/chemprop_task_affinity_advanced.yaml``:

.. code-block:: yaml

    # Advanced Chemprop configuration with full task affinity settings
    model:
      type: "chemprop"
      message_passing:
        depth: 4
        hidden_size: 400
        dropout: 0.1
      ffn:
        num_layers: 3
        hidden_size: 400
        dropout: 0.1
    
    # Full task affinity configuration
    task_affinity:
      # Enable/disable
      enabled: true
      
      # Grouping parameters
      n_groups: 4
      clustering_method: "agglomerative"  # or "spectral"
      
      # Affinity computation phase
      affinity_epochs: 2
      affinity_batch_size: 32
      affinity_lr: 0.0005
      
      # Affinity scoring
      affinity_type: "cosine"  # or "dot_product"
      normalize_gradients: true
      
      # Advanced settings
      encoder_param_patterns: []  # Use default exclusion patterns
      device: "cuda"
      seed: 42
      log_affinity_matrix: true
    
    # Training configuration
    training:
      epochs: 100
      batch_size: 64
      learning_rate: 0.001
      warmup_epochs: 5
      max_lr: 0.001
      final_lr: 0.0001
    
    # Data configuration
    data:
      smiles_column: "SMILES"
      target_columns:
        - "LogD"
        - "KSOL"
        - "PAMPA"
        - "Papp"
        - "hERG"
        - "CYP3A4"
        - "CLint"
        - "CLhep"
      split_type: "scaffold"
      split_ratios: [0.8, 0.1, 0.1]

Exploratory Configuration (Finding Optimal Groups)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``configs/chemprop_explore_affinity.yaml`` to try different group numbers:

.. code-block:: yaml

    # Configuration for exploring different task groupings
    model:
      type: "chemprop"
      message_passing:
        depth: 3
        hidden_size: 300
      ffn:
        num_layers: 2
        hidden_size: 300
    
    # Task affinity for exploration
    task_affinity:
      enabled: true
      n_groups: 2  # Try 2, 3, 4, 5 in separate runs
      affinity_epochs: 2  # More epochs for stable estimates
      affinity_batch_size: 64
      affinity_type: "cosine"
      clustering_method: "agglomerative"
      log_affinity_matrix: true  # Critical for analysis
    
    # Shorter training for quick experiments
    training:
      epochs: 30
      batch_size: 64
      learning_rate: 0.001
    
    data:
      smiles_column: "SMILES"
      target_columns:
        - "LogD"
        - "KSOL"
        - "PAMPA"
        - "hERG"
        - "CLint"
      split_type: "scaffold"
      split_ratios: [0.8, 0.1, 0.1]

Command Line Interface Examples
--------------------------------

Basic Usage with YAML Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a model with task affinity enabled:

.. code-block:: bash

    # Using a config file
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --data-path data/admet_train.csv \
      --save-dir models/chemprop_task_affinity

Override Config with CLI Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start with a base config and override specific parameters:

.. code-block:: bash

    # Override task affinity parameters via CLI
    python -m admet.cli.train \
      --config configs/chemprop.yaml \
      --task-affinity.enabled true \
      --task-affinity.n-groups 3 \
      --task-affinity.affinity-epochs 2 \
      --task-affinity.clustering-method agglomerative \
      --data-path data/admet_train.csv \
      --save-dir models/chemprop_affinity_3groups

Experiment with Different Group Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run multiple experiments to find optimal grouping:

.. code-block:: bash

    # Try 2 groups
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.n-groups 2 \
      --save-dir models/affinity_2groups

    # Try 3 groups
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.n-groups 3 \
      --save-dir models/affinity_3groups

    # Try 4 groups
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.n-groups 4 \
      --save-dir models/affinity_4groups

    # Try 5 groups
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.n-groups 5 \
      --save-dir models/affinity_5groups

Compare Clustering Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test different clustering algorithms:

.. code-block:: bash

    # Agglomerative clustering (hierarchical)
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.clustering-method agglomerative \
      --save-dir models/affinity_agglomerative

    # Spectral clustering (graph-based)
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.clustering-method spectral \
      --save-dir models/affinity_spectral

Compute Affinity Matrix Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute and save the affinity matrix without full training:

.. code-block:: bash

    # Quick affinity computation (1 epoch)
    python -m admet.cli.compute_task_affinity \
      --data-path data/admet_train.csv \
      --smiles-column SMILES \
      --target-columns LogD KSOL PAMPA hERG CLint \
      --affinity-epochs 1 \
      --affinity-batch-size 64 \
      --save-path results/task_affinity.npz \
      --plot-heatmap results/task_affinity_heatmap.png

Production Training with Optimized Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After determining optimal grouping, train production model:

.. code-block:: bash

    # Production training with task affinity
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --task-affinity.enabled true \
      --task-affinity.n-groups 3 \
      --task-affinity.affinity-epochs 1 \
      --data-path data/admet_full_train.csv \
      --save-dir models/production/chemprop_affinity \
      --seed 42 \
      --training.epochs 100

Python API Examples
-------------------

Using the configuration class directly:

.. code-block:: python

    from admet.model.chemprop.task_affinity import (
        TaskAffinityConfig,
        compute_task_affinity,
    )
    import pandas as pd

    # Load data
    df_train = pd.read_csv("data/admet_train.csv")
    target_cols = ["LogD", "KSOL", "PAMPA", "hERG", "CLint"]

    # Configure task affinity
    config = TaskAffinityConfig(
        enabled=True,
        n_groups=3,
        affinity_epochs=1,
        affinity_batch_size=64,
    )

    # Compute affinity and groups
    affinity_matrix, task_names, groups = compute_task_affinity(
        df_train=df_train,
        smiles_col="SMILES",
        target_cols=target_cols,
        config=config,
    )

    print("Task groups:", groups)
    # Output: [['LogD', 'KSOL'], ['PAMPA', 'hERG'], ['CLint']]

Complete Workflow: Compute, Visualize, and Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from admet.model.chemprop.task_affinity import (
        TaskAffinityConfig,
        compute_task_affinity,
        plot_task_affinity_heatmap,
        affinity_matrix_to_dataframe,
    )
    import pandas as pd

    # Load training data
    df_train = pd.read_csv("data/admet_train.csv")
    target_cols = ["LogD", "KSOL", "PAMPA", "hERG", "CLint", "CYP3A4"]

    # Step 1: Compute task affinity
    config = TaskAffinityConfig(
        enabled=True,
        n_groups=3,
        affinity_epochs=2,
        affinity_batch_size=64,
        affinity_type="cosine",
        clustering_method="agglomerative",
        log_affinity_matrix=True,
    )

    affinity_matrix, task_names, groups = compute_task_affinity(
        df_train=df_train,
        smiles_col="SMILES",
        target_cols=target_cols,
        config=config,
    )

    # Step 2: Save affinity matrix
    df_affinity = affinity_matrix_to_dataframe(affinity_matrix, task_names)
    df_affinity.to_csv("results/task_affinity_matrix.csv")

    # Step 3: Visualize affinity
    fig = plot_task_affinity_heatmap(
        affinity_matrix,
        task_names,
        title="ADMET Task Affinity Matrix",
        save_path="results/task_affinity_heatmap.png",
    )

    # Step 4: Print groups
    print("\nTask Groups Formed:")
    for i, group in enumerate(groups):
        print(f"  Group {i}: {group}")

    # Step 5: Train model with these groups
    # (use groups in your training pipeline)

Exploring Different Numbers of Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    from admet.model.chemprop.task_affinity import (
        TaskAffinityConfig,
        TaskAffinityComputer,
        TaskGrouper,
    )

    # Load data
    df_train = pd.read_csv("data/admet_train.csv")
    target_cols = ["LogD", "KSOL", "PAMPA", "hERG", "CLint"]

    # Compute affinity once
    config = TaskAffinityConfig(affinity_epochs=2, affinity_batch_size=64)
    computer = TaskAffinityComputer(config)
    affinity_matrix, task_names = computer.compute_from_dataframe(
        df_train, "SMILES", target_cols
    )

    # Try different numbers of groups
    for n_groups in [2, 3, 4, 5]:
        grouper = TaskGrouper(n_groups=n_groups, method="agglomerative")
        groups = grouper.cluster(affinity_matrix, task_names)
        
        print(f"\n{n_groups} Groups:")
        for i, group in enumerate(groups):
            print(f"  Group {i}: {group}")

Analyzing Affinity Results
---------------------------

Extracting and Examining the Affinity Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run training with affinity computation
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --data-path data/admet_train.csv \
      --save-dir models/affinity_analysis \
      --task-affinity.log-affinity-matrix true

    # The affinity matrix will be logged to console like:
    # Task affinity matrix computed:
    #   LogD: [1.000 0.872 0.234 -0.156 0.089]
    #   KSOL: [0.872 1.000 0.189 -0.201 0.045]
    #   PAMPA: [0.234 0.189 1.000 0.445 0.123]
    #   hERG: [-0.156 -0.201 0.445 1.000 0.567]
    #   CLint: [0.089 0.045 0.123 0.567 1.000]

Reading the Affinity Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the example above:

- **LogD ↔ KSOL (0.872)**: Very high positive affinity → should be grouped together
- **hERG ↔ CLint (0.567)**: Moderate positive affinity → can benefit from grouping
- **PAMPA ↔ hERG (0.445)**: Moderate affinity → borderline case
- **LogD ↔ hERG (-0.156)**: Negative affinity → should be in separate groups
- **LogD ↔ PAMPA (0.234)**: Weak positive affinity → may or may not group

Expected groupings for n_groups=3:

- Group 0: [LogD, KSOL] (solubility properties)
- Group 1: [PAMPA, hERG, CLint] (permeability/clearance)
- Or alternative based on dendrogram structure

Visualizing Affinity with Heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from admet.model.chemprop.task_affinity import (
        compute_task_affinity,
        plot_task_affinity_heatmap,
    )
    import pandas as pd

    # Compute affinity
    df_train = pd.read_csv("data/admet_train.csv")
    target_cols = ["LogD", "KSOL", "PAMPA", "hERG", "CLint"]
    
    affinity_matrix, task_names, groups = compute_task_affinity(
        df_train,
        smiles_col="SMILES",
        target_cols=target_cols,
    )

    # Create heatmap
    fig = plot_task_affinity_heatmap(
        affinity_matrix,
        task_names,
        title="ADMET Task Affinity Matrix",
        figsize=(10, 8),
        cmap="RdBu_r",  # Red for negative, blue for positive
        save_path="results/task_affinity_heatmap.png",
    )

Batch Processing Multiple Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    #!/bin/bash
    # Script: analyze_task_affinities.sh
    # Compute affinities for multiple datasets

    datasets=("admet_small" "admet_large" "admet_diverse")
    
    for dataset in "${datasets[@]}"; do
        echo "Analyzing dataset: $dataset"
        
        python -m admet.cli.compute_task_affinity \
          --data-path "data/${dataset}.csv" \
          --smiles-column SMILES \
          --target-columns LogD KSOL PAMPA hERG CLint \
          --affinity-epochs 2 \
          --affinity-batch-size 64 \
          --save-path "results/${dataset}_affinity.npz" \
          --plot-heatmap "results/${dataset}_affinity.png"
        
        echo "Results saved to results/${dataset}_affinity.*"
        echo "---"
    done

Practical Workflow
==================

Step-by-Step Guide to Using Task Affinity
------------------------------------------

Step 1: Initial Affinity Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start by computing the affinity matrix to understand task relationships:

.. code-block:: bash

    # Compute affinity matrix with visualization
    python -m admet.cli.compute_task_affinity \
      --data-path data/admet_train.csv \
      --smiles-column SMILES \
      --target-columns LogD KSOL PAMPA Papp hERG CYP3A4 CLint CLhep \
      --affinity-epochs 2 \
      --affinity-batch-size 64 \
      --clustering-method agglomerative \
      --save-path results/initial_affinity.npz \
      --plot-heatmap results/initial_affinity_heatmap.png

**What to look for:**

- Strong positive affinities (> 0.6) indicate tasks that should be grouped
- Negative affinities (< -0.2) indicate tasks that interfere
- Review the heatmap for natural clustering patterns

Step 2: Determine Optimal Number of Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run quick experiments with different group numbers:

.. code-block:: bash

    # Create experiment script
    cat > scripts/find_optimal_groups.sh << 'EOF'
    #!/bin/bash
    
    for n_groups in 2 3 4 5; do
        echo "Training with $n_groups groups..."
        
        python -m admet.cli.train \
          --config configs/chemprop_task_affinity.yaml \
          --data-path data/admet_train.csv \
          --task-affinity.n-groups $n_groups \
          --training.epochs 30 \
          --save-dir models/group_search/n${n_groups} \
          --seed 42
        
        echo "Evaluating model with $n_groups groups..."
        python -m admet.cli.evaluate \
          --model-dir models/group_search/n${n_groups} \
          --data-path data/admet_test.csv \
          --output results/eval_n${n_groups}.json
    done
    
    # Compare results
    python -m admet.cli.compare_models \
      --result-files results/eval_n*.json \
      --output results/group_comparison.csv
    EOF
    
    chmod +x scripts/find_optimal_groups.sh
    bash scripts/find_optimal_groups.sh

**Decision criteria:**

- Compare validation metrics across different n_groups
- Check for overfitting (train vs. val performance)
- Consider model complexity vs. performance gain
- Ensure groups make scientific sense

Step 3: Train Production Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once optimal grouping is determined, train the final model:

.. code-block:: bash

    # Train production model with optimal grouping (e.g., 3 groups)
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity_production.yaml \
      --data-path data/admet_full_train.csv \
      --task-affinity.enabled true \
      --task-affinity.n-groups 3 \
      --task-affinity.affinity-epochs 1 \
      --task-affinity.affinity-batch-size 64 \
      --training.epochs 100 \
      --save-dir models/production/chemprop_affinity_v1 \
      --seed 42

Step 4: Compare Against Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare task affinity model against standard approaches:

.. code-block:: bash

    # Single-task baseline
    python -m admet.cli.train \
      --config configs/chemprop_single_task.yaml \
      --data-path data/admet_train.csv \
      --save-dir models/baseline/single_task
    
    # Standard multi-task (no affinity)
    python -m admet.cli.train \
      --config configs/chemprop.yaml \
      --data-path data/admet_train.csv \
      --task-affinity.enabled false \
      --save-dir models/baseline/multi_task
    
    # Task affinity multi-task
    python -m admet.cli.train \
      --config configs/chemprop_task_affinity.yaml \
      --data-path data/admet_train.csv \
      --task-affinity.enabled true \
      --task-affinity.n-groups 3 \
      --save-dir models/affinity/multi_head
    
    # Compare all three
    python -m admet.cli.compare_models \
      --model-dirs models/baseline/single_task \
                    models/baseline/multi_task \
                    models/affinity/multi_head \
      --test-data data/admet_test.csv \
      --output results/final_comparison.csv

Decision Making Guide
---------------------

When to Use Task Affinity
^^^^^^^^^^^^^^^^^^^^^^^^^^

✅ **Use task affinity when:**

- You have 5-20 diverse prediction tasks
- Tasks span different ADMET categories (e.g., solubility, permeability, toxicity)
- Standard multi-task learning shows negative transfer
- You suspect some tasks help each other, but not all
- You have moderate amounts of data per task (100-1000+ samples)

❌ **Skip task affinity when:**

- You have < 5 tasks (manually group or use standard multi-task)
- All tasks are very similar (e.g., all solubility at different pH)
- You have extremely limited data (< 100 samples total)
- Computational resources are severely constrained
- You need the absolute simplest model

How to Choose Number of Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rule of thumb:**

- **2 groups**: Clear dichotomy in tasks (e.g., physicochemical vs. biological)
- **3 groups**: Most common choice for mixed ADMET tasks
- **4-5 groups**: Many heterogeneous tasks with complex relationships
- **N groups** (where N = num_tasks): Equivalent to single-task learning

**Empirical approach:**

1. Start with ``n_groups = ceil(num_tasks / 2.5)``
2. Try ±1 around this value
3. Select based on validation performance
4. Verify groups make scientific sense

**Example decisions:**

.. code-block:: text

    5 tasks → Try 2-3 groups
    8 tasks → Try 3-4 groups
    12 tasks → Try 4-5 groups
    20 tasks → Try 6-8 groups

Interpreting Affinity Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Strong positive affinity (0.7 - 1.0):**

- Tasks should definitely be grouped together
- Likely share similar molecular features
- Example: LogD and KSOL (both solubility-related)

**Moderate positive affinity (0.3 - 0.7):**

- Tasks may benefit from grouping
- Shared features but also unique aspects
- Example: PAMPA and Caco-2 (both permeability)

**Weak affinity (-0.3 - 0.3):**

- Tasks are largely independent
- Can group or separate based on other criteria
- Example: Solubility and metabolic stability

**Negative affinity (< -0.3):**

- Tasks interfere with each other
- Should be in separate groups
- Example: Sometimes physicochemical vs. complex biological endpoints

Common Configurations
=====================

Small ADMET Panel (5 tasks)
----------------------------

.. code-block:: yaml

    # configs/admet_small_affinity.yaml
    task_affinity:
      enabled: true
      n_groups: 2
      affinity_epochs: 1
      affinity_batch_size: 64
      clustering_method: "agglomerative"
    
    data:
      target_columns:
        - "LogD"      # Physicochemical
        - "KSOL"      # Physicochemical
        - "PAMPA"     # Permeability
        - "hERG"      # Safety
        - "CLint"     # Metabolism

Standard ADMET Panel (8 tasks)
-------------------------------

.. code-block:: yaml

    # configs/admet_standard_affinity.yaml
    task_affinity:
      enabled: true
      n_groups: 3
      affinity_epochs: 1
      affinity_batch_size: 64
      clustering_method: "agglomerative"
    
    data:
      target_columns:
        - "LogD"      # Physicochemical
        - "KSOL"      # Physicochemical
        - "PAMPA"     # Permeability
        - "Papp"      # Permeability
        - "hERG"      # Safety
        - "CYP3A4"    # Metabolism
        - "CLint"     # Metabolism
        - "CLhep"     # Metabolism

Extended ADMET Panel (15+ tasks)
---------------------------------

.. code-block:: yaml

    # configs/admet_extended_affinity.yaml
    task_affinity:
      enabled: true
      n_groups: 5
      affinity_epochs: 2
      affinity_batch_size: 32
      clustering_method: "agglomerative"
      affinity_type: "cosine"
    
    data:
      target_columns:
        # Physicochemical (likely Group 0)
        - "LogD"
        - "KSOL"
        - "pKa"
        
        # Permeability (likely Group 1)
        - "PAMPA"
        - "Papp"
        - "Caco2"
        
        # Safety (likely Group 2)
        - "hERG"
        - "AMES"
        - "Hepatotox"
        
        # Phase I Metabolism (likely Group 3)
        - "CYP1A2"
        - "CYP2C9"
        - "CYP2D6"
        - "CYP3A4"
        
        # Clearance (likely Group 4)
        - "CLint"
        - "CLhep"
        - "Thalf"

Performance Considerations
==========================

Computational Cost
------------------

Task affinity computation adds overhead:

* **Affinity phase**: 1-2 epochs of training (quick)
* **Clustering**: Negligible for < 50 tasks
* **Total overhead**: Typically < 5% of total training time

Memory Usage
------------

* Similar to regular training during affinity computation
* Slightly higher memory for gradient computation
* Use smaller ``affinity_batch_size`` if memory-constrained

When to Use Task Affinity
--------------------------

**Recommended for:**

* 5+ tasks with varying properties
* Heterogeneous task types (e.g., mixed ADMET properties)
* When negative transfer is suspected
* Limited data per task

**May skip for:**

* < 5 tasks (manual grouping easier)
* Very homogeneous tasks (e.g., all solubility measurements)
* Extremely large task sets (> 100 tasks, use approximate methods)

Troubleshooting
===============

Common Issues
-------------

No Gradient Steps
^^^^^^^^^^^^^^^^^

**Error**: ``RuntimeError: No gradient steps were performed``

**Causes**:

* All target values are NaN
* Batch size larger than dataset
* Invalid SMILES strings

**Solutions**:

* Check data for missing values
* Reduce ``affinity_batch_size``
* Validate SMILES with canonicalization

Unstable Affinity Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom**: Affinity matrix has extreme values or NaNs

**Solutions**:

* Increase ``affinity_epochs`` for more stable estimates
* Check for very imbalanced target distributions
* Normalize target values before training
* Use ``affinity_type="cosine"`` for robustness

Poor Groupings
^^^^^^^^^^^^^^

**Symptom**: Clustering produces unintuitive groups

**Solutions**:

* Try different ``n_groups`` values
* Use ``clustering_method="spectral"`` for non-convex clusters
* Manually inspect affinity matrix for patterns
* Consider increasing ``affinity_epochs``

Best Practices
==============

1. **Start Simple**: Begin with ``n_groups=3`` and default parameters
2. **Inspect Matrix**: Always log and visualize the affinity matrix
3. **Validate Groups**: Check that groups make scientific sense
4. **Tune Carefully**: Only adjust parameters if default results are poor
5. **Compare Baselines**: Compare against standard multi-task and single-task models
6. **Cross-Validate**: Use consistent grouping across CV folds

References
==========

* Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., & Finn, C. (2021).
  **Efficiently Identifying Task Groupings for Multi-Task Learning**.
  *Advances in Neural Information Processing Systems*, 34.
  `arXiv:2109.04617 <https://arxiv.org/abs/2109.04617>`_

* Implementation inspired by Google Research TAG:
  https://github.com/google-research/google-research/tree/master/tag

API Reference
=============

For detailed API documentation, see:

* :class:`admet.model.chemprop.task_affinity.TaskAffinityConfig`
* :class:`admet.model.chemprop.task_affinity.TaskAffinityComputer`
* :class:`admet.model.chemprop.task_affinity.TaskGrouper`
* :func:`admet.model.chemprop.task_affinity.compute_task_affinity`
