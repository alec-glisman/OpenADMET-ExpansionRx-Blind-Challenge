"""
Task Affinity Grouping for Multi-Task Learning
===============================================

This module implements the Task Affinity Grouping (TAG) algorithm from
"Efficiently Identifying Task Groupings for Multi-Task Learning"
(Fifty et al., NeurIPS 2021, https://arxiv.org/abs/2109.04617).

The TAG algorithm computes inter-task affinity scores by measuring the cosine
similarity between per-task gradients with respect to shared encoder parameters
during a short training run. These affinity scores are then used to cluster
tasks into groups that benefit from being trained together.

Algorithm Overview
------------------

1. **Affinity Computation**: Run a short joint training phase (typically 1-2 epochs)
   and compute per-task gradients with respect to shared encoder parameters at
   each training step.

2. **Affinity Scoring**: Measure cosine similarity between gradient vectors to
   quantify task affinity. High positive values indicate tasks that benefit from
   joint training; negative values suggest potential negative transfer.

3. **Task Clustering**: Group tasks using hierarchical or spectral clustering based
   on the computed affinity matrix. Tasks with high mutual affinity are grouped
   together.

4. **Multi-Head Training**: Train a single model with separate prediction heads for
   each task group, allowing beneficial knowledge sharing within groups while
   avoiding negative transfer across groups.

Key Components
--------------
- :class:`TaskAffinityConfig`: Configuration for task affinity computation
- :class:`TaskAffinityComputer`: Computes gradient-based task affinity matrix
- :class:`TaskGrouper`: Clusters tasks based on affinity scores
- :func:`compute_task_affinity`: High-level function to compute task affinity

Configuration Parameters
------------------------

Basic Parameters
^^^^^^^^^^^^^^^^

enabled : bool, default=False
    Whether to enable task affinity grouping. When False, standard multi-task
    learning is used.

n_groups : int, default=3
    Number of task groups to create via clustering. Controls how many separate
    prediction heads will be created. Must satisfy: 1 ≤ n_groups ≤ num_tasks.
    Typical range: 2-5 groups for 5-20 tasks.

Affinity Computation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

affinity_epochs : int, default=1
    Number of epochs for the affinity computation phase. More epochs provide
    more stable gradient statistics but increase computation time. For most
    datasets, 1-2 epochs are sufficient.

affinity_batch_size : int, default=64
    Batch size during affinity computation. Larger batches provide more stable
    gradient estimates but use more memory. Adjust based on dataset size and
    available memory.

affinity_lr : float, default=1e-3
    Learning rate during affinity computation. The goal is to compute meaningful
    gradients without over-training. Default value (1e-3) works well in most cases.

Affinity Scoring Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

affinity_type : str, default="cosine"
    Type of affinity measure: "cosine" or "dot_product". Cosine similarity
    (recommended) focuses on gradient direction and is robust to different loss
    scales. Dot product considers both direction and magnitude.

normalize_gradients : bool, default=True
    Whether to normalize gradient vectors before computing affinity. When True,
    focuses on gradient direction rather than magnitude, providing better
    robustness to different task scales.

Clustering Parameters
^^^^^^^^^^^^^^^^^^^^^

clustering_method : str, default="agglomerative"
    Clustering algorithm: "agglomerative" (hierarchical) or "spectral".
    Agglomerative clustering is recommended for most cases as it's more
    interpretable and stable.

Advanced Parameters
^^^^^^^^^^^^^^^^^^^

encoder_param_patterns : List[str], default=[]
    String patterns to identify shared encoder parameters. If empty, uses default
    exclusion patterns (predictor, ffn, output, readout, head). Custom patterns
    are only needed for non-standard architectures.

device : str, default="auto"
    Computation device: "auto" (selects CUDA if available), "cpu", "cuda", or
    "cuda:N" for specific GPU.

seed : int, default=42
    Random seed for reproducibility in data shuffling and spectral clustering.

log_affinity_matrix : bool, default=True
    Whether to log the computed affinity matrix to console/log file for inspection.

Usage Example
-------------
>>> from admet.model.chemprop.task_affinity import (
...     TaskAffinityConfig,
...     compute_task_affinity,
... )
>>> import pandas as pd
>>>
>>> # Load training data
>>> df_train = pd.read_csv("train.csv")
>>> target_cols = ["LogD", "KSOL", "PAMPA", "hERG", "CLint"]
>>>
>>> # Configure task affinity
>>> config = TaskAffinityConfig(
...     enabled=True,
...     n_groups=3,
...     affinity_epochs=1,
...     affinity_batch_size=64,
... )
>>>
>>> # Compute affinity and groups
>>> affinity_matrix, task_names, groups = compute_task_affinity(
...     df_train=df_train,
...     smiles_col="SMILES",
...     target_cols=target_cols,
...     config=config,
... )
>>>
>>> print("Task groups:", groups)
>>> # Output: [['LogD', 'KSOL'], ['PAMPA', 'hERG'], ['CLint']]

Interpreting Results
--------------------

Affinity scores range from -1 to 1:

- **High positive (0.7 - 1.0)**: Tasks strongly benefit from joint training
- **Moderate positive (0.3 - 0.7)**: Tasks somewhat compatible
- **Near zero (-0.3 - 0.3)**: Tasks are independent
- **Negative (-1.0 - -0.3)**: Tasks may interfere with each other

Tasks are clustered into groups where:

- Tasks within a group share a prediction head (multi-task learning)
- Tasks across groups use separate heads (isolated learning)
- This balances positive transfer within groups against negative transfer across groups

When to Use
-----------

**Recommended for:**

- 5+ tasks with varying properties
- Heterogeneous task types (e.g., mixed ADMET properties)
- When negative transfer is suspected
- Limited data per task

**May skip for:**

- < 5 tasks (manual grouping easier)
- Very homogeneous tasks (all benefit from joint training)
- Extremely large task sets (> 100 tasks)

References
----------
Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., & Finn, C. (2021).
Efficiently Identifying Task Groupings for Multi-Task Learning.
Advances in Neural Information Processing Systems, 34.
https://arxiv.org/abs/2109.04617

See Also
--------
- Full documentation: docs/guide/task_affinity.rst
- Google Research TAG implementation: https://github.com/google-research/google-research/tree/master/tag
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from chemprop import data, featurizers, models, nn as chemprop_nn
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from torch.utils.data import DataLoader

from admet.data.smiles import parallel_canonicalize_smiles

# Module logger
logger = logging.getLogger("admet.model.chemprop.task_affinity")


@dataclass
class TaskAffinityConfig:
    """
    Configuration for task affinity computation and grouping.

    The task affinity algorithm runs a short training phase to compute
    gradient-based affinity scores between tasks, then clusters tasks
    into groups based on their affinity.

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable task affinity grouping.
    affinity_epochs : int, default=1
        Number of epochs for the affinity computation phase.
    affinity_batch_size : int, default=64
        Batch size during affinity computation.
    affinity_lr : float, default=1e-3
        Learning rate during affinity computation.
    n_groups : int, default=3
        Number of task groups to create via clustering.
    clustering_method : str, default="agglomerative"
        Clustering algorithm: "agglomerative" or "spectral".
    affinity_type : str, default="cosine"
        Type of affinity to compute: "cosine" or "dot_product".
    normalize_gradients : bool, default=True
        Whether to normalize gradient vectors before computing affinity.
    encoder_param_patterns : List[str], optional
        Patterns to identify shared encoder parameters. If None, uses
        default patterns that exclude predictor/FFN layers.
    device : str, default="auto"
        Device for computation: "auto", "cpu", or "cuda".
    seed : int, default=42
        Random seed for reproducibility.
    log_affinity_matrix : bool, default=True
        Whether to log the computed affinity matrix.
    """

    enabled: bool = False
    affinity_epochs: int = 1
    affinity_batch_size: int = 64
    affinity_lr: float = 1.0e-3
    n_groups: int = 3
    clustering_method: str = "agglomerative"
    affinity_type: str = "cosine"
    normalize_gradients: bool = True
    encoder_param_patterns: List[str] = field(default_factory=list)
    device: str = "auto"
    seed: int = 42
    log_affinity_matrix: bool = True
    save_plots: bool = False
    plot_dpi: int = 150


def _get_device(device_str: str) -> torch.device:
    """
    Resolve device string to torch.device.

    Parameters
    ----------
    device_str : str
        Device string: "auto", "cpu", or "cuda".

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _is_encoder_param(name: str, patterns: List[str]) -> bool:
    """
    Determine if a parameter belongs to the shared encoder.

    Parameters
    ----------
    name : str
        Parameter name.
    patterns : List[str]
        Patterns to match for encoder parameters.

    Returns
    -------
    bool
        True if the parameter is an encoder parameter.
    """
    # Default exclusion patterns for predictor/FFN layers
    exclude_patterns = [
        "predictor",
        "predict",
        "ffn",
        "output",
        "readout",
        "head",
    ]

    # If custom patterns provided, use them
    if patterns:
        return any(p in name.lower() for p in patterns)

    # Default: exclude predictor-related parameters
    return not any(p in name.lower() for p in exclude_patterns)


def _masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Compute MSE loss over non-NaN entries.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions tensor.
    target : torch.Tensor
        Target tensor (may contain NaN values).

    Returns
    -------
    Optional[torch.Tensor]
        MSE loss scalar, or None if no valid entries.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return None
    diff = pred[mask] - target[mask]
    return (diff**2).mean()


def _flatten_gradients(grads: Tuple[Optional[torch.Tensor], ...], device: torch.device) -> np.ndarray:
    """
    Flatten a tuple of gradients to a single 1D numpy array.

    Parameters
    ----------
    grads : Tuple[Optional[torch.Tensor], ...]
        Tuple of gradient tensors (some may be None).
    device : torch.device
        Device for creating zero tensors.

    Returns
    -------
    np.ndarray
        Flattened gradient vector as numpy array.
    """
    parts = []
    for g in grads:
        if g is None:
            parts.append(torch.zeros(1, device=device))
        else:
            parts.append(g.reshape(-1))
    flat = torch.cat(parts)
    return flat.detach().cpu().numpy()


class TaskAffinityComputer:
    """
    Compute gradient-based task affinity matrix.

    This class implements the core TAG algorithm: running a short training
    phase and computing per-task gradients with respect to shared encoder
    parameters to measure inter-task affinity.

    Parameters
    ----------
    config : TaskAffinityConfig
        Configuration for affinity computation.

    Attributes
    ----------
    config : TaskAffinityConfig
        The configuration object.
    device : torch.device
        The computation device.
    affinity_matrix : Optional[np.ndarray]
        The computed affinity matrix (T x T).
    task_names : Optional[List[str]]
        Names of the tasks.

    Examples
    --------
    >>> config = TaskAffinityConfig(affinity_epochs=1)
    >>> computer = TaskAffinityComputer(config)
    >>> affinity, tasks = computer.compute_from_dataframe(
    ...     df_train, smiles_col="SMILES", target_cols=["LogD", "KSOL"]
    ... )
    """

    def __init__(self, config: TaskAffinityConfig) -> None:
        """
        Initialize the TaskAffinityComputer.

        Parameters
        ----------
        config : TaskAffinityConfig
            Configuration for affinity computation.
        """
        self.config = config
        self.device = _get_device(config.device)
        self.affinity_matrix: Optional[np.ndarray] = None
        self.task_names: Optional[List[str]] = None

    def compute_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        target_cols: List[str],
        model: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute task affinity matrix from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with SMILES and target columns.
        smiles_col : str
            Name of the SMILES column.
        target_cols : List[str]
            List of target column names.
        model : Optional[nn.Module]
            Pre-built model. If None, creates a default MPNN.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Tuple of (affinity_matrix, task_names).
        """
        logger.info("Building dataset for task affinity computation...")

        # Canonicalize SMILES and convert to Mol objects
        from rdkit import Chem

        smiles_list = parallel_canonicalize_smiles(df[smiles_col].tolist())
        targets = df[target_cols].values.astype(float).tolist()

        # Build chemprop dataset with Mol objects
        datapoints = []
        for smi, y in zip(smiles_list, targets):
            if smi is not None:  # Skip invalid SMILES
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    dp = data.MoleculeDatapoint(mol, y)
                    datapoints.append(dp)

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dataset = data.MoleculeDataset(datapoints, featurizer=featurizer)

        loader = data.build_dataloader(
            dataset,
            batch_size=self.config.affinity_batch_size,
            num_workers=0,
            shuffle=True,
        )

        # Build model if not provided
        if model is None:
            model = self._build_default_model(len(target_cols))

        return self.compute(loader, model, target_cols)

    def _build_default_model(self, n_tasks: int) -> nn.Module:
        """
        Build a default MPNN model for affinity computation.

        Parameters
        ----------
        n_tasks : int
            Number of prediction tasks.

        Returns
        -------
        nn.Module
            The MPNN model.
        """
        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.RegressionFFN(n_tasks=n_tasks)
        mpnn = models.MPNN(mp, agg, ffn)
        return mpnn

    def compute(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        target_cols: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute task affinity matrix using gradient-based method.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data.
        model : nn.Module
            The neural network model (MPNN).
        target_cols : List[str]
            List of target column names.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Tuple of (affinity_matrix, task_names).
        """
        logger.info(
            "Computing task affinity matrix for %d tasks over %d epochs...",
            len(target_cols),
            self.config.affinity_epochs,
        )

        model = model.to(self.device)
        model.train()

        # Identify shared encoder parameters
        shared_params = []
        shared_param_names = []
        for name, p in model.named_parameters():
            if _is_encoder_param(name, self.config.encoder_param_patterns):
                shared_params.append(p)
                shared_param_names.append(name)

        if len(shared_params) == 0:
            logger.warning("No shared encoder parameters found, using all parameters")
            shared_params = list(model.parameters())
            shared_param_names = [n for n, _ in model.named_parameters()]

        logger.debug("Using %d shared encoder parameters for affinity computation", len(shared_params))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.affinity_lr)

        n_tasks = len(target_cols)
        affinity_sum = np.zeros((n_tasks, n_tasks), dtype=float)
        count = 0

        for epoch in range(self.config.affinity_epochs):
            for batch in dataloader:
                # Move batch to device
                bmg, _, _, targets, *_ = batch
                # BatchMolGraph.to mutates in-place and returns None, avoid reassigning
                bmg.to(self.device)
                targets = targets.to(self.device).float()

                # Forward pass
                preds = model(bmg)

                # Compute per-task losses
                per_task_losses = []
                for t in range(n_tasks):
                    loss_t = _masked_mse_loss(preds[:, t], targets[:, t])
                    per_task_losses.append(loss_t)

                # Compute per-task gradient vectors
                grad_vecs: List[Optional[np.ndarray]] = [None] * n_tasks
                for t, loss_t in enumerate(per_task_losses):
                    if loss_t is None:
                        continue
                    grads = torch.autograd.grad(
                        loss_t,
                        shared_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_vecs[t] = _flatten_gradients(grads, self.device)

                # Accumulate affinity scores (dot products or cosine similarity)
                for i in range(n_tasks):
                    if grad_vecs[i] is None:
                        continue
                    for j in range(i, n_tasks):
                        if grad_vecs[j] is None:
                            continue

                        if self.config.affinity_type == "cosine":
                            # Cosine similarity
                            norm_i = np.linalg.norm(grad_vecs[i])
                            norm_j = np.linalg.norm(grad_vecs[j])
                            if norm_i > 0 and norm_j > 0:
                                affinity = float(np.dot(grad_vecs[i], grad_vecs[j]) / (norm_i * norm_j))
                            else:
                                affinity = 0.0
                        else:
                            # Raw dot product
                            affinity = float(np.dot(grad_vecs[i], grad_vecs[j]))

                        affinity_sum[i, j] += affinity
                        if i != j:
                            affinity_sum[j, i] += affinity

                count += 1

                # Perform optimization step on combined loss
                valid_losses = [loss for loss in per_task_losses if loss is not None]
                if len(valid_losses) > 0:
                    combined_loss = torch.stack(valid_losses).mean()
                    optimizer.zero_grad()
                    combined_loss.backward()
                    optimizer.step()

        if count == 0:
            raise RuntimeError("No gradient steps were performed; check your dataset")

        # Normalize affinity matrix
        affinity_matrix = affinity_sum / count

        # For dot product affinity, convert to cosine-style normalization
        if self.config.affinity_type == "dot_product":
            cosine_aff = np.zeros_like(affinity_matrix)
            for i in range(n_tasks):
                for j in range(n_tasks):
                    denom = math.sqrt(abs(affinity_matrix[i, i]) * abs(affinity_matrix[j, j]))
                    if denom > 0:
                        cosine_aff[i, j] = affinity_matrix[i, j] / denom
                    else:
                        cosine_aff[i, j] = 0.0
            affinity_matrix = cosine_aff

        self.affinity_matrix = affinity_matrix
        self.task_names = list(target_cols)

        if self.config.log_affinity_matrix:
            logger.info("Task affinity matrix computed:")
            for i, task_i in enumerate(target_cols):
                row_str = " ".join([f"{affinity_matrix[i, j]:.3f}" for j in range(n_tasks)])
                logger.info("  %s: [%s]", task_i, row_str)

        return affinity_matrix, list(target_cols)


class TaskGrouper:
    """
    Cluster tasks into groups based on affinity scores.

    This class implements the network selection step of TAG, using
    hierarchical clustering or spectral clustering to group tasks
    with high affinity together.

    Parameters
    ----------
    n_groups : int
        Number of task groups to create.
    method : str, default="agglomerative"
        Clustering method: "agglomerative" or "spectral".
    linkage : str, default="average"
        Linkage method for agglomerative clustering.
    seed : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    groups : Optional[List[List[str]]]
        The computed task groups.
    labels : Optional[np.ndarray]
        Cluster labels for each task.

    Examples
    --------
    >>> grouper = TaskGrouper(n_groups=3)
    >>> groups = grouper.cluster(affinity_matrix, task_names)
    >>> print(groups)
    [['LogD', 'KSOL'], ['PAMPA'], ['hERG', 'CLint']]
    """

    def __init__(
        self,
        n_groups: int,
        method: str = "agglomerative",
        linkage: str = "average",
        seed: int = 42,
    ) -> None:
        """
        Initialize the TaskGrouper.

        Parameters
        ----------
        n_groups : int
            Number of task groups to create.
        method : str, default="agglomerative"
            Clustering method: "agglomerative" or "spectral".
        linkage : str, default="average"
            Linkage method for agglomerative clustering.
        seed : int, default=42
            Random seed for reproducibility.
        """
        self.n_groups = n_groups
        self.method = method
        self.linkage = linkage
        self.seed = seed
        self.groups: Optional[List[List[str]]] = None
        self.labels: Optional[np.ndarray] = None

    def cluster(
        self,
        affinity_matrix: np.ndarray,
        task_names: List[str],
    ) -> List[List[str]]:
        """
        Cluster tasks based on affinity matrix.

        Parameters
        ----------
        affinity_matrix : np.ndarray
            Task affinity matrix (T x T).
        task_names : List[str]
            Names of the tasks.

        Returns
        -------
        List[List[str]]
            List of task groups, where each group is a list of task names.
        """
        n_tasks = len(task_names)

        # Validate clustering method early to ensure invalid methods raise
        if self.method not in ("agglomerative", "spectral"):
            raise ValueError("Unknown clustering method")

        # Handle edge cases
        if n_tasks <= self.n_groups:
            logger.warning(
                "Number of tasks (%d) <= number of groups (%d), " "assigning one task per group",
                n_tasks,
                self.n_groups,
            )
            self.groups = [[t] for t in task_names]
            self.labels = np.arange(n_tasks)
            return self.groups

        # Convert affinity to distance: higher affinity => smaller distance
        max_abs = np.max(np.abs(affinity_matrix)) if affinity_matrix.size else 1.0
        norm_aff = affinity_matrix / (max_abs + 1e-12)
        distance_matrix = 1.0 - norm_aff

        # Ensure distance matrix is valid
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.maximum(distance_matrix, 0.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        if self.method == "agglomerative":
            clustering = AgglomerativeClustering(
                n_clusters=self.n_groups,
                metric="precomputed",
                linkage=self.linkage,
            )
            labels = clustering.fit_predict(distance_matrix)
        elif self.method == "spectral":
            # For spectral clustering, use affinity directly
            # Ensure affinity is non-negative
            affinity_shifted = norm_aff - norm_aff.min() + 1e-6
            clustering = SpectralClustering(
                n_clusters=self.n_groups,
                affinity="precomputed",
                random_state=self.seed,
            )
            labels = clustering.fit_predict(affinity_shifted)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Build groups from labels
        groups_dict: Dict[int, List[str]] = defaultdict(list)
        for task_name, label in zip(task_names, labels):
            groups_dict[int(label)].append(task_name)

        self.groups = [groups_dict[k] for k in sorted(groups_dict.keys())]
        self.labels = labels

        logger.info("Task groups formed:")
        for i, group in enumerate(self.groups):
            logger.info("  Group %d: %s", i, group)

        return self.groups


def compute_task_affinity(
    df_train: pd.DataFrame,
    smiles_col: str,
    target_cols: List[str],
    config: Optional[TaskAffinityConfig] = None,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """
    Compute task affinity and cluster tasks into groups.

    This is a high-level convenience function that combines affinity
    computation and task clustering.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with SMILES and target columns.
    smiles_col : str
        Name of the SMILES column.
    save_path: Optional[str] = None
        Optional directory path to save affinity artifacts.
    List of target column names.
    config : Optional[TaskAffinityConfig]
        Configuration for affinity computation. If None, uses defaults.

    Returns
    -------
    Tuple[np.ndarray, List[str], List[List[str]]]
        Tuple of (affinity_matrix, task_names, task_groups).

    Examples
    --------
    >>> affinity, tasks, groups = compute_task_affinity(
    ...     df_train,
    ...     smiles_col="SMILES",
    ...     target_cols=["LogD", "KSOL", "PAMPA", "hERG"],
    ...     config=TaskAffinityConfig(n_groups=2),
    ... )
    """
    if config is None:
        config = TaskAffinityConfig()

    # Compute affinity matrix
    computer = TaskAffinityComputer(config)
    affinity_matrix, task_names = computer.compute_from_dataframe(df_train, smiles_col, target_cols)

    # Cluster tasks
    grouper = TaskGrouper(
        n_groups=config.n_groups,
        method=config.clustering_method,
        seed=config.seed,
    )
    groups = grouper.cluster(affinity_matrix, task_names)
    # Optionally save affinity artifacts
    if config is not None and getattr(config, "save_plots", False) and save_path:
        try:
            import matplotlib.pyplot as plt

            outdir = Path(save_path)
            outdir.mkdir(parents=True, exist_ok=True)
            # plotting functions are available in this module

            df = affinity_matrix_to_dataframe(affinity_matrix, task_names)
            csv_path = outdir / "affinity_matrix.csv"
            df.to_csv(csv_path)

            heat_path = outdir / "affinity_heatmap.png"
            fig_hm = plot_task_affinity_heatmap(
                affinity_matrix,
                task_names,
                save_path=str(heat_path),
                dpi=config.plot_dpi,
            )
            plt.close(fig_hm)

            clus_path = outdir / "affinity_clustermap.png"
            fig_cm = plot_task_affinity_clustermap(
                affinity_matrix,
                task_names,
                groups=groups,
                save_path=str(clus_path),
                dpi=config.plot_dpi,
            )
            plt.close(fig_cm)
        except Exception as e:
            logger.warning("Failed to save affinity artifacts: %s", e)

    return affinity_matrix, task_names, groups


def get_task_group_indices(
    groups: List[List[str]],
    target_cols: List[str],
) -> List[List[int]]:
    """
    Convert task groups (by name) to indices.

    Parameters
    ----------
    groups : List[List[str]]
        List of task groups, where each group is a list of task names.
    target_cols : List[str]
        List of all target column names.

    Returns
    -------
    List[List[int]]
        List of task groups, where each group is a list of task indices.

    Examples
    --------
    >>> groups = [['LogD', 'KSOL'], ['PAMPA', 'hERG']]
    >>> target_cols = ['LogD', 'KSOL', 'PAMPA', 'hERG']
    >>> get_task_group_indices(groups, target_cols)
    [[0, 1], [2, 3]]
    """
    task_to_idx = {t: i for i, t in enumerate(target_cols)}
    return [[task_to_idx[t] for t in group] for group in groups]


def affinity_matrix_to_dataframe(
    affinity_matrix: np.ndarray,
    task_names: List[str],
) -> pd.DataFrame:
    """
    Convert affinity matrix to a pandas DataFrame for visualization.

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Task affinity matrix (T x T).
    task_names : List[str]
        Names of the tasks.

    Returns
    -------
    pd.DataFrame
        DataFrame with task names as index and columns.

    Examples
    --------
    >>> df = affinity_matrix_to_dataframe(affinity, task_names)
    >>> df.to_csv("task_affinity.csv")
    """
    return pd.DataFrame(
        affinity_matrix,
        index=task_names,
        columns=task_names,
    )


def plot_task_affinity_heatmap(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    title: str = "Task Affinity Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Any:
    """
    Plot task affinity matrix as a heatmap.

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Task affinity matrix (T x T).
    task_names : List[str]
        Names of the tasks.
    title : str, default="Task Affinity Matrix"
        Plot title.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    cmap : str, default="RdBu_r"
        Colormap for the heatmap.
    save_path : Optional[str]
        Path to save the figure. If None, displays the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        affinity_matrix,
        xticklabels=task_names,
        yticklabels=task_names,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved task affinity heatmap to %s", save_path)

    return fig


def plot_task_affinity_clustermap(
    affinity_matrix: np.ndarray,
    task_names: List[str],
    groups: Optional[List[List[str]]] = None,
    title: str = "Task Affinity Clustermap",
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    dpi: int = 150,
    method: str = "average",
    metric: str = "euclidean",
) -> Any:
    """
    Plot a clustermap with dendrogram for the affinity matrix.

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Task affinity matrix (T x T).
    task_names : List[str]
        Names of the tasks.
    groups : Optional[List[List[str]]]
        Optional precomputed task groups used to color rows/cols.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap.
    save_path : Optional[str]
        Path to save the figure. If None, displays the figure.
    method : str
        Linkage method for hierarchical clustering.
    metric : str
        Distance metric for hierarchical clustering.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    """
    import seaborn as sns

    # Create a DataFrame for labels
    df = pd.DataFrame(affinity_matrix, index=task_names, columns=task_names)

    # Create a color mapping for groups if provided
    row_colors = None
    if groups is not None:
        # Create a label -> color mapping
        label_to_group = {}
        for idx, group in enumerate(groups):
            for t in group:
                label_to_group[t] = idx

        palette = sns.color_palette("tab10", n_colors=max(1, len(groups)))
        row_colors = [palette[label_to_group[t]] for t in task_names]

    g = sns.clustermap(
        df,
        cmap=cmap,
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        method=method,
        metric=metric,
        row_colors=row_colors,
        col_colors=row_colors,
        center=0,
        annot=True,
        fmt=".2f",
    )

    g.fig.suptitle(title)
    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved task affinity clustermap to %s", save_path)

    return g.fig
