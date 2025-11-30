from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Sequence, Optional, Tuple, Union, Any, Callable
import argparse
import logging

from matplotlib.figure import Figure
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import mlflow
from tqdm import tqdm

import useful_rdkit_utils as uru
from rdkit import Chem  # type: ignore[import-not-found]

from bitbirch.bitbirch import bitbirch as bb

from admet.plot.split import (
    plot_cluster_size_histogram,
    plot_train_cluster_size_boxplots,
    plot_train_val_dataset_sizes,
    plot_endpoint_finite_value_counts,
)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def diameter_prune_tolerance_reassign(fps, tol=0.05, branching_factor=50, threshold=0.65):
    """
    Diameter + Prune + Tolerance + Reassign
    SOURCE: https://github.com/mqcomplab/bitbirch/blob/main/examples/refinement/refine_tutorial.ipynb
    """
    bb.set_merge("diameter")
    brc = bb.BitBirch(branching_factor=branching_factor, threshold=threshold)
    brc.fit(fps, singly=False)
    bb.set_merge("tolerance", tolerance=tol)
    brc.prune(fps)
    brc.reassign(fps)
    return brc


def get_bitbirch_clusters(smiles_list):
    LOGGER.debug("get_bitbirch_clusters: starting clustering for %d SMILES", len(smiles_list))

    # Data preparation
    mols = [
        Chem.MolFromSmiles(smiles)  # type: ignore
        for smiles in tqdm(smiles_list, desc="Parsing SMILES", dynamic_ncols=True)
    ]
    fps = np.array(
        [
            Chem.RDKFingerprint(mol)  # type: ignore
            for mol in tqdm(mols, desc="Computing fingerprints", dynamic_ncols=True)
        ]
    )

    # Input type checking
    is_binary = np.all(np.isin(fps, [0, 1]))
    is_int64 = fps.dtype == np.int64
    if fps.shape[0] != len(smiles_list):
        raise ValueError("Fingerprint array row count does not match number of SMILES.")
    if not is_binary:
        raise ValueError("Fingerprints must be binary.")
    if not is_int64:
        raise ValueError("Fingerprints must be of dtype int64.")

    LOGGER.debug("get_bitbirch_clusters: fingerprint array shape: %s, dtype: %s", fps.shape, fps.dtype)

    # Perform BitBirch clustering with diameter, prune, tolerance, and reassign
    brc = diameter_prune_tolerance_reassign(fps)
    cluster_list = brc.get_cluster_mol_ids()

    # Map each mol ID to its cluster ID
    n_molecules = len(fps)
    cluster_labels = [0] * n_molecules
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

    LOGGER.info("BitBirch clustering resulted in %d clusters for %d molecules.", len(cluster_list), n_molecules)
    LOGGER.debug("Cluster distribution: %s", [len(indices) for indices in cluster_list])

    return cluster_labels


DEFAULT_CLUSTER_METHODS: Dict[str, Callable] = {
    "random": uru.get_random_clusters,
    "scaffold": uru.get_bemis_murcko_clusters,
    "kmeans": uru.get_kmeans_clusters,  # n_clusters = 10 by default
    "umap": uru.get_umap_clusters,  # n_clusters = 7 by default
    "butina": uru.get_butina_clusters,  # cutoff = 0.65 by default
    "bitbirch": get_bitbirch_clusters,
}

# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------


@dataclass
class ClusterCVFold:
    """Container for a single fold of cluster-based CV."""

    fold_id: int

    # Molecule-level indices (map directly onto df.index)
    train_indices: np.ndarray
    val_indices: np.ndarray

    # Cluster IDs in this fold
    train_clusters: np.ndarray
    val_clusters: np.ndarray

    # Convenience stats
    n_train_mols: int
    n_val_mols: int
    n_train_clusters: int
    n_val_clusters: int


@dataclass
class ClusterCVDiagnostics:
    """
    Diagnostics & optional plots for cluster-based CV.

    Notes
    -----
    - `cluster_ids` and `cluster_sizes` describe all clusters.
    - `fold_train_cluster_sizes[fold_id]` = cluster sizes for training set of that fold.
    """

    cluster_ids: np.ndarray
    cluster_sizes: np.ndarray  # same order as cluster_ids

    # Per-fold training cluster sizes
    fold_train_cluster_sizes: Dict[int, np.ndarray]

    # Optional matplotlib figures (None if make_plots=False)
    hist_fig: Optional[Figure] = None
    boxplot_fig: Optional[Figure] = None
    bar_fig: Optional[Figure] = None
    endpoint_figs: Optional[Dict[int, Figure]] = None


def _build_cluster_cv_diagnostics(
    df: pd.DataFrame,
    task_cols: Sequence[str],
    folds: Sequence[ClusterCVFold],
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    make_plots: bool,
) -> ClusterCVDiagnostics:
    """Shared helper to assemble diagnostics and optional plots for cluster CV."""
    fold_train_cluster_sizes: Dict[int, np.ndarray] = {}
    for fold in folds:
        train_cluster_sizes = cluster_sizes[np.isin(cluster_ids, fold.train_clusters)]
        fold_train_cluster_sizes[fold.fold_id] = train_cluster_sizes

    diag = ClusterCVDiagnostics(
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        fold_train_cluster_sizes=fold_train_cluster_sizes,
    )

    if not make_plots:
        return diag

    hist_fig, _ = plot_cluster_size_histogram(
        cluster_sizes,
        title="Global cluster size distribution",
    )
    boxplot_fig, _ = plot_train_cluster_size_boxplots(
        fold_train_cluster_sizes,
        title="Training cluster size distribution across folds",
    )
    bar_fig, _ = plot_train_val_dataset_sizes(
        fold_train_sizes=[fold.n_train_mols for fold in folds],
        fold_val_sizes=[fold.n_val_mols for fold in folds],
        title="Training and Validation Set Sizes Across Folds",
    )
    endpoint_figs = {}
    for fold in folds:
        fig, _ = plot_endpoint_finite_value_counts(
            df_train=df.iloc[fold.train_indices],
            df_test=df.iloc[fold.val_indices],
            target_cols=list(task_cols),
            title=f"Fold {fold.fold_id} Endpoint Finite Value Counts",
        )
        endpoint_figs[fold.fold_id] = fig

    diag.hist_fig = hist_fig
    diag.boxplot_fig = boxplot_fig
    diag.bar_fig = bar_fig
    diag.endpoint_figs = endpoint_figs
    return diag


# -------------------------------------------------------------------
# Cluster Data for Splitting
# -------------------------------------------------------------------


def cluster_data(
    df: DataFrame,
    smiles_col: str = "SMILES",
    cluster_method: str = "bitbirch",
    cluster_params: Dict[str, Any] | None = None,
) -> Dict[int, Any]:
    """Cluster data using specified method."""
    if cluster_params is None:
        cluster_params = {}

    if cluster_method not in DEFAULT_CLUSTER_METHODS:
        raise ValueError(f"Unknown split method: {cluster_method}")

    LOGGER.debug("cluster_data: invoking clustering function '%s' for %d rows", cluster_method, len(df))
    LOGGER.debug("cluster_data: split_params=%s", cluster_params)
    clustering_function = DEFAULT_CLUSTER_METHODS[cluster_method]
    smiles_list = df[smiles_col].tolist()
    cluster_labels = clustering_function(smiles_list, **cluster_params)

    clusters: Dict[int, Any] = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(idx)

    LOGGER.info("Data clustered into %d clusters using method '%s'", len(clusters), cluster_method)
    # Provide more internal debug stats about cluster sizes
    cluster_counts = [len(indices) for indices in clusters.values()]
    LOGGER.debug(
        "Cluster sizes: min=%d, max=%d, mean=%.2f",
        min(cluster_counts),
        max(cluster_counts),
        float(np.mean(cluster_counts)),
    )
    return clusters


def build_cluster_label_matrix(
    df: pd.DataFrame,
    cluster_col: str,
    task_cols: Sequence[str],
    quality_col: Optional[str] = None,
    coverage_threshold: int = 1,
    add_cluster_size: bool = False,
    cluster_size_bins: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse molecule-level regression targets into cluster-level
    multi-label availability vectors.

    For regression tasks, we stratify based on which tasks are present
    (non-NaN) in each cluster, NOT on the regression values themselves.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule-level dataframe containing at least the `cluster_col` and
        `task_cols`. Each row is an assay for a molecule with possibly NaN
        target entries.
    cluster_col : str
        Column name in ``df`` that contains cluster assignments per row.
        Returned ``cluster_ids`` are sorted unique values from this column.
    task_cols : Sequence[str]
        Columns in ``df`` representing regression targets (endpoints). For
        each cluster, the function will compute whether each task is
        "present" by counting non-NaN entries per cluster and comparing the
        count to ``coverage_threshold``.
    coverage_threshold : int, default=1
        Minimum number of non-NaN observations in a cluster required to mark a
        task as present (1 => any non-NaN in the cluster marks presence for
        that task).
    add_cluster_size : bool, default=False
        If set to True, append additional one-hot columns to the right of the
        ``task_cols`` in the returned ``cluster_labels`` representing cluster
        size bins (quantile-based). These additional columns are suitable for
        use as additional label dimensions in stratified splitting.
    cluster_size_bins : int, default=5
        Number of quantile bins to create for cluster sizes when
        ``add_cluster_size=True``. If ``cluster_size_bins==1`` a single
        constant column of ones is appended. If ``cluster_size_bins>1``
        quantile edges are computed across clusters.

    Returns
    -------
    cluster_ids : numpy.ndarray
        1D array of sorted unique cluster identifiers corresponding to the
        rows of the returned ``cluster_labels`` and ``cluster_sizes``.
    cluster_labels : numpy.ndarray
        2-D integer array of shape ``(n_clusters, n_task_cols + n_quality_cols + n_size_bins)``,
        containing 0/1 indicators. The first ``len(task_cols)`` columns
        correspond in order to ``task_cols`` and contain ``1`` when the task
        is present in a cluster (>= ``coverage_threshold`` non-NaN values).
        If ``quality_col`` is provided, the next columns after the task columns
        will be one-hot presence flags for each unique level in ``quality_col``.
        If ``add_cluster_size=True`` additional columns are appended for
        size bins (one-hot encoded) with ``cluster_size_bins`` columns.
    cluster_sizes : numpy.ndarray
        1-D array of integer cluster sizes (number of rows per cluster), with
        length equal to ``len(cluster_ids)``.

    Raises
    ------
    ValueError
        If ``cluster_col`` or any ``task_cols`` are not present in ``df``.

    Examples
    --------
    >>> cluster_ids, cluster_labels, cluster_sizes = build_cluster_label_matrix(
    ...     df, cluster_col="cluster", task_cols=["A", "B"], quality_col=None, add_cluster_size=True
    ... )
    >>> cluster_labels.shape
    (n_clusters, 2 + n_size_bins)
    """
    # Prepare DataFrame
    df = df.copy()
    LOGGER.debug(
        "build_cluster_label_matrix: computing cluster labels for %d rows, cluster_col=%s",
        len(df),
        cluster_col,
    )
    LOGGER.debug(
        "build_cluster_label_matrix: %d task columns, add_cluster_size=%s",
        len(task_cols),
        add_cluster_size,
    )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Validate that required columns exist
    missing = [c for c in [cluster_col] + list(task_cols) if c not in df.columns]
    if quality_col is not None and quality_col not in df.columns:
        missing.append(quality_col)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    # Groupby cluster id and compute per-cluster metrics vectorized
    # cluster_size_series: index=cluster_id sorted
    cluster_size_series = df.groupby(cluster_col).size().sort_index()
    cluster_ids = cluster_size_series.index.to_numpy()
    cluster_sizes = cluster_size_series.to_numpy(dtype=int)

    # Compute non-NaN counts per cluster and per task (DataFrame): rows=clusters
    task_counts_df = df.groupby(cluster_col)[list(task_cols)].apply(lambda g: g.notna().sum())
    # Reindex to ensure sorted cluster_ids order
    task_counts_df = task_counts_df.reindex(cluster_ids)
    # Convert to binary presence (0/1) according to coverage_threshold
    cluster_labels = (task_counts_df.to_numpy(dtype=int) >= int(coverage_threshold)).astype(int)

    # Optionally append quality presence one-hot encoding (one column per unique category)
    if quality_col is not None:
        if quality_col not in df.columns:
            raise ValueError(f"quality_col '{quality_col}' not in DataFrame columns")
        # Build presence/absence of each quality category per cluster
        quality_counts = df.groupby(cluster_col)[quality_col].value_counts().unstack(fill_value=0).reindex(cluster_ids)
        if quality_counts.columns is not None:
            quality_counts = quality_counts.reindex(columns=sorted(quality_counts.columns.astype(str)))
        # Convert to binary presence flags
        quality_presence = (quality_counts.to_numpy(dtype=int) >= 1).astype(int)
        # Append quality presence columns after the task_cols before cluster size bins
        cluster_labels = np.concatenate([cluster_labels, quality_presence], axis=1)
        LOGGER.debug(
            "build_cluster_label_matrix: appended quality presence one-hot; new label matrix shape=%s",
            cluster_labels.shape,
        )

    # Optionally append binned cluster size one-hot encoding
    if add_cluster_size and cluster_size_bins > 0:
        if cluster_size_bins == 1:
            size_one_hot = np.ones((len(cluster_sizes), 1), dtype=int)
        else:
            # Compute interior quantile edges
            quantiles = np.linspace(0, 1, cluster_size_bins + 1)[1:-1]
            bin_edges = np.quantile(cluster_sizes, quantiles)

            # Assign each cluster to a bin [0, cluster_size_bins - 1]
            size_bin = np.digitize(cluster_sizes, bin_edges, right=False)

            size_one_hot = np.zeros((len(cluster_sizes), cluster_size_bins), dtype=int)
            size_one_hot[np.arange(len(cluster_sizes)), size_bin] = 1

        cluster_labels = np.concatenate([cluster_labels, size_one_hot], axis=1)
        LOGGER.debug(
            "build_cluster_label_matrix: appended cluster size one-hot; new label matrix shape=%s",
            cluster_labels.shape,
        )

    # Log final cluster information
    LOGGER.info("build_cluster_label_matrix: built labels for %d clusters", len(cluster_ids))
    LOGGER.debug(
        "build_cluster_label_matrix: cluster_sizes: min=%d, max=%d, median=%s",
        int(np.min(cluster_sizes)),
        int(np.max(cluster_sizes)),
        np.median(cluster_sizes),
    )
    return cluster_ids, cluster_labels, cluster_sizes


# -------------------------------------------------------------------
# K-fold Cluster CV methods
# -------------------------------------------------------------------


def cluster_multilabel_stratified_kfold(
    df: pd.DataFrame,
    cluster_col: str,
    task_cols: List[str],
    quality_col: Optional[str] = None,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    coverage_threshold: int = 1,
    add_cluster_size: bool = True,
    cluster_size_bins: int = 5,
    diagnostics: bool = False,
    make_plots: bool = False,
) -> Union[List[ClusterCVFold], Tuple[List[ClusterCVFold], ClusterCVDiagnostics]]:
    """
    Perform K-fold cross-validation over clusters, preserving:

      * Cluster integrity (clusters never split across folds)
      * Multi-task assay coverage balance across folds (for regression)
      * Approximate balance of cluster sizes across folds

    Additionally (optionally):

      * Compute diagnostics on cluster sizes and per-fold training sizes
      * Automatically generate:
          - Histogram of global cluster sizes
          - Boxplots of training cluster sizes across folds

    Parameters
    ----------
    df : pd.DataFrame
        Molecule-level dataframe with columns:
        - cluster_col
        - task_cols (regression targets, may have NaNs)
    cluster_col : str
        Cluster ID column (e.g., BitBirch output).
    task_cols : Sequence[str]
        Regression target column names.
    n_folds : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle clusters before splitting.
    random_state : int or None, default=42
        Random seed for reproducibility.
    coverage_threshold : int, default=1
        Min # assayed molecules in a cluster to treat a task as present.
    add_cluster_size : bool, default=True
        Whether to include binned cluster size as extra labels.
    cluster_size_bins : int, default=3
        Number of size bins (only if add_cluster_size=True).
    diagnostics : bool, default=False
        If True, return a `ClusterCVDiagnostics` object alongside folds.
    make_plots : bool, default=False
        If True, also create matplotlib figures and attach to diagnostics.
        Requires matplotlib to be installed.

    Returns
    -------
    folds : List[ClusterCVFold]
        One entry per fold with molecule- and cluster-level assignments.

    If diagnostics=True:
        (folds, diag) where diag is `ClusterCVDiagnostics`.
    quality_col : str, optional
        Optional column name in ``df`` to include quality categorical presence
        as additional one-hot columns used for stratification.
    quality_col : str, optional
        Optional column name in ``df`` to include quality categorical presence
        as additional one-hot columns used for stratification.
    """
    df = df.copy()
    LOGGER.debug(
        "cluster_multilabel_stratified_kfold: start with %d rows, cluster_col=%s",
        len(df),
        cluster_col,
    )
    LOGGER.debug(
        "cluster_multilabel_stratified_kfold: n_folds=%d, shuffle=%s, random_state=%s",
        n_folds,
        shuffle,
        random_state,
    )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Build cluster-level label matrix (incl. size, optionally)
    cluster_ids, cluster_labels, cluster_sizes = build_cluster_label_matrix(
        df,
        cluster_col=cluster_col,
        task_cols=task_cols,
        quality_col=quality_col,
        coverage_threshold=coverage_threshold,
        add_cluster_size=add_cluster_size,
        cluster_size_bins=cluster_size_bins,
    )

    splitter = MultilabelStratifiedKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )

    folds: List[ClusterCVFold] = []

    mol_indices = df.index.to_numpy()
    mol_clusters = df[cluster_col].to_numpy()

    for fold_id, (train_cluster_idx, val_cluster_idx) in tqdm(
        enumerate(splitter.split(cluster_ids, cluster_labels)),
        desc="Generating folds",
        total=n_folds,
        dynamic_ncols=True,
        leave=False,
    ):
        train_clusters = cluster_ids[train_cluster_idx]
        val_clusters = cluster_ids[val_cluster_idx]

        train_mask = np.isin(mol_clusters, train_clusters)
        val_mask = np.isin(mol_clusters, val_clusters)

        train_indices = mol_indices[train_mask]
        val_indices = mol_indices[val_mask]

        folds.append(
            ClusterCVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_clusters=train_clusters,
                val_clusters=val_clusters,
                n_train_mols=len(train_indices),
                n_val_mols=len(val_indices),
                n_train_clusters=len(train_clusters),
                n_val_clusters=len(val_clusters),
            )
        )
        LOGGER.debug(
            "cluster_multilabel_stratified_kfold: fold %d has %d train clusters and %d val clusters",
            fold_id,
            len(train_clusters),
            len(val_clusters),
        )

    if not diagnostics:
        return folds

    LOGGER.info("cluster_multilabel_stratified_kfold: completed generation of %d folds", len(folds))
    diag = _build_cluster_cv_diagnostics(
        df=df,
        task_cols=task_cols,
        folds=folds,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        make_plots=make_plots,
    )
    return folds, diag


def _encode_cluster_labels_for_stratification(cluster_labels: np.ndarray) -> np.ndarray:
    """Pack multi-label rows into a single integer for use with StratifiedKFold."""
    if cluster_labels.ndim == 1:
        return cluster_labels

    weights = np.array([1 << i for i in range(cluster_labels.shape[1])], dtype=np.int64)
    return cluster_labels.astype(np.int64) @ weights


def cluster_stratified_kfold(
    df: pd.DataFrame,
    cluster_col: str,
    task_cols: List[str],
    quality_col: Optional[str] = None,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    coverage_threshold: int = 1,
    add_cluster_size: bool = True,
    cluster_size_bins: int = 3,
    diagnostics: bool = False,
    make_plots: bool = False,
) -> Union[List[ClusterCVFold], Tuple[List[ClusterCVFold], ClusterCVDiagnostics]]:
    """Cluster-aware StratifiedKFold based on collapsed cluster-level labels."""
    df = df.copy()
    LOGGER.debug(
        "cluster_stratified_kfold: start with %d rows, cluster_col=%s, n_folds=%d",
        len(df),
        cluster_col,
        n_folds,
    )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    cluster_ids, cluster_labels, cluster_sizes = build_cluster_label_matrix(
        df,
        cluster_col=cluster_col,
        task_cols=task_cols,
        quality_col=quality_col,
        coverage_threshold=coverage_threshold,
        add_cluster_size=add_cluster_size,
        cluster_size_bins=cluster_size_bins,
    )
    encoded_labels = _encode_cluster_labels_for_stratification(cluster_labels)

    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )

    mol_indices = df.index.to_numpy()
    mol_clusters = df[cluster_col].to_numpy()

    folds: List[ClusterCVFold] = []
    try:
        split_iter = splitter.split(cluster_ids, encoded_labels)
    except ValueError as exc:
        LOGGER.warning(
            "cluster_stratified_kfold: stratification failed (falling back to GroupKFold): %s",
            exc,
        )
        return cluster_group_kfold(
            df,
            cluster_col=cluster_col,
            task_cols=task_cols,
            _quality_col=quality_col,
            n_folds=n_folds,
            random_state=random_state,
            diagnostics=diagnostics,
            make_plots=make_plots,
        )

    for fold_id, (train_cluster_idx, val_cluster_idx) in enumerate(split_iter):
        train_clusters = cluster_ids[train_cluster_idx]
        val_clusters = cluster_ids[val_cluster_idx]

        train_mask = np.isin(mol_clusters, train_clusters)
        val_mask = np.isin(mol_clusters, val_clusters)
        train_indices = mol_indices[train_mask]
        val_indices = mol_indices[val_mask]

        folds.append(
            ClusterCVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_clusters=train_clusters,
                val_clusters=val_clusters,
                n_train_mols=len(train_indices),
                n_val_mols=len(val_indices),
                n_train_clusters=len(train_clusters),
                n_val_clusters=len(val_clusters),
            )
        )
        LOGGER.debug(
            "cluster_stratified_kfold: fold %d has %d train clusters and %d val clusters",
            fold_id,
            len(train_clusters),
            len(val_clusters),
        )

    if not diagnostics:
        return folds

    LOGGER.info("cluster_stratified_kfold: completed generation of %d folds", len(folds))
    diag = _build_cluster_cv_diagnostics(
        df=df,
        task_cols=task_cols,
        folds=folds,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        make_plots=make_plots,
    )
    return folds, diag


def cluster_group_kfold(
    df: pd.DataFrame,
    cluster_col: str,
    task_cols: List[str],
    _quality_col: Optional[str] = None,
    n_folds: int = 5,
    random_state: Optional[int] = None,
    diagnostics: bool = False,
    make_plots: bool = False,
) -> Union[List[ClusterCVFold], Tuple[List[ClusterCVFold], ClusterCVDiagnostics]]:
    """
    Baseline: K-fold CV that ONLY respects clusters as groups,
    with no multi-task stratification.

    Optionally computes/plots the same cluster-size diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Molecule-level dataframe with column `cluster_col`.
    cluster_col : str
        Cluster ID column.
    endpoint_cols : Sequence[str]
        Regression target column names (for plotting only).
    n_folds : int, default=5
        Number of folds.
    random_state : int or None, default=None
        If provided, shuffle input order before splitting to get reproducible fold assignments.
    diagnostics : bool, default=False
        If True, return a `ClusterCVDiagnostics` object alongside folds.
    make_plots : bool, default=False
        If True, also create matplotlib figures and attach to diagnostics.
        Requires matplotlib to be installed.

    Returns
    -------
    folds : List[ClusterCVFold]

    If diagnostics=True:
        (folds, diag) where diag is `ClusterCVDiagnostics`.
    """
    df = df.copy()
    LOGGER.debug("cluster_group_kfold: starting GroupKFold with %d rows and %d folds", len(df), n_folds)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Optional shuffle to vary fold composition while respecting groups
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        df = df.iloc[rng.permutation(len(df))]

    mol_indices = df.index.to_numpy()
    groups = df[cluster_col].to_numpy()

    splitter = GroupKFold(n_splits=n_folds)
    folds: List[ClusterCVFold] = []

    # diagnostics pieces
    cluster_size_series = df.groupby(cluster_col).size().sort_index()
    cluster_ids = cluster_size_series.index.to_numpy()
    cluster_sizes = cluster_size_series.to_numpy()

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X=mol_indices, y=None, groups=groups)):
        train_clusters = np.unique(groups[train_idx])
        val_clusters = np.unique(groups[val_idx])

        folds.append(
            ClusterCVFold(
                fold_id=fold_id,
                train_indices=mol_indices[train_idx],
                val_indices=mol_indices[val_idx],
                train_clusters=train_clusters,
                val_clusters=val_clusters,
                n_train_mols=len(train_idx),
                n_val_mols=len(val_idx),
                n_train_clusters=len(train_clusters),
                n_val_clusters=len(val_clusters),
            )
        )

        LOGGER.debug(
            "cluster_group_kfold: fold %d has %d train clusters and %d val clusters",
            fold_id,
            len(train_clusters),
            len(val_clusters),
        )

    if not diagnostics:
        return folds

    LOGGER.info("cluster_group_kfold: completed generation of %d folds", len(folds))
    diag = _build_cluster_cv_diagnostics(
        df=df,
        task_cols=task_cols,
        folds=folds,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        make_plots=make_plots,
    )
    return folds, diag


def cluster_kfold(
    df: pd.DataFrame,
    cluster_col: str,
    task_cols: List[str],
    quality_col: Optional[str] = None,
    split_method: str = "multilabel_stratified_kfold",
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    coverage_threshold: int = 1,
    add_cluster_size: bool = True,
    cluster_size_bins: int = 3,
    diagnostics: bool = False,
    make_plots: bool = False,
) -> Union[List[ClusterCVFold], Tuple[List[ClusterCVFold], ClusterCVDiagnostics]]:
    """
    Unified entry point for cluster-aware K-fold CV.

    Dispatches to GroupKFold, StratifiedKFold, or MultilabelStratifiedKFold
    while keeping a single surface for diagnostics/plotting knobs.
    """
    method = split_method.lower()
    LOGGER.info(
        "cluster_kfold: method=%s n_folds=%d shuffle=%s random_state=%s",
        method,
        n_folds,
        shuffle,
        random_state,
    )

    if method == "group_kfold":
        return cluster_group_kfold(
            df,
            cluster_col=cluster_col,
            task_cols=task_cols,
            _quality_col=quality_col,
            n_folds=n_folds,
            random_state=random_state,
            diagnostics=diagnostics,
            make_plots=make_plots,
        )
    if method == "stratified_kfold":
        return cluster_stratified_kfold(
            df,
            cluster_col=cluster_col,
            task_cols=task_cols,
            quality_col=quality_col,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            coverage_threshold=coverage_threshold,
            add_cluster_size=add_cluster_size,
            cluster_size_bins=cluster_size_bins,
            diagnostics=diagnostics,
            make_plots=make_plots,
        )
    if method == "multilabel_stratified_kfold":
        return cluster_multilabel_stratified_kfold(
            df,
            cluster_col=cluster_col,
            task_cols=task_cols,
            quality_col=quality_col,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            coverage_threshold=coverage_threshold,
            add_cluster_size=add_cluster_size,
            cluster_size_bins=cluster_size_bins,
            diagnostics=diagnostics,
            make_plots=make_plots,
        )

    raise ValueError(f"Unknown split method: {split_method}")


# -------------------------------------------------------------------
# Overall pipeline
# -------------------------------------------------------------------


def pipeline(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    quality_col: str = "Quality",
    target_cols: Optional[List[str]] = None,
    cluster_method: str = "bitbirch",
    split_method: str = "multilabel_stratified_kfold",
    n_splits: int = 5,
    n_folds: int = 5,
    fig_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Example pipeline function that adds cluster assignments to the dataframe."""
    LOGGER.info(
        "pipeline: starting pipeline; cluster_method=%s split_method=%s n_splits=%d n_folds=%d",
        cluster_method,
        split_method,
        n_splits,
        n_folds,
    )

    # Default target columns if not provided (avoid mutable default arg)
    if target_cols is None:
        target_cols = [
            "LogD",
            "Log KSOL",
            "Log HLM CLint",
            "Log MLM CLint",
            "Log Caco-2 Permeability Papp A>B",
            "Log Caco-2 Permeability Efflux",
            "Log MPPB",
            "Log MBPB",
            "Log MGMB",
        ]

    # Cluster the data
    cluster_labels = cluster_data(
        df,
        smiles_col=smiles_col,
        cluster_method=cluster_method,
    )

    df = df.copy()
    for cluster_id, indices in cluster_labels.items():
        for idx in indices:
            df.at[idx, f"{cluster_method}_cluster"] = cluster_id
    if any(df[f"{cluster_method}_cluster"].isna()):
        raise ValueError("Some molecules were not assigned to any cluster.")
    LOGGER.info("pipeline: assigned %d molecules to clusters using method '%s'", len(df), cluster_method)

    # Repeated stratified K-fold splitting
    splits: List[List[ClusterCVFold]] = []
    diags: List[ClusterCVDiagnostics] = []
    for i in tqdm(range(n_splits), desc="Generating splits", dynamic_ncols=True):
        res = cluster_kfold(
            df,
            cluster_col=f"{cluster_method}_cluster",
            task_cols=list(target_cols),
            quality_col=quality_col,
            split_method=split_method,
            n_folds=n_folds,
            shuffle=True,
            random_state=42 + i,
            coverage_threshold=1,
            add_cluster_size=True,
            cluster_size_bins=5,
            diagnostics=True,
            make_plots=True,
        )
        if isinstance(res, tuple):
            folds, diag = res
        else:
            folds, diag = res, None
        splits.append(folds)
        if diag is not None:
            diags.append(diag)
        LOGGER.debug("pipeline: finished split %d", i)

    save_func: Optional[Callable[..., None]] = None
    if mlflow.active_run() is not None:

        def _mlflow_save(fig: Figure, path: str) -> None:
            mlflow.log_figure(fig, path)

        save_func = _mlflow_save  # type: ignore[assignment]
    elif fig_dir is not None:

        def _save_fig_local(fig: Figure, path: str) -> None:
            fig.savefig(Path(fig_dir) / path)

        save_func = _save_fig_local  # type: ignore[assignment]
    else:
        save_func = None

    if save_func is not None:
        for split_id, diag in enumerate(diags):
            if diag.hist_fig is not None:
                save_func(diag.hist_fig, f"split_{split_id}_cluster_size_histogram.png")
            if diag.boxplot_fig is not None:
                save_func(diag.boxplot_fig, f"split_{split_id}_train_cluster_size_boxplots.png")
            if diag.bar_fig is not None:
                save_func(diag.bar_fig, f"split_{split_id}_train_val_dataset_sizes.png")
            if diag.endpoint_figs is not None:
                for fold_id, fig in diag.endpoint_figs.items():
                    save_func(fig, f"split_{split_id}_fold_{fold_id}_endpoint_finite_value_counts.png")

    LOGGER.info("pipeline: creating training/validation fold columns in dataframe for %d splits", len(splits))
    df_assignments = df.copy()
    # Use train and val indices to generate fold assignment columns
    for split_id, folds in enumerate(splits):
        for fold in folds:
            fold_col = f"split_{split_id}_fold_{fold.fold_id}"
            df_assignments[fold_col] = "unassigned"
            df_assignments.loc[fold.train_indices, fold_col] = "train"
            df_assignments.loc[fold.val_indices, fold_col] = "validation"

            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)
            if not train_set.isdisjoint(val_set):
                raise ValueError(f"Train and validation sets overlap in {fold_col}.")

            all_indices = set(fold.train_indices).union(set(fold.val_indices))
            if all_indices != set(df.index):
                raise ValueError(f"Train and validation sets do not cover all data in {fold_col}.")

    LOGGER.debug("pipeline: finished pipeline; returning dataframe with %d columns", df.shape[1])
    return df_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run example data splitting pipeline")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input CSV file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for results",
        type=Path,
        default=Path("./temp/split_output"),
    )
    parser.add_argument(
        "--quality-col",
        help="Column name containing the quality bucket for stratification",
        type=str,
        default="Quality",
    )
    parser.add_argument(
        "--smiles-col",
        help="Column name with SMILES (defaults to 'SMILES')",
        type=str,
        default="SMILES",
    )
    parser.add_argument(
        "--target-cols",
        "--targets",
        help="List of target columns to use for stratification (space separated)",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--cluster-method",
        help="Clustering method for data splitting",
        type=str,
        default="bitbirch",
        choices=list(DEFAULT_CLUSTER_METHODS.keys()),
    )
    parser.add_argument(
        "-m",
        "--split-method",
        help="Data splitting method",
        type=str,
        default="multilabel_stratified_kfold",
        choices=["group_kfold", "stratified_kfold", "multilabel_stratified_kfold"],
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def example_default_pipeline(
    input_file: Path,
    output_dir: Path,
    cluster_method: str = "bitbirch",
    split_method: str = "multilabel_stratified_kfold",
    quality_col: str = "Quality",
    smiles_col: str = "SMILES",
    target_cols: Optional[List[str]] = None,
):
    """Example usage of the pipeline function with default parameters.

    Parameters
    ----------
    input_file : Path
        Path to input CSV file containing sample data.
    output_dir : Path
        Directory to save output clustered splits CSV.
    """
    output_dir = Path(output_dir) / cluster_method / split_method
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"

    for dir_path in [fig_dir, data_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("example_default_pipeline: reading sample data from %s", input_file)
    df = pd.read_csv(input_file, low_memory=False)
    df_assigned = pipeline(
        df,
        fig_dir=str(fig_dir),
        cluster_method=cluster_method,
        split_method=split_method,
        n_splits=5,
        n_folds=5,
        quality_col=quality_col,
        smiles_col=smiles_col,
        target_cols=target_cols,
    )

    # save data
    Path(output_dir / "data").mkdir(parents=True, exist_ok=True)
    df_assigned.to_csv(output_dir / "data" / "clustered_splits.csv", index=False)
    # for each fold and split, output train/val datasets as HuggingFace Datasets
    for split_id in range(5):
        for fold_id in range(5):
            fold_col = f"split_{split_id}_fold_{fold_id}"
            df_train = df[df_assigned[fold_col] == "train"]
            df_val = df[df_assigned[fold_col] == "val"]

            output_dir_split = data_dir / f"split_{split_id}/fold_{fold_id}"
            output_dir_split.mkdir(parents=True, exist_ok=True)
            df_train.to_csv(output_dir_split / "train.csv", index=False)
            df_val.to_csv(output_dir_split / "validation.csv", index=False)

    LOGGER.info("example_default_pipeline: wrote clustered splits to %s", output_dir)


if __name__ == "__main__":
    LOGGER.info("split.py invoked as main; running example_default_pipeline via argparse CLI")
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    example_default_pipeline(
        args.input,
        args.output,
        args.cluster_method,
        args.split_method,
        quality_col=args.quality_col,
    )
