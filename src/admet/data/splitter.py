"""admet.data.splitter
=======================

Dataset splitting utilities for multi‑dataset train/validation/test
stratification with chemical diversity preservation.

The :class:`DatasetSplitter` orchestrates a two‑level strategy:

1. Cluster molecules (per dataset group) using a selected method to capture
   structural diversity.
2. Apply group‑aware k‑fold shuffling so cluster composition is balanced
   across folds before carving out validation and test subsets.

Nested Output Structure
-----------------------
``create_splits`` returns a deeply nested mapping::

    {dataset_name: {method_name: {"split_{i}": {"fold_{j}": {
        group_name: {"train": idx_arr, "validation": idx_arr, "test": idx_arr},
        ...,
        "total": {"train": idx_arr, "validation": idx_arr, "test": idx_arr}
    }}}}

Where ``total`` contains concatenated indices across all groups for the fold.

Performance Notes
-----------------
Cluster assignment can dominate runtime depending on the method (e.g.
UMAP or Butina). The implementation uses in‑memory index arrays (``numpy``)
and performs a garbage collection pass per fold to reduce peak memory usage
for large datasets.

Extensibility
-------------
Custom clustering methods can be registered via ``add_split_method``. They
must accept a ``pandas.Series`` of SMILES and return a sequence of cluster
labels with the same length.
"""

from typing import Dict, Callable, Any
from pathlib import Path
import logging
import gc


from bitbirch.bitbirch import bitbirch as bb
from rdkit import Chem
import useful_rdkit_utils as uru

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_bitbirch_clusters(smiles_list):
    BRANCHING_FACTOR = 50
    THRESHOLD = 0.65
    bb.set_merge("radius")

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = np.array([Chem.RDKFingerprint(mol) for mol in mols])
    bitbirch = bb.BitBirch(branching_factor=BRANCHING_FACTOR, threshold=THRESHOLD)
    bitbirch.fit(fps)
    cluster_list = bitbirch.get_cluster_mol_ids()

    # Map each mol ID to its cluster ID
    n_molecules = len(fps)
    cluster_labels = [0] * n_molecules
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

    return cluster_labels


class DatasetSplitter:
    """Stratified multi‑dataset splitter.

    Creates stratified k‑fold splits across multiple datasets using cluster
    assignments (chemical diversity proxy) and a group shuffle strategy.

    Parameters
    ----------
    n_splits : int, optional
        Number of outer random splits to create (default ``5``).
    n_folds : int, optional
        Number of k‑folds within each split (default ``5``).
    test_percentage : float, optional
        Fraction of total samples allocated to the test subset (default ``0.1``).
    validation_percentage : float, optional
        Fraction of training portion allocated to validation (default ``0.1``).
    stratify_column : str, optional
        Column used to group data sources (default ``'Dataset'``).

    Attributes
    ----------
    split_methods : dict[str, callable]
        Registry of clustering functions mapping SMILES -> cluster labels.
    """

    # Default clustering methods
    DEFAULT_SPLIT_METHODS: Dict[str, Callable] = {
        "random_cluster": uru.get_random_clusters,
        "scaffold_cluster": uru.get_bemis_murcko_clusters,
        "kmeans_cluster": uru.get_kmeans_clusters,
        "umap_cluster": uru.get_umap_clusters,
        # "butina_cluster": uru.get_butina_clusters,
        "bitbirch_cluster": get_bitbirch_clusters,
    }

    def __init__(
        self,
        n_splits: int = 5,
        n_folds: int = 5,
        test_percentage: float = 0.1,
        validation_percentage: float = 0.1,
        stratify_column: str = "Dataset",
    ) -> None:
        """
        Initialize the dataset splitter.

        Parameters
        ----------
        n_splits : int, optional
            Number of random splits to create (default: 5).
        n_folds : int, optional
            Number of k-folds within each split (default: 5).
        test_percentage : float, optional
            Percentage of data to reserve for testing (default: 0.1).
        validation_percentage : float, optional
            Percentage of training data for validation (default: 0.1).
        stratify_column : str, optional
            Column name to stratify on (default: "Dataset").
        """
        self.n_splits = n_splits
        self.n_folds = n_folds
        self.test_percentage = test_percentage
        self.validation_percentage = validation_percentage
        self.stratify_column = stratify_column
        self.split_methods = self.DEFAULT_SPLIT_METHODS.copy()

        logger.debug(
            "Initialized DatasetSplitter: n_splits=%s, n_folds=%s, test_pct=%s, val_pct=%s",
            n_splits,
            n_folds,
            test_percentage,
            validation_percentage,
        )

    def add_split_method(self, name: str, method: Callable) -> None:
        """Register a custom clustering / split method.

        Parameters
        ----------
        name : str
            Identifier for later reference.
        method : callable
            Function accepting a ``pandas.Series`` of SMILES and returning a
            sequence of cluster labels of matching length.
        """
        self.split_methods[name] = method
        logger.debug("Added custom split method: %s", name)

    def _split_group_stratified(
        self,
        subdata: pd.DataFrame,
        split_method: Callable,
        random_state: int,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate k‑fold stratified indices for a single group.

        Parameters
        ----------
        subdata : pandas.DataFrame
            Subset corresponding to one group (value of ``stratify_column``).
        split_method : callable
            Cluster assignment function.
        random_state : int
            Seed base for reproducibility; per‑fold perturbations applied.

        Returns
        -------
        dict[int, dict[str, numpy.ndarray]]
            Mapping fold -> split type -> index array.
        """
        cluster_list = split_method(subdata["SMILES"])
        subdata_indices = subdata.index.to_numpy().copy()

        group_kfold_shuffle = uru.GroupKFoldShuffle(
            n_splits=self.n_folds, random_state=random_state, shuffle=True
        )

        fold_splits: Dict[int, Dict[str, np.ndarray]] = {}

        for fold_idx, (train_idx, test_idx) in enumerate(
            group_kfold_shuffle.split(subdata_indices, groups=cluster_list)
        ):
            # Map back to original indices
            train_idx_orig = subdata_indices[train_idx]
            test_idx_orig = subdata_indices[test_idx]

            # Split train into train/validation
            n_train_samples = len(train_idx_orig)
            n_val_samples = int(n_train_samples * self.validation_percentage)
            np.random.seed(random_state + fold_idx)
            shuffled_train_idx = np.random.permutation(train_idx_orig)
            val_idx = shuffled_train_idx[:n_val_samples]
            train_idx_final = shuffled_train_idx[n_val_samples:]

            fold_splits[fold_idx] = {
                "train": train_idx_final,
                "validation": val_idx,
                "test": test_idx_orig,
            }

            gc.collect()

        return fold_splits

    def create_splits(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Create stratified indices for all datasets and registered methods.

        Parameters
        ----------
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset name -> DataFrame containing at least ``SMILES``
            and ``stratify_column``.

        Returns
        -------
        dict
            Nested mapping (see module docstring for structure).

        Raises
        ------
        ValueError
            If fold assembly fails (missing fold key).
        """
        split_datasets: Dict[str, Any] = {}
        n_iter = len(datasets) * len(self.split_methods) * self.n_splits

        logger.info("Creating %d total dataset splits", n_iter)

        with tqdm(total=n_iter, desc="Creating dataset splits") as pbar:
            for dset_name, data in datasets.items():
                split_datasets[dset_name] = {}

                for split_name, split_method in self.split_methods.items():
                    logger.info("Processing dataset: %s, split: %s", dset_name, split_name)
                    split_datasets[dset_name][split_name] = {}

                    for split_id in range(self.n_splits):
                        split_datasets[dset_name][split_name][f"split_{split_id}"] = {}

                        for group in data[self.stratify_column].unique():
                            subdata = data[data[self.stratify_column] == group]
                            group_folds = self._split_group_stratified(subdata, split_method, split_id)

                            # Store group splits
                            for fold_id, fold_indices in group_folds.items():
                                if (
                                    f"fold_{fold_id}"
                                    not in split_datasets[dset_name][split_name][f"split_{split_id}"]
                                ):
                                    split_datasets[dset_name][split_name][f"split_{split_id}"][
                                        f"fold_{fold_id}"
                                    ] = {}

                                split_datasets[dset_name][split_name][f"split_{split_id}"][
                                    f"fold_{fold_id}"
                                ][group] = fold_indices

                        # Combine group splits into final indices for each fold
                        self._combine_group_splits(
                            split_datasets[dset_name][split_name][f"split_{split_id}"],
                            data,
                        )

                        pbar.update(1)

        return split_datasets

    def _combine_group_splits(self, split_folds: Dict, data: pd.DataFrame) -> None:
        """Merge per‑group indices into overall fold indices.

        Parameters
        ----------
        split_folds : dict
            Fold mapping generated by ``_split_group_stratified`` calls.
        data : pandas.DataFrame
            Original dataset for validation / grouping.

        Raises
        ------
        ValueError
            If an expected fold key is missing in ``split_folds``.
        """
        for fold_id in range(self.n_folds):
            fold_key = f"fold_{fold_id}"
            if fold_key not in split_folds:
                raise ValueError(f"Fold {fold_id} not found in split")

            combined_train_idx = []
            combined_val_idx = []
            combined_test_idx = []

            for group in data[self.stratify_column].unique():
                if group not in split_folds[fold_key]:
                    continue

                group_split = split_folds[fold_key][group]
                combined_train_idx.extend(group_split["train"])
                combined_val_idx.extend(group_split["validation"])
                combined_test_idx.extend(group_split["test"])

            split_folds[fold_key]["total"] = {
                "train": np.array(combined_train_idx),
                "validation": np.array(combined_val_idx),
                "test": np.array(combined_test_idx),
            }

    def save_splits_as_huggingface(
        self,
        split_structure: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame],
        output_dir: str | Path,
    ) -> None:
        """Persist generated splits as Hugging Face ``DatasetDict`` objects.

        Parameters
        ----------
        split_structure : dict
            Nested mapping produced by ``create_splits``.
        datasets : dict[str, pandas.DataFrame]
            Original datasets keyed by name.
        output_dir : str | Path
            Root directory under which split folders are created.
        """
        # Use module-level Path import

        output_dir = Path(output_dir)

        for dset_name, splits in split_structure.items():
            for split_name, split_data in splits.items():
                for split_number, folds in split_data.items():
                    for fold_number, datasets_dict in folds.items():
                        if "total" not in datasets_dict:
                            continue

                        split_output_dir = (
                            output_dir
                            / f"{dset_name}_quality/{split_name}/{split_number}/{fold_number}/hf_dataset"
                        )
                        split_output_dir.mkdir(parents=True, exist_ok=True)

                        train_idx = datasets_dict["total"]["train"]
                        val_idx = datasets_dict["total"]["validation"]
                        test_idx = datasets_dict["total"]["test"]

                        data = datasets[dset_name]

                        # Convert to Hugging Face datasets
                        train_hf = Dataset.from_pandas(data.loc[train_idx], preserve_index=False)
                        val_hf = Dataset.from_pandas(data.loc[val_idx], preserve_index=False)
                        test_hf = Dataset.from_pandas(data.loc[test_idx], preserve_index=False)
                        dset = DatasetDict({"train": train_hf, "validation": val_hf, "test": test_hf})

                        dset.save_to_disk(str(split_output_dir))
                        logger.debug("Saved HF dataset to %s", split_output_dir)

                # Print folder size
                folder_size = sum(
                    f.stat().st_size for f in split_output_dir.parent.glob("**/*") if f.is_file()
                )
                folder_size_mb = folder_size / (1024 * 1024)
                logger.info("Saved splits for %s, %s: %.2f MB", dset_name, split_name, folder_size_mb)

    def create_and_save_splits(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_dir: str | Path,
        visualizer=None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convenience wrapper to generate and persist splits in one pass.

        Parameters
        ----------
        datasets : dict[str, pandas.DataFrame]
            Dataset mapping used as input.
        output_dir : Path | str
            Directory root for output structure.
        visualizer : DatasetVisualizer, optional
            If provided, its ``visualize_fold_immediately`` method is invoked
            per fold for quick exploratory plots.
        overwrite : bool, optional
            Overwrite existing ``*_quality`` directories if present.

        Returns
        -------
        dict
            Nested mapping identical to ``create_splits`` output.

        Raises
        ------
        ValueError
            If fold combination encounters missing keys.
        """
        # Use module-level Path import

        output_dir = Path(output_dir)
        split_datasets: Dict[str, Any] = {}

        # Check if splits already exist
        n_iter = len(datasets) * len(self.split_methods) * self.n_splits

        logger.info("Creating %d total dataset splits", n_iter)

        with tqdm(total=n_iter, desc="Creating and saving dataset splits") as pbar:
            for dset_name, data in datasets.items():
                split_datasets[dset_name] = {}

                for split_name, split_method in self.split_methods.items():
                    logger.info("Processing dataset: %s, split: %s", dset_name, split_name)
                    split_datasets[dset_name][split_name] = {}

                    for split_id in range(self.n_splits):
                        split_datasets[dset_name][split_name][f"split_{split_id}"] = {}

                        split_output_dir = output_dir / f"{dset_name}_quality/{split_name}/split_{split_id}"
                        if not overwrite and split_output_dir.exists():
                            logger.info(
                                "Skipping existing split: %s/%s/split_%s",
                                dset_name,
                                split_name,
                                split_id,
                            )
                            pbar.update(len(data[self.stratify_column].unique()))
                            continue
                        elif overwrite and split_output_dir.exists():
                            logger.warning(
                                "Overwriting existing split: %s/%s/split_%s",
                                dset_name,
                                split_name,
                                split_id,
                            )
                        else:
                            logger.debug(
                                "Creating new split: %s/%s/split_%s",
                                dset_name,
                                split_name,
                                split_id,
                            )

                        for group in data[self.stratify_column].unique():
                            subdata = data[data[self.stratify_column] == group]
                            group_folds = self._split_group_stratified(subdata, split_method, split_id)

                            # Store group splits
                            for fold_id, fold_indices in group_folds.items():
                                if (
                                    f"fold_{fold_id}"
                                    not in split_datasets[dset_name][split_name][f"split_{split_id}"]
                                ):
                                    split_datasets[dset_name][split_name][f"split_{split_id}"][
                                        f"fold_{fold_id}"
                                    ] = {}

                                split_datasets[dset_name][split_name][f"split_{split_id}"][
                                    f"fold_{fold_id}"
                                ][group] = fold_indices

                        # Combine group splits into final indices for each fold
                        self._combine_group_splits(
                            split_datasets[dset_name][split_name][f"split_{split_id}"],
                            data,
                        )

                        # Save splits immediately after creation
                        self._save_splits_for_split_id(
                            dset_name,
                            split_name,
                            f"split_{split_id}",
                            split_datasets[dset_name][split_name][f"split_{split_id}"],
                            data,
                            output_dir,
                            visualizer=visualizer,
                            overwrite=overwrite,
                        )

                        pbar.update(1)

        return split_datasets

    def _save_splits_for_split_id(
        self,
        dset_name: str,
        split_name: str,
        split_id_key: str,
        split_folds: Dict,
        data: pd.DataFrame,
        output_dir,
        visualizer=None,
        overwrite=False,
    ) -> None:
        """Persist all folds belonging to a single split identifier.

        Parameters
        ----------
        dset_name : str
            Name of the dataset.
        split_name : str
            Clustering / split method label.
        split_id_key : str
            Outer split identifier (``'split_i'``).
        split_folds : dict
            Mapping fold -> group -> indices (with ``total`` aggregation).
        data : pandas.DataFrame
            Source dataset for row extraction.
        output_dir : Path | str
            Root output directory.
        visualizer : DatasetVisualizer, optional
            If provided, produces per‑fold visualisations.
        overwrite : bool, optional
            Overwrite existing split directories if present. Default is False.
        """
        # Use module-level Path import

        output_dir = Path(output_dir)

        for fold_id_key, datasets_dict in split_folds.items():
            if "total" not in datasets_dict:
                continue

            split_output_dir = (
                output_dir / f"{dset_name}_quality/{split_name}/{split_id_key}/{fold_id_key}/hf_dataset"
            )
            if not overwrite and split_output_dir.exists():
                logger.info(
                    "Skipping existing split: %s/%s/%s/%s",
                    dset_name,
                    split_name,
                    split_id_key,
                    fold_id_key,
                )
                return
            elif overwrite and split_output_dir.exists():
                logger.warning(
                    "Overwriting existing split: %s/%s/%s/%s",
                    dset_name,
                    split_name,
                    split_id_key,
                    fold_id_key,
                )
            else:
                split_output_dir.mkdir(parents=True)

            train_idx = datasets_dict["total"]["train"]
            val_idx = datasets_dict["total"]["validation"]
            test_idx = datasets_dict["total"]["test"]

            # Convert to Hugging Face datasets
            train_hf = Dataset.from_pandas(data.loc[train_idx], preserve_index=False)
            val_hf = Dataset.from_pandas(data.loc[val_idx], preserve_index=False)
            test_hf = Dataset.from_pandas(data.loc[test_idx], preserve_index=False)
            dset = DatasetDict({"train": train_hf, "validation": val_hf, "test": test_hf})

            dset.save_to_disk(str(split_output_dir))

            # Log each saved component
            logger.debug("Saved train set (%d samples) to %s/train", len(train_idx), split_output_dir)
            logger.debug(
                "Saved validation set (%d samples) to %s/validation", len(val_idx), split_output_dir
            )
            logger.debug("Saved test set (%d samples) to %s/test", len(test_idx), split_output_dir)
            logger.info(
                "Saved %s (%d|%d|%d train|val|test) for %s_%s_%s",
                fold_id_key,
                len(train_idx),
                len(val_idx),
                len(test_idx),
                dset_name,
                split_name,
                split_id_key,
            )

            # Create visualizations immediately after saving
            if visualizer is not None:
                viz_output_dir = Path(output_dir) / "figures" / "fold_visualizations"
                visualizer.visualize_fold_immediately(
                    dset_name,
                    split_name,
                    split_id_key.split("_")[1],  # Extract number from "split_0"
                    fold_id_key.split("_")[1],  # Extract number from "fold_0"
                    datasets_dict,
                    data,
                    viz_output_dir,
                )
