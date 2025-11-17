"""
Dataset splitting utilities for train/validation/test stratification.

This module provides classes for creating stratified k-fold splits across
multiple datasets, with support for various clustering algorithms to ensure
chemical diversity in splits.
"""

from typing import Dict, List, Callable, Tuple, Any
import logging
import gc

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import useful_rdkit_utils as uru
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Stratified dataset splitter for creating train/validation/test splits.

    This class handles creating stratified k-fold splits across multiple
    datasets with support for grouping by data source and various clustering
    algorithms to maintain chemical diversity.

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

    # Default clustering methods
    DEFAULT_SPLIT_METHODS: Dict[str, Callable] = {
        "random_cluster": uru.get_random_clusters,
        "scaffold_cluster": uru.get_bemis_murcko_clusters,
        "kmeans_cluster": uru.get_kmeans_clusters,
        "umap_cluster": uru.get_umap_clusters,
        "butina_cluster": uru.get_butina_clusters,
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
            f"Initialized DatasetSplitter: n_splits={n_splits}, n_folds={n_folds}, "
            f"test_pct={test_percentage}, val_pct={validation_percentage}"
        )

    def add_split_method(self, name: str, method: Callable) -> None:
        """
        Register a custom splitting method.

        Parameters
        ----------
        name : str
            Name to identify the splitting method.
        method : callable
            Function that takes SMILES series and returns cluster assignments.
        """
        self.split_methods[name] = method
        logger.debug(f"Added custom split method: {name}")

    def _split_group_stratified(
        self,
        subdata: pd.DataFrame,
        split_method: Callable,
        random_state: int,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Perform stratified k-fold split on a single group of data.

        Parameters
        ----------
        subdata : pd.DataFrame
            Subset of data for a single group (e.g., single dataset source).
        split_method : callable
            Function to generate cluster assignments.
        random_state : int
            Random state for reproducibility.

        Returns
        -------
        dict
            Nested dictionary with fold index -> {"train", "validation", "test"} indices.
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

    def create_splits(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]]]:
        """
        Create stratified splits for all datasets and split methods.

        Parameters
        ----------
        datasets : dict
            Dictionary mapping dataset names to dataframes.

        Returns
        -------
        dict
            Nested dictionary: dataset -> split_method -> split_id -> fold_id -> group -> indices.
        """
        split_datasets: Dict[str, Any] = {}
        n_iter = len(datasets) * len(self.split_methods) * self.n_splits

        logger.info(f"Creating {n_iter} total dataset splits")

        with tqdm(total=n_iter, desc="Creating dataset splits") as pbar:
            for dset_name, data in datasets.items():
                split_datasets[dset_name] = {}

                for split_name, split_method in self.split_methods.items():
                    logger.info(f"Processing dataset: {dset_name}, split: {split_name}")
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
        """
        Combine per-group splits into final train/validation/test indices.

        Parameters
        ----------
        split_folds : dict
            Dictionary of fold splits to combine.
        data : pd.DataFrame
            Original dataframe for validation.
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
        split_structure: Dict,
        datasets: Dict[str, pd.DataFrame],
        output_dir: str,
    ) -> None:
        """
        Save splits as Hugging Face datasets to disk.

        Parameters
        ----------
        split_structure : dict
            Nested split dictionary from create_splits.
        datasets : dict
            Dictionary of original dataframes.
        output_dir : str or Path
            Root output directory for saving splits.
        """
        from pathlib import Path

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
                        logger.debug(f"Saved HF dataset to {split_output_dir}")

                # Print folder size
                folder_size = sum(
                    f.stat().st_size for f in split_output_dir.parent.glob("**/*") if f.is_file()
                )
                folder_size_mb = folder_size / (1024 * 1024)
                logger.info(f"Saved splits for {dset_name}, {split_name}: {folder_size_mb:.2f} MB")

    def create_and_save_splits(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_dir,
        visualizer=None,
        overwrite=False,
    ) -> Dict:
        """
        Create stratified splits and save them immediately to disk.

        Optionally creates visualizations for each fold immediately after creation.

        Parameters
        ----------
        datasets : dict
            Dictionary mapping dataset names to dataframes.
        output_dir : Path or str
            Root output directory for saving splits.
        visualizer : DatasetVisualizer, optional
            Visualizer instance for creating plots immediately. If None, no plots created.
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).

        Returns
        -------
        dict
            Nested dictionary: dataset -> split_method -> split_id -> fold_id -> group -> indices.
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        split_datasets: Dict[str, Any] = {}

        # Check if splits already exist
        if not overwrite:
            quality_dirs = [d for d in output_dir.glob("*_quality") if d.is_dir()]
            if quality_dirs:
                logger.warning(
                    "Stratified splits already exist at %s. Skipping (use --overwrite to replace)",
                    output_dir,
                )
                return split_datasets
        n_iter = len(datasets) * len(self.split_methods) * self.n_splits

        logger.info(f"Creating {n_iter} total dataset splits")

        with tqdm(total=n_iter, desc="Creating and saving dataset splits") as pbar:
            for dset_name, data in datasets.items():
                split_datasets[dset_name] = {}

                for split_name, split_method in self.split_methods.items():
                    logger.info(f"Processing dataset: {dset_name}, split: {split_name}")
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

                        # Save splits immediately after creation
                        self._save_splits_for_split_id(
                            dset_name,
                            split_name,
                            f"split_{split_id}",
                            split_datasets[dset_name][split_name][f"split_{split_id}"],
                            data,
                            output_dir,
                            visualizer,
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
    ) -> None:
        """
        Save all folds for a specific split immediately.

        Optionally creates visualizations for each fold immediately after saving.

        Parameters
        ----------
        dset_name : str
            Dataset name.
        split_name : str
            Split method name.
        split_id_key : str
            Split ID key (e.g., "split_0").
        split_folds : dict
            Dictionary of fold splits.
        data : pd.DataFrame
            Original dataframe.
        output_dir : Path or str
            Root output directory.
        visualizer : DatasetVisualizer, optional
            Visualizer for creating plots. If None, no plots created.
        """
        from pathlib import Path

        output_dir = Path(output_dir)

        for fold_id_key, datasets_dict in split_folds.items():
            if "total" not in datasets_dict:
                continue

            split_output_dir = (
                output_dir / f"{dset_name}_quality/{split_name}/{split_id_key}/{fold_id_key}/hf_dataset"
            )
            split_output_dir.mkdir(parents=True, exist_ok=True)

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
            logger.debug(f"Saved train set ({len(train_idx)} samples) to " f"{split_output_dir}/train")
            logger.debug(
                f"Saved validation set ({len(val_idx)} samples) to " f"{split_output_dir}/validation"
            )
            logger.debug(f"Saved test set ({len(test_idx)} samples) to " f"{split_output_dir}/test")
            logger.info(
                f"Saved {fold_id_key} ({len(train_idx)}|{len(val_idx)}|{len(test_idx)} "
                f"train|val|test) for {dset_name}_{split_name}_{split_id_key}"
            )

            # Create visualizations immediately after saving
            if visualizer is not None:
                from pathlib import Path as PathlibPath

                viz_output_dir = PathlibPath(output_dir) / "figures" / "fold_visualizations"
                visualizer.visualize_fold_immediately(
                    dset_name,
                    split_name,
                    split_id_key.split("_")[1],  # Extract number from "split_0"
                    fold_id_key.split("_")[1],  # Extract number from "fold_0"
                    datasets_dict,
                    data,
                    viz_output_dir,
                )
