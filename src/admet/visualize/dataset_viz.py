"""
Visualization utilities for dataset split analysis.

This module provides classes and functions for visualizing the characteristics
and distributions of dataset splits, including endpoint coverage and split sizes.
"""

from typing import Dict, List
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Default ADMET endpoints
DEFAULT_ENDPOINTS: List[str] = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]


class DatasetVisualizer:
    """
    Visualize characteristics of dataset splits.

    This class provides methods to create various plots for analyzing
    the properties and distributions of train/validation/test splits.

    Parameters
    ----------
    dpi : int, optional
        Resolution for saved figures (default: 600).
    endpoints : list of str, optional
        ADMET endpoints to analyze (default: DEFAULT_ENDPOINTS).
    """

    def __init__(
        self,
        dpi: int = 600,
        endpoints: List[str] = None,
    ) -> None:
        """
        Initialize the dataset visualizer.

        Parameters
        ----------
        dpi : int, optional
            Resolution for saved figures (default: 600).
        endpoints : list of str, optional
            ADMET endpoints to analyze (default: DEFAULT_ENDPOINTS).
        """
        self.dpi = dpi
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        logger.debug(f"Initialized DatasetVisualizer with {len(self.endpoints)} endpoints")

    def plot_endpoint_coverage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """
        Plot the number of non-null samples for each endpoint across splits.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training set dataframe.
        val_df : pd.DataFrame
            Validation set dataframe.
        test_df : pd.DataFrame
            Test set dataframe.
        output_path : Path or str
            Path to save the figure.
        """
        output_path = Path(output_path)

        # Count non-null values for each endpoint
        counts = {
            "train": [train_df[ep].notnull().sum() for ep in self.endpoints],
            "validation": [val_df[ep].notnull().sum() for ep in self.endpoints],
            "test": [test_df[ep].notnull().sum() for ep in self.endpoints],
        }

        counts_df = pd.DataFrame(counts, index=self.endpoints)

        # Create plot
        ax = counts_df.plot.bar(rot=45, figsize=(10, 6))
        ax.set_title("Sample Counts per Endpoint")
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Endpoint")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

        logger.debug(f"Saved endpoint coverage plot to {output_path}")
        logger.info(f"Saved endpoint coverage plot: {output_path.name}")

    def plot_test_set_distribution(
        self,
        split_structure: Dict,
        dataset_name: str,
        output_path: Path,
    ) -> None:
        """
        Create boxplot of test set sizes across split methods.

        Parameters
        ----------
        split_structure : dict
            Nested split dictionary from DatasetSplitter.create_splits.
        dataset_name : str
            Name of the dataset to visualize.
        output_path : Path or str
            Path to save the figure.
        """
        output_path = Path(output_path)

        plot_data = []
        splits = split_structure[dataset_name]

        for split_name, split_data in splits.items():
            for split_id, folds in split_data.items():
                for fold_id, datasets_dict in folds.items():
                    if "total" not in datasets_dict:
                        continue
                    n_test_samples = len(datasets_dict["total"]["test"])
                    plot_data.append({"Split Method": split_name, "Number of Test Samples": n_test_samples})

        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Split Method", y="Number of Test Samples", data=plot_df, ax=ax)
        ax.set_title(f"Distribution of Test Set Sizes for Different Split Methods\nDataset: {dataset_name}")
        ax.set_ylabel("Number of Test Samples")
        ax.set_xlabel("Split Method")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close()

        logger.debug(f"Saved test set distribution plot to {output_path}")
        logger.info(f"Saved test set distribution plot: {output_path.name}")

    def plot_split_size_distribution(
        self,
        split_structure: Dict,
        dataset_name: str,
        split_method: str,
        output_path: Path,
    ) -> None:
        """
        Create boxplots of train/test set sizes by data source.

        Parameters
        ----------
        split_structure : dict
            Nested split dictionary from DatasetSplitter.create_splits.
        dataset_name : str
            Name of the dataset to visualize.
        split_method : str
            Name of the split method to visualize.
        output_path : Path or str
            Path to save the figure.
        """
        output_path = Path(output_path)

        fold_sizes = []
        split_data = split_structure[dataset_name][split_method]

        for split_id, folds in split_data.items():
            for fold_id, groups in folds.items():
                for group_name, datasets in groups.items():
                    if group_name == "total":
                        continue

                    train_size = len(datasets["train"])
                    test_size = len(datasets["test"])
                    fold_sizes.append(
                        {
                            "Split ID": split_id,
                            "Fold ID": fold_id,
                            "Group": group_name,
                            "Train Size": train_size,
                            "Test Size": test_size,
                        }
                    )

        fold_sizes_df = pd.DataFrame(fold_sizes)

        # Create figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        sns.boxplot(x="Group", y="Train Size", data=fold_sizes_df, ax=axs[0])
        axs[0].set_title(
            f"Train Set Size Distribution: {dataset_name.capitalize()} Quality, "
            f"{split_method.replace('_', ' ').capitalize()} Split"
        )

        sns.boxplot(x="Group", y="Test Size", data=fold_sizes_df, ax=axs[1])
        axs[1].set_title(
            f"Test Set Size Distribution: {dataset_name.capitalize()} Quality, "
            f"{split_method.replace('_', ' ').capitalize()} Split"
        )

        for ax in axs:
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel("Number of Data Points")
            ax.set_xlabel("Provenance")

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close()

        logger.debug(f"Saved split size distribution plot to {output_path}")
        logger.info(f"Saved split size distribution plot: {output_path.name}")

    def visualize_all_splits(
        self,
        split_structure: Dict,
        datasets: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> None:
        """
        Generate all visualizations for split analysis.

        Creates endpoint coverage plots and size distribution plots for all
        splits and saves them to the specified output directory.

        Parameters
        ----------
        split_structure : dict
            Nested split dictionary from DatasetSplitter.create_splits.
        datasets : dict
            Dictionary of original dataframes.
        output_dir : Path or str
            Root output directory for saving figures.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating visualizations in {output_dir}")

        for dset_name, splits in split_structure.items():
            logger.info(f"Creating visualizations for dataset: {dset_name}")

            # Plot test set size distributions
            test_dist_path = output_dir / f"{dset_name}_test_set_size_distribution.png"
            self.plot_test_set_distribution(split_structure, dset_name, test_dist_path)

            # Plot endpoint coverage and split size distributions for each method
            for split_name, split_data in splits.items():
                split_size_path = (
                    output_dir
                    / f"{dset_name}_quality_{split_name}_split_train_test_size_distribution_boxplot.png"
                )
                self.plot_split_size_distribution(split_structure, dset_name, split_name, split_size_path)

            # Plot endpoint coverage for first split/fold as representative
            for split_name, split_data in splits.items():
                first_split = split_data["split_0"]
                first_fold = first_split["fold_0"]

                if "total" not in first_fold:
                    continue

                train_idx = first_fold["total"]["train"]
                val_idx = first_fold["total"]["validation"]
                test_idx = first_fold["total"]["test"]

                data = datasets[dset_name]
                train_df = data.loc[train_idx]
                val_df = data.loc[val_idx]
                test_df = data.loc[test_idx]

                endpoint_path = output_dir / f"{dset_name}_{split_name}_split_0_fold_0_endpoint_coverage.png"
                self.plot_endpoint_coverage(train_df, val_df, test_df, endpoint_path)

    def visualize_fold_immediately(
        self,
        dset_name: str,
        split_name: str,
        split_id: str,
        fold_id: str,
        fold_dict: Dict,
        data: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Create and save visualizations for a fold immediately after creation.

        Creates endpoint coverage plot for the fold's train/val/test split.

        Parameters
        ----------
        dset_name : str
            Dataset quality name (e.g., "high", "medium", "low").
        split_name : str
            Splitting method name (e.g., "random_cluster", "scaffold_cluster").
        split_id : str
            Split identifier (e.g., "split_0").
        fold_id : str
            Fold identifier (e.g., "fold_0").
        fold_dict : dict
            Dictionary with "total" key containing train/val/test indices.
        data : pd.DataFrame
            Original dataframe.
        output_dir : Path or str
            Root output directory for saving figures.
        """
        if "total" not in fold_dict:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_idx = fold_dict["total"]["train"]
        val_idx = fold_dict["total"]["validation"]
        test_idx = fold_dict["total"]["test"]

        train_df = data.loc[train_idx]
        val_df = data.loc[val_idx]
        test_df = data.loc[test_idx]

        # Create endpoint coverage plot for this fold
        endpoint_path = output_dir / f"{dset_name}_{split_name}_{split_id}_{fold_id}_endpoint_coverage.png"
        self.plot_endpoint_coverage(train_df, val_df, test_df, endpoint_path)

        logger.debug(
            f"Visualized {dset_name}_{split_name}_{split_id}_{fold_id}: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
