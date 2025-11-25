"""End-to-end dataset split pipeline with visuals (fingerprints optional).

This module provides the ``DatasetSplitPipeline`` class which loads cleaned
datasets, creates temporal and stratified splits, persists HuggingFace
``DatasetDict`` objects, and generates coverage / size distribution
visualizations via ``DatasetVisualizer``. Fingerprints are no longer
materialized during splitting to keep disk usage low; models generate
fingerprints on-the-fly during training/evaluation.

Contents
--------
Classes
    DatasetSplitPipeline : Orchestrates loading, fingerprinting, splitting & plotting.

Functions
    setup_logging : Lightweight logging configuration helper for CLI usage.

Typical Workflow
----------------
>>> pipeline = DatasetSplitPipeline(base_data_dir)
>>> pipeline.run(create_temporal=True, create_stratified=True)

Notes
-----
The class stores intermediate datasets in-memory; large datasets may demand
substantial RAM. Fingerprints are inserted near the front of the column list
to maintain readability of target endpoints.
"""

from pathlib import Path
import logging
from datetime import datetime

import pandas as pd

from admet.data.splitter import DatasetSplitter
from admet.visualize.dataset_viz import DatasetVisualizer

logger = logging.getLogger(__name__)


class DatasetSplitPipeline:
    """
    End-to-end pipeline for dataset splitting and analysis.

    This class coordinates loading data, calculating fingerprints, creating
    stratified splits, and generating visualizations.

    Parameters
    ----------
    base_data_dir : Path or str
        Root directory containing cleaned dataset CSV files.
    output_dir : Path or str, optional
        Root directory for all outputs. Defaults to parent of base_data_dir.
    """

    # Dataset quality levels and their corresponding file names
    DATASET_CONFIGS = {
        "high": "cleaned_combined_datasets_high_quality.csv",
        "medium": "cleaned_combined_datasets_medium_high_quality.csv",
        "low": "cleaned_combined_datasets_low_medium_high_quality.csv",
    }

    # Temporal split configuration
    TEMPORAL_SPLIT_CONFIG = {
        "quality": "high",
        "train_percentage": 0.9,
        "validation_percentage": 0.1,
        "sort_column": "Molecule Name",
        "random_state": 42,
    }

    def __init__(
        self,
        base_data_dir,
        output_dir=None,
        overwrite=False,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        base_data_dir : Path or str
            Root directory containing cleaned dataset CSV files.
        output_dir : Path or str, optional
            Root directory for all outputs. Defaults to parent of base_data_dir.
        overwrite : bool, optional
            Whether to overwrite existing files (default: False).
        """
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_data_dir.parent.parent / "splits"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.overwrite = overwrite

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.figure_dir = self.output_dir / f"figures/{timestamp}"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {self.base_data_dir}")

        logger.info("Output directory set to %s", self.output_dir)
        logger.info("Input data directory found at %s", self.base_data_dir)
        logger.info("Figure directory set to %s", self.figure_dir)
        logger.info("Overwrite mode: %s", self.overwrite)

        self.splitter = DatasetSplitter()
        self.visualizer = DatasetVisualizer()

        self.datasets = {}
        self.split_structure = {}

    def load_datasets(self) -> None:
        """Load all configured datasets from CSV files.

        Raises
        ------
        ValueError
            If no datasets are successfully loaded.
        FileNotFoundError
            If the base data directory is missing.
        """
        logger.info("Loading datasets...")

        for quality, filename in self.DATASET_CONFIGS.items():
            filepath = self.base_data_dir / filename
            if not filepath.exists():
                logger.warning(f"Dataset not found: {filepath}")
                continue

            logger.info(f"Loading {quality} quality dataset from {filepath}")
            self.datasets[quality] = pd.read_csv(filepath, low_memory=False if quality != "high" else True)

            df = self.datasets[quality]
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Columns: {df.columns.tolist()}")
            logger.info(f"  Unique sources: {df['Dataset'].unique()}")

        if not self.datasets:
            raise ValueError("No datasets loaded successfully")

    def add_fingerprints(self) -> None:
        """Calculate and add Morgan fingerprints to all loaded datasets."""
        from admet.data.fingerprinting import MorganFingerprintGenerator

        logger.info("Calculating fingerprints for all datasets...")

        for name, df in self.datasets.items():
            logger.info(f"Adding fingerprints to {name} quality dataset")
            self.datasets[name] = self.fp_generator.add_fingerprints_to_dataframe(
                df, smiles_column="SMILES", insertion_index=3
            )

    def create_temporal_split(self) -> None:
        """Create and save a temporal split for high-quality dataset with visualizations.

        The temporal split is created by sorting on a stable column and slicing
        into train / validation / test according to configured percentages.
        Saves HuggingFace datasets immediately and generates endpoint coverage
        for the temporal grouping.
        """
        logger.info("Creating and saving temporal split for high-quality dataset...")

        config = self.TEMPORAL_SPLIT_CONFIG
        quality = config["quality"]

        if quality not in self.datasets:
            logger.warning("Cannot create temporal split: %s quality dataset not loaded", quality)
            return

        # Check if temporal split already exists
        temporal_dir = self.output_dir / "high_quality/temporal_split/hf_dataset"
        if temporal_dir.exists() and not self.overwrite:
            logger.warning(
                "Temporal split already exists at %s. Skipping (use --overwrite to replace)",
                temporal_dir,
            )
            return

        data = self.datasets[quality].sort_values(by=config["sort_column"]).reset_index(drop=True)
        n_total = len(data)
        n_train = int(n_total * config["train_percentage"])

        train_df = data.iloc[:n_train]
        test_df = data.iloc[n_train:]

        # Split train into train/validation
        train_df = train_df.sample(frac=1, random_state=config["random_state"]).reset_index(drop=True)
        n_validation = int(train_df.shape[0] * config["validation_percentage"])
        validation_df = train_df.iloc[:n_validation]
        train_df = train_df.iloc[n_validation:]

        logger.info("  Total samples: %d", n_total)
        logger.info("  Training samples: %d", train_df.shape[0])
        logger.info("  Validation samples: %d", validation_df.shape[0])
        logger.info("  Testing samples: %d", test_df.shape[0])

        # Save temporal split immediately
        from datasets import Dataset, DatasetDict

        temporal_dir.mkdir(parents=True, exist_ok=True)

        train_hf = Dataset.from_pandas(train_df, preserve_index=False)
        validation_hf = Dataset.from_pandas(validation_df, preserve_index=False)
        test_hf = Dataset.from_pandas(test_df, preserve_index=False)
        temporal_hf = DatasetDict({"train": train_hf, "validation": validation_hf, "test": test_hf})
        temporal_hf.save_to_disk(str(temporal_dir))

        logger.debug("Saved temporal split train set to %s/train", temporal_dir)
        logger.debug("Saved temporal split validation set to %s/validation", temporal_dir)
        logger.debug("Saved temporal split test set to %s/test", temporal_dir)
        logger.info("Temporal split saved to %s", temporal_dir.parent)

        # Generate visualizations for temporal split
        logger.info("Generating visualizations for temporal split...")
        temporal_viz_dir = self.figure_dir / "temporal_split"
        temporal_viz_dir.mkdir(parents=True, exist_ok=True)

        # Plot endpoint coverage
        coverage_path = temporal_viz_dir / "temporal_endpoint_coverage.png"
        self.visualizer.plot_endpoint_coverage(train_df, validation_df, test_df, coverage_path)

        logger.info(f"Temporal split visualizations saved to {temporal_viz_dir}")

    def create_stratified_splits_with_visualization(self) -> None:
        """Create and save stratified k-fold splits with concurrent visualization.

        Visualizations are generated alongside split creation rather than as a
        separate post-processing step.
        """
        logger.info("Creating stratified k-fold splits with concurrent visualization...")

        self.split_structure = self.splitter.create_and_save_splits(
            self.datasets,
            self.output_dir,
            visualizer=self.visualizer,
            overwrite=self.overwrite,
        )

        # Generate aggregate visualizations across all splits
        logger.info("Generating aggregate visualizations...")
        self.visualizer.visualize_all_splits(self.split_structure, self.datasets, self.figure_dir)

        logger.info(
            f"Created and saved {len(self.split_structure)} quality levels of splits with visualizations"
        )

    def run(self, create_temporal: bool = True, create_stratified: bool = True) -> None:
        """
        Execute the full pipeline.

        Parameters
        ----------
        create_temporal : bool, optional
            Whether to create temporal split (default: True).
        create_stratified : bool, optional
            Whether to create stratified splits (default: True).
        """
        logger.info("Starting dataset split pipeline...")

        self.load_datasets()

        if create_temporal:
            self.create_temporal_split()

        if create_stratified:
            self.create_stratified_splits_with_visualization()

        logger.info("Pipeline completed successfully!")


def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Parameters
    ----------
    level : int, optional
        Logging level (default: DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)

    return logger


__all__ = ["DatasetSplitPipeline", "setup_logging"]
