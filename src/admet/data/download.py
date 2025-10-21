"""data download helpers

Small helper to download a Hugging Face dataset and save split(s) to CSV.
"""

from typing import Optional, Union
from pathlib import Path
import logging

import pandas as pd


from admet.data.constants import DATASETS


logger = logging.getLogger(__name__)


def download_hf_dataset_to_csv(
    dataset_uri: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    """Download a Hugging Face dataset and save it to a CSV file.

    Args:
        dataset_uri (Union[str, Path]): The URI of the Hugging Face dataset.
        output_file (Union[str, Path]): The path to the output CSV file.
    """
    logger.debug(f"Loading dataset from {dataset_uri}...")
    df = pd.read_csv(dataset_uri)
    logger.info(f"Saving dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.debug("Download and save complete.")


class Downloader:
    """Dataset downloader class.

    This class can be extended in the future to support more complex
    downloading logic, such as handling multiple files, authentication,
    etc.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the DatasetDownloader.

        Args:
            logger (Optional[logging.Logger]): Logger instance. If None,
                a default logger will be created.
        """
        self.logger = logger or logging.getLogger(__name__)

    def download(
        self,
        dataset_type: str,
        dataset_uri: Union[str, Path],
        output_file: Union[str, Path],
    ) -> None:
        """Download a dataset and save it to a CSV file.

        Args:
            dataset_type (str): The type of the dataset (e.g., 'huggingface').
            dataset_uri (Union[str, Path]): The URI of the dataset.
            output_file (Union[str, Path]): The path to the output CSV file.

        Raises:
            ValueError: If the dataset type is unsupported.
        """
        dataset_type = dataset_type.lower()

        if dataset_type == "huggingface":
            download_hf_dataset_to_csv(dataset_uri, output_file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def download_all(
        self,
    ) -> None:
        """Download all datasets defined in DATASETS constant."""
        for dataset_name, dataset_info in DATASETS.items():
            dataset_type = dataset_info.get("type")
            dataset_uri = dataset_info.get("uri")
            output_file = dataset_info.get("output_file")

            if dataset_type and dataset_uri and output_file:
                self.logger.info(f"Downloading dataset: {dataset_name}")
                self.download(dataset_type, dataset_uri, output_file)
            else:
                self.logger.warning(f"Dataset info incomplete for {dataset_name}, skipping.")
