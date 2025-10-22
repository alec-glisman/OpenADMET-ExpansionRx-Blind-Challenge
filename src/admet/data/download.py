"""data download helpers

Small helper to download a Hugging Face dataset and save split(s) to CSV.
"""

from typing import Optional, Union
from pathlib import Path
import logging

import pandas as pd
import polaris as po
from tqdm import tqdm
from tdc.benchmark_group import admet_group

from admet.data.constants import DATASETS, DEFAULT_DATASET_DIR


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


def download_polaris_dataset_to_csv(
    dataset_uri: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    """Download a Polaris dataset and save it to a CSV file.

    Args:
        dataset_uri (Union[str, Path]): The identifier of the Polaris dataset
            (e.g., 'asap-discovery/antiviral-admet-2025-unblinded').
        output_file (Union[str, Path]): The path to the output CSV file.
    """
    logger.debug(f"Loading Polaris dataset: {dataset_uri}...")
    dataset = po.load_dataset(str(dataset_uri))
    logger.info(f"Dataset loaded with size {dataset.size()}")
    print(f"Dataset loaded with size {dataset.size()}")

    logger.info("Converting Polaris dataset to DataFrame...")

    # Convert Polaris dataset to pandas DataFrame with tqdm progress bar
    data = []
    for i in tqdm(range(dataset.size()[0]), desc="Converting rows", unit="row"):
        row = dataset[i]
        data.append(row)
    df = pd.DataFrame(data)

    logger.info(f"Saving dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.debug("Download and save complete.")


def download_tdc_dataset_to_csv(
    dataset_uri: str,
    output_file: Union[str, Path],
) -> None:
    """Download a TDC ADMET_Group dataset and save it to a CSV file.

    Args:
        dataset_uri (str): TDC benchmark name (e.g., 'Caco2_Wang').
        output_file (Union[str, Path]): The path to the output CSV file.
    """
    logger.debug(f"Loading TDC benchmark: {dataset_uri}...")
    group = admet_group(path="/tmp/tdc_download/")
    benchmark = group.get(dataset_uri)
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]

    logger.info(f"Converting TDC benchmark '{name}' to DataFrame...")
    # Concatenate train/val and test splits
    import pandas as pd

    df = pd.concat([pd.DataFrame(train_val), pd.DataFrame(test)], ignore_index=True)
    # tqdm progress bar for demonstration (not needed for concat, but for row-wise ops)
    # for _ in tqdm(df.iterrows(), total=len(df), desc="Converting rows", unit="row"):
    #     pass
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
            dataset_type (str): The type of the dataset
                (e.g., 'huggingface', 'polaris').
            dataset_uri (Union[str, Path]): The URI of the dataset.
            output_file (Union[str, Path]): The path to the output CSV file.

        Raises:
            ValueError: If the dataset type is unsupported.
        """
        dataset_type = dataset_type.lower()

        if dataset_type == "huggingface":
            download_hf_dataset_to_csv(dataset_uri, output_file)
        elif dataset_type == "polaris":
            download_polaris_dataset_to_csv(dataset_uri, output_file)
        elif dataset_type == "tdc":
            download_tdc_dataset_to_csv(dataset_uri, output_file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def download_all(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Download all datasets defined in DATASETS constant.

        Args:
            output_dir (Optional[Path]): Output directory for datasets.
                Defaults to DEFAULT_DATASET_DIR.
        """
        if output_dir is None:
            output_dir = DEFAULT_DATASET_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset_info in DATASETS.items():
            dataset_type = dataset_info.get("type")
            dataset_uri = dataset_info.get("uri")
            output_file = dataset_info.get("output_file")

            if dataset_type and dataset_uri and output_file:
                self.logger.info(f"Downloading dataset: {dataset_name}")
                output_file_path = output_dir / Path(str(output_file))
                self.download(dataset_type, dataset_uri, output_file_path)
            else:
                msg = f"Dataset info incomplete for {dataset_name}, " "skipping."
                self.logger.warning(msg)
