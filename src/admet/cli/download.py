"""CLI commands for downloading datasets."""

from typing import Optional
from pathlib import Path
import logging

import typer

from admet.data.download import Downloader
from admet.data.constants import DATASETS, DEFAULT_DATASET_DIR


logger = logging.getLogger(__name__)


def download(
    dataset_name: str = typer.Argument(
        "",
        help=(
            "Name of the dataset to download. "
            "Available datasets: {0}. "
            "Use 'all' or omit to download all datasets."
        ).format(", ".join(DATASETS.keys())),
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=("Output directory for downloaded datasets. " f"Defaults to {DEFAULT_DATASET_DIR}."),
    ),
) -> None:
    """
    Download a dataset or all datasets.

    Examples:
        # Download a specific dataset
        admet download expansion_teaser

        # Download all available datasets
        admet download all

        # Download all available datasets (omit dataset name)
        admet download

        # Download a dataset to a specific directory
        admet download expansion_teaser --output-dir ./data
    """
    # Use default directory if not specified
    if output_dir is None:
        output_dir = DEFAULT_DATASET_DIR

    downloader = Downloader(logger=logger)

    if not dataset_name or dataset_name.lower() == "all":
        # Download all datasets
        typer.echo("Downloading all available datasets...")
        downloader.download_all()
        typer.echo("All datasets downloaded successfully.")
    else:
        # Download specific dataset
        if dataset_name not in DATASETS:
            available = ", ".join(DATASETS.keys())
            msg = f"Error: Dataset '{dataset_name}' not found. " f"Available datasets: {available}"
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        dataset_info = DATASETS[dataset_name]
        dataset_type = dataset_info.get("type")
        dataset_uri = dataset_info.get("uri")
        output_file = dataset_info.get("output_file")

        if not all([dataset_type, dataset_uri, output_file]):
            msg = f"Error: Dataset '{dataset_name}' has " "incomplete configuration."
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        # Adjust output file path if output directory is specified
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / Path(str(output_file))

        typer.echo(f"Downloading dataset: {dataset_name}...")
        downloader.download(str(dataset_type), str(dataset_uri), output_file_path)
        msg = f"Dataset '{dataset_name}' downloaded successfully " f"to {output_file_path}."
        typer.echo(msg)


if __name__ == "__main__":
    typer.run(download)
