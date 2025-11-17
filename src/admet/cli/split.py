"""
Dataset splitting CLI command.

This module provides a command-line interface for generating dataset splits
with fingerprints and creating stratified k-fold cross-validation splits.
"""

from pathlib import Path
import logging
from typing import Optional

import typer

from admet.data.dataset_split_pipeline import DatasetSplitPipeline, setup_logging

logger = logging.getLogger(__name__)


def datasets(
    base_data_dir: Path = typer.Argument(
        ...,
        help="Path to directory containing cleaned dataset CSV files.",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for splits and figures. Defaults to parent/splits.",
    ),
    create_temporal: bool = typer.Option(
        True,
        "--temporal/--no-temporal",
        help="Whether to create temporal splits for high-quality dataset.",
    ),
    create_stratified: bool = typer.Option(
        True,
        "--stratified/--no-stratified",
        help="Whether to create stratified k-fold splits.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing files. If False, skip existing files with warning.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """
    Create dataset splits with fingerprints and stratified k-fold cross-validation.

    This command orchestrates the full workflow of:
    1. Loading cleaned datasets
    2. Calculating Morgan fingerprints
    3. Creating temporal and/or stratified splits
    4. Generating analysis visualizations

    Examples:

        # Create all splits in default location
        openadmet split datasets path/to/data

        # Create only stratified splits with custom output
        openadmet split datasets path/to/data \\
            --output ./my_splits \\
            --no-temporal

        # Debug mode with verbose logging
        openadmet split datasets path/to/data \\
            --log-level DEBUG
    """
    # Setup logging
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=log_level_int)

    logger.info("Starting dataset split pipeline")
    logger.info("Input directory: %s", base_data_dir)
    logger.info("Creating temporal splits: %s", create_temporal)
    logger.info("Creating stratified splits: %s", create_stratified)
    logger.info("Overwrite existing files: %s", overwrite)

    try:
        pipeline = DatasetSplitPipeline(base_data_dir, output_dir, overwrite=overwrite)
        pipeline.run(
            create_temporal=create_temporal,
            create_stratified=create_stratified,
        )
        logger.info("Dataset splitting completed successfully!")
        typer.echo(
            typer.style(
                "✓ Pipeline completed successfully!",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )
        typer.echo(f"Output directory: {pipeline.output_dir}")
        typer.echo(f"Figures directory: {pipeline.figure_dir}")
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        typer.echo(
            typer.style(
                f"✗ Error: {e}",
                fg=typer.colors.RED,
                bold=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
