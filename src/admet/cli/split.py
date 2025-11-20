"""admet.cli.split
===================

Generate temporal and stratified dataset splits with optional visualizations.

Wraps :class:`admet.data.dataset_split_pipeline.DatasetSplitPipeline` exposing
flags to control temporal vs stratified generation, overwrite behavior, and
logging verbosity.

Examples
--------
Create all splits (temporal + stratified)::

    admet split datasets path/to/data

Stratified only with custom output directory::

    admet split datasets path/to/data --output ./my_splits --no-temporal

Verbose debugging::

    admet split datasets path/to/data --log-level DEBUG
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
    """Create dataset splits with fingerprints and optional visualizations.

    Parameters
    ----------
    base_data_dir : pathlib.Path
        Directory containing cleaned dataset CSV files.
    output_dir : pathlib.Path, optional
        Destination directory for splits & figures; defaults to ``parent/splits``.
    create_temporal : bool, optional
        Generate temporal train/validation/test split for high-quality dataset.
    create_stratified : bool, optional
        Generate stratified k-fold splits per dataset quality.
    overwrite : bool, optional
        Overwrite existing artifacts if present; otherwise skip.
    log_level : str, optional
        Logging level name (``DEBUG``, ``INFO``, etc.).
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
    except Exception as e:  # pragma: no cover - CLI runtime error path
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


__all__ = ["datasets"]
