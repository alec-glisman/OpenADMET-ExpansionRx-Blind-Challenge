"""CLI wrappers for data utilities (splitting, preprocessing)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from admet.data.split import pipeline

data_app = typer.Typer(name="data", help="Data utilities (splits, preprocessing)")


@data_app.command("split")
def split_command(
    input: Path = typer.Argument(..., help="Path to input CSV file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for split CSV"),
    smiles_col: str = typer.Option("SMILES", help="SMILES column name"),
    quality_col: str = typer.Option("Quality", help="Quality column name"),
    target_columns: Optional[str] = typer.Option(None, help="Space-separated target column names"),
    cluster_method: str = typer.Option("bitbirch", help="Clustering method"),
    split_method: str = typer.Option("multilabel_stratified_kfold", help="Splitting method"),
    n_splits: int = typer.Option(5, help="Number of repeated splits"),
    n_folds: int = typer.Option(5, help="Folds per split"),
    fig_dir: Optional[Path] = typer.Option(None, help="Directory to save diagnostic figures"),
) -> None:
    """Run the data splitting pipeline and save the augmented dataframe.

    Example:
        admet data split data/admet_train.csv --output outputs/ --smiles-col SMILES
    """
    df = pd.read_csv(input)

    target_cols: Optional[List[str]] = None
    if target_columns:
        target_cols = target_columns.split()

    result_df = pipeline(
        df,
        smiles_col=smiles_col,
        quality_col=quality_col,
        target_cols=target_cols,
        cluster_method=cluster_method,
        split_method=split_method,
        n_splits=n_splits,
        n_folds=n_folds,
        fig_dir=str(fig_dir) if fig_dir is not None else None,
    )

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / input.name
    result_df.to_csv(out_path, index=False)
    typer.echo(f"Split completed. Augmented file written to: {out_path}")
