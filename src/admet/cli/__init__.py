"""OpenADMET CLI application."""

import typer

from admet.cli import download
from admet.cli import train as train_module
from admet.cli import split as split_module


app = typer.Typer(
    help="OpenADMET Challenge tools and utilities.",
    no_args_is_help=True,
)

download_app = typer.Typer(
    help="Download datasets for the OpenADMET challenge.",
)
download_app.command()(download.download)
app.add_typer(download_app, name="download")

train_app = typer.Typer(help="Model training commands.")
train_app.command()(train_module.xgb)
app.add_typer(train_app, name="train")

split_app = typer.Typer(help="Dataset splitting commands.")
split_app.command()(split_module.datasets)
app.add_typer(split_app, name="split")


if __name__ == "__main__":  # pragma: no cover
    app()
