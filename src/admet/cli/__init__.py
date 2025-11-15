"""OpenADMET CLI application."""

import typer

from admet.cli import download


app = typer.Typer(
    help="OpenADMET Challenge tools and utilities.",
    no_args_is_help=True,
)

download_app = typer.Typer(
    help="Download datasets for the OpenADMET challenge.",
)
download_app.command()(download.download)
app.add_typer(download_app)


if __name__ == "__main__":  # pragma: no cover
    app()
