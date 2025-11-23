"""
OpenADMET CLI Application
=========================

Command-line interface for OpenADMET challenge tools and utilities.

.. module:: admet.cli

"""

import typer

from admet.cli import download
from admet.cli import train as train_module
from admet.cli import split as split_module
from admet.cli import ensemble as ensemble_module


app = typer.Typer(
    help="OpenADMET Challenge tools and utilities.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def _configure_global_logging(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Global logging level for the application (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    log_format: str | None = typer.Option(None, "--log-format", help="Optional log format string."),
    log_file: str | None = typer.Option(None, "--log-file", help="Optional log file path to write logs to."),
    log_json: bool = typer.Option(False, "--log-json", help="Emit structured JSON logs."),
) -> None:
    """Configure global logging for the CLI and any invoked subcommands.

    This callback is invoked before any subcommand and applies a root-level
    logging handler based on CLI options.
    """
    # Import locally to avoid side effects during module import in tests etc.
    from admet.logging import configure_logging

    configure_logging(level=log_level, fmt=log_format, file=log_file, structured=log_json)


download_app = typer.Typer(
    help="Download datasets for the OpenADMET challenge.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
download_app.command()(download.download)
app.add_typer(download_app, name="download")

train_app = typer.Typer(
    help="Model training commands.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
train_app.command()(train_module.xgb)
app.add_typer(train_app, name="train")

split_app = typer.Typer(
    help="Dataset splitting commands.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
split_app.command()(split_module.datasets)
app.add_typer(split_app, name="split")

app.command(name="ensemble-eval")(ensemble_module.ensemble_eval)


__all__ = ["app"]

if __name__ == "__main__":  # pragma: no cover
    app()
