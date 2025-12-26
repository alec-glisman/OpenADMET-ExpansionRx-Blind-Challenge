"""CLI package for ADMET tools."""

import typer

app = typer.Typer(
    name="admet",
    help="ADMET modeling and analysis tools",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

from admet.cli.data import data_app  # noqa: E402

# Register sub-typer commands at import time so the app can be used programmatically
# (e.g., in tests) without invoking the full CLI entrypoint.
from admet.cli.leaderboard import leaderboard_app  # noqa: E402
from admet.cli.model import model_app  # noqa: E402

app.add_typer(leaderboard_app, name="leaderboard")
app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")

__all__ = ["app"]
