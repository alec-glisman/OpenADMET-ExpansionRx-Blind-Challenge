"""Main CLI entry point for ADMET tools."""

from admet.cli import app
from admet.cli.data import data_app
from admet.cli.leaderboard import leaderboard_app
from admet.cli.model import model_app


def main() -> None:
    """Main CLI entry point."""
    # Register subcommands
    app.add_typer(leaderboard_app, name="leaderboard")
    app.add_typer(model_app, name="model")
    app.add_typer(data_app, name="data")

    # Run the CLI
    app()


if __name__ == "__main__":
    main()
