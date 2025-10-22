"""Entry point for the `admet.cli` package when run as a module.

This file allows `python -m admet.cli` to work by invoking the Typer
application defined in :mod:`admet.cli` (the package's ``app``).
"""

from __future__ import annotations

from admet.cli import app


def main() -> None:  # pragma: no cover - trivial wrapper
    """Run the Typer CLI application."""
    app()


if __name__ == "__main__":
    main()
