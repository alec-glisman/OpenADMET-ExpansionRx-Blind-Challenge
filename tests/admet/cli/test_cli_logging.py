"""Unit tests for CLI logging behavior.

These tests check that the CLI correctly configures the global logging level
based on `--log-level` flags and defaults.
"""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from admet.cli import app

# No third-party typing required in these small tests


@pytest.mark.integration
@pytest.mark.slow
def test_cli_sets_log_level_debug() -> None:
    """Ensure the CLI callback sets the global logging level to DEBUG.

    This uses the CLI runner to invoke the `download` subcommand help with
    `--log-level DEBUG` so that the CLI callback is executed.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["--log-level", "DEBUG", "download", "--help"])
    if result.exit_code != 0:
        pytest.fail(f"CLI invocation failed with exit code {result.exit_code}: {result.output}")
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.integration
@pytest.mark.slow
def test_default_log_level_is_info() -> None:
    """Verify that the default global log level is INFO when not specified."""
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    if result.exit_code != 0:
        pytest.fail(f"CLI invocation failed with exit code {result.exit_code}: {result.output}")
    assert logging.getLogger().level == logging.INFO
