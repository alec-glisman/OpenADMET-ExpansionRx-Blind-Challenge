import logging
import pytest
from typer.testing import CliRunner

from admet.cli import app


def test_cli_sets_log_level_debug():
    """Ensure the CLI callback sets the global logging level.

    We call the CLI with `--log-level DEBUG` and check the root logger level.
    """
    runner = CliRunner()
    result = runner.invoke(
        app, ["--log-level", "DEBUG", "download", "--help"]
    )  # callback invoked before subcommand
    if result.exit_code != 0:
        pytest.fail(f"CLI invocation failed with exit code {result.exit_code}: {result.output}")
    assert logging.getLogger().level == logging.DEBUG


def test_default_log_level_is_info():
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])  # callback invoked
    if result.exit_code != 0:
        pytest.fail(f"CLI invocation failed with exit code {result.exit_code}: {result.output}")
    assert logging.getLogger().level == logging.INFO
