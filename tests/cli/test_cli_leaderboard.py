"""Tests for the `admet leaderboard` CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from typer.testing import CliRunner

from admet.cli.leaderboard import leaderboard_app


def test_leaderboard_scrape_creates_reports(tmp_path, monkeypatch):
    # Prepare a fake tables dict
    tables = {
        "Average": pd.DataFrame({"rank": [1], "user": ["alice"], "mae": [0.5]}),
        "LogD": pd.DataFrame({"rank": [1], "user": ["alice"], "mae": [0.5]}),
    }

    fake_client = MagicMock()
    fake_client.fetch_all_tables.return_value = tables

    monkeypatch.setattr("admet.cli.leaderboard.LeaderboardClient", lambda cfg: fake_client)

    # Patch report and plot functions to avoid heavy I/O
    called = {}

    def fake_generate_markdown_report(results, path, include_figures=True):
        called["md"] = str(path)

    monkeypatch.setattr("admet.cli.leaderboard.generate_markdown_report", fake_generate_markdown_report)
    monkeypatch.setattr(
        "admet.cli.leaderboard.save_csv_data", lambda results, data_dir: called.update({"csv": str(data_dir)})
    )
    monkeypatch.setattr(
        "admet.cli.leaderboard.save_summary_statistics", lambda results, path: called.update({"summary": str(path)})
    )
    monkeypatch.setattr(
        "admet.cli.leaderboard.generate_all_plots", lambda *args, **kwargs: called.update({"plots": True})
    )

    outdir = tmp_path / "results"
    runner = CliRunner()
    result = runner.invoke(leaderboard_app, ["scrape", "--user", "alice", "--output", str(outdir)])

    assert result.exit_code == 0
    assert "md" in called
    assert "csv" in called
    assert "summary" in called
