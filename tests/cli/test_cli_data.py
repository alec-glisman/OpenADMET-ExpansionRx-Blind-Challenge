"""Tests for the `admet data` CLI commands."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

import admet.cli.data as data_module
from admet.cli import app as main_app


def test_split_writes_file(monkeypatch, tmp_path):
    df = pd.DataFrame({"SMILES": ["C", "CC"], "Quality": [1, 0], "LogD": [1.2, 2.3]})
    input_path = tmp_path / "admet_train.csv"
    df.to_csv(input_path, index=False)

    out_dir = tmp_path / "out"

    called = {}

    def fake_pipeline(input_df, **kwargs):
        called["ok"] = True
        # basic sanity checks on passed dataframe
        assert list(input_df["SMILES"]) == list(df["SMILES"])
        return input_df.assign(_augmented=1)

    monkeypatch.setattr(data_module, "pipeline", fake_pipeline)

    runner = CliRunner()
    # Invoke top-level `admet` app so subcommand parsing behaves consistently
    res = runner.invoke(main_app, ["data", "split", "--output", str(out_dir), str(input_path)])

    assert res.exit_code == 0, res.output
    out_file = out_dir / input_path.name
    assert out_file.exists()
    out_df = pd.read_csv(out_file)
    assert "_augmented" in out_df.columns
    assert "Split completed. Augmented file written to:" in res.output


def test_target_columns_and_figdir(monkeypatch, tmp_path):
    df = pd.DataFrame({"SMILES": ["N"], "Quality": [1], "LogD": [0.5], "KSol": [10.0]})
    input_path = tmp_path / "data.csv"
    df.to_csv(input_path, index=False)

    fig_dir = tmp_path / "figs"
    out_dir = tmp_path / "out2"

    def fake_pipeline(input_df, **kwargs):
        # ensure target columns were parsed and forwarded
        assert kwargs.get("target_cols") == ["LogD", "KSol"]
        # the CLI converts fig_dir to a string before passing
        assert kwargs.get("fig_dir") == str(fig_dir)
        return input_df

    monkeypatch.setattr(data_module, "pipeline", fake_pipeline)

    runner = CliRunner()
    # Invoke the top-level `admet` app so subcommand parsing behaves consistently
    res = runner.invoke(
        main_app,
        [
            "data",
            "split",
            "-o",
            str(out_dir),
            "--target-columns",
            "LogD KSol",
            "--fig-dir",
            str(fig_dir),
            str(input_path),
        ],
    )

    assert res.exit_code == 0, res.output
