"""CLI tests for the ensemble-eval subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from admet.cli import app


@pytest.mark.integration
@pytest.mark.slow
def test_ensemble_cli_eval_and_blind(tmp_path: Path):
    runner = CliRunner()
    # Create a dummy model saved on disk that matches detection heuristics.
    model_dir = tmp_path / "model1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"endpoints": ["LogD", "KSOL"], "model_params": {}, "random_state": 123, "n_features": 2048})
    )

    # Create labeled dataset CSV
    eval_csv = tmp_path / "eval.csv"
    eval_csv.write_text("Molecule Name,SMILES,Dataset,LogD,KSOL\n" + "mol1,CCO,A,0.1,1.1\n" + "mol2,CCN,A,0.2,1.2\n")

    # Create blind CSV
    blind_csv = tmp_path / "blind.csv"
    blind_csv.write_text("Molecule Name,SMILES\n" + "mol3,CCC\n")

    out_dir = tmp_path / "out"
    # Run CLI: name of command is ensemble-eval
    result = runner.invoke(
        app,
        [
            "--log-level",
            "DEBUG",
            "ensemble-eval",
            "--model-dirs",
            str(model_dir),
            "--eval-dataset",
            str(eval_csv),
            "--blind-dataset",
            str(blind_csv),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"
    # Check outputs exist
    assert (out_dir / "eval" / "predictions_log.csv").exists()
    assert (out_dir / "eval" / "predictions_linear.csv").exists()
    assert (out_dir / "blind" / "predictions_log.csv").exists()
    assert (out_dir / "blind" / "predictions_linear.csv").exists()
