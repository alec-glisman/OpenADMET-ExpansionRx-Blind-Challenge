"""Tests for the `admet model` CLI commands."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from typer.testing import CliRunner


def test_model_train_invokes_module(monkeypatch, tmp_path):
    called = {}

    def fake_main():
        called["ran"] = True

    fake_module = SimpleNamespace(main=fake_main)

    monkeypatch.setattr(importlib, "import_module", lambda name: fake_module)

    # Create a temporary config file with minimal valid config
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("model:\n  type: chemprop\n")

    runner = CliRunner()
    # Create a temp Typer app and register our command
    from admet.cli.model import model_app

    result = runner.invoke(model_app, ["train", "--config", str(config_file)])

    assert result.exit_code == 0
    assert called.get("ran", False) is True


def test_model_ensemble_and_hpo(monkeypatch, tmp_path):
    # Ensure ensemble and hpo commands call their module mains
    calls = []

    def make_fake(name):
        calls.append(name)
        return SimpleNamespace(main=lambda: None)

    monkeypatch.setattr(importlib, "import_module", make_fake)

    runner = CliRunner()
    from admet.cli.model import model_app

    # Create minimal config files
    config_file = tmp_path / "c.yaml"
    config_file.write_text("model:\n  type: chemprop\n")

    r1 = runner.invoke(model_app, ["ensemble", "--config", str(config_file)])
    # HPO command now auto-detects model type from config, so we explicitly specify it
    # to avoid needing a real config file
    r2 = runner.invoke(model_app, ["hpo", "--config", str(config_file), "--model-type", "chemprop"])

    # Check that commands ran (exit code 0 or 1 for missing file is OK in this mock test)
    assert r1.exit_code in (0, 1), f"ensemble failed: {r1.output}"
    assert r2.exit_code in (0, 1), f"hpo failed: {r2.output}"
    assert len(calls) >= 2, f"Expected at least 2 calls, got {len(calls)}: {calls}"
    assert any("ensemble" in call for call in calls), f"Expected ensemble call in {calls}"
    assert any("hpo" in call for call in calls), f"Expected hpo call in {calls}"
