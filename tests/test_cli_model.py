"""Tests for the `admet model` CLI commands."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from typer.testing import CliRunner


def test_model_train_invokes_module(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    fake_module = SimpleNamespace(main=fake_main)

    monkeypatch.setattr(importlib, "import_module", lambda name: fake_module)

    runner = CliRunner()
    # Create a temp Typer app and register our command
    from admet.cli.model import model_app

    result = runner.invoke(model_app, ["train", "--config", "configs/foo.yaml"])

    assert result.exit_code == 0
    assert called.get("ran", False) is True


def test_model_ensemble_and_hpo(monkeypatch):
    # Ensure ensemble and hpo commands call their module mains
    calls = []

    def make_fake(name):
        return SimpleNamespace(main=lambda: calls.append(name))

    monkeypatch.setattr(importlib, "import_module", lambda name: make_fake(name))

    runner = CliRunner()
    from admet.cli.model import model_app

    r1 = runner.invoke(model_app, ["ensemble", "--config", "c.yaml"])
    r2 = runner.invoke(model_app, ["hpo", "--config", "c.yaml"])

    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert "admet.model.chemprop.ensemble" in calls[0]
    assert "admet.model.chemprop.hpo" in calls[1]
