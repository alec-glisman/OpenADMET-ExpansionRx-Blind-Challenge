"""Unit tests for MLflow integration helpers and CLI logging.

These tests validate parameter flattening, metric flattening, and the
integration of MLflow param/metric/artifact logging from CLI workflows.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("mlflow") is None:
    pytest.skip("typer or mlflow not installed in test environment", allow_module_level=True)

from admet.cli.train import xgb
from admet.train.mlflow_utils import flatten_metrics, flatten_params, set_mlflow_tracking


class FakeInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id


class FakeRun:
    def __init__(self, mlflow, run_name=None, tags=None, run_id="run_1"):
        self.mlflow = mlflow
        self.run_name = run_name
        self.tags = tags or {}
        self.info = FakeInfo(run_id)

    def __enter__(self):
        self.mlflow.active_run_stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.mlflow.active_run_stack:
            self.mlflow.active_run_stack.pop()
        return False


class FakeMlflow:
    def __init__(self):
        self.params_calls = []
        self.metrics_calls = []
        self.metric_calls = []
        self.artifact_calls = []
        self.artifacts_calls = []
        self.tag_calls = []
        self.start_run_calls = []
        self.set_tracking_uri_calls = []
        self.set_experiment_calls = []
        self.active_run_stack = []
        self._run_counter = 0

    def set_tracking_uri(self, uri):
        self.set_tracking_uri_calls.append(uri)

    def set_experiment(self, name):
        self.set_experiment_calls.append(name)

    def start_run(self, run_name=None, tags=None):
        self._run_counter += 1
        run = FakeRun(self, run_name=run_name, tags=tags, run_id=f"run_{self._run_counter}")
        self.start_run_calls.append(run)
        return run

    def active_run(self):
        return self.active_run_stack[-1] if self.active_run_stack else None

    def log_params(self, params):
        self.params_calls.append(params)

    def log_metrics(self, metrics):
        self.metrics_calls.append(metrics)

    def log_metric(self, key, value):
        self.metric_calls.append((key, value))

    def log_artifact(self, path):
        self.artifact_calls.append(path)

    def log_artifacts(self, path):
        self.artifacts_calls.append(path)

    def set_tag(self, key, value):
        current = self.active_run()
        run_id = current.info.run_id if current else None
        self.tag_calls.append((run_id, key, value))


@pytest.fixture()
def fake_mlflow(monkeypatch) -> FakeMlflow:
    fake = FakeMlflow()
    monkeypatch.setattr("admet.cli.train.mlflow", fake)
    monkeypatch.setattr("admet.train.mlflow_utils.mlflow", fake)
    return fake


def test_flatten_params_handles_nested_structures() -> None:
    cfg = {
        "models": {"xgboost": {"objective": "mae", "model_params": {"lr": 0.1, "layers": [64, 32]}}},
        "training": {"seed": None, "tracking_uri": "file:///tmp/mlruns"},
        "ray": {"multi": True},
    }
    flat = flatten_params(cfg, prefix="cfg")
    assert flat["cfg.models.xgboost.objective"] == "mae"
    assert flat["cfg.models.xgboost.model_params.lr"] == 0.1
    assert flat["cfg.models.xgboost.model_params.layers.0"] == 64
    assert flat["cfg.training.seed"] == "null"
    assert flat["cfg.training.tracking_uri"] == "file:///tmp/mlruns"
    assert flat["cfg.ray.multi"] is True


def test_flatten_metrics_ignores_non_numeric_and_nans() -> None:
    run_metrics = {
        "train": {
            "macro": {
                "log": {"mae": 1.0, "rmse": np.nan},
                "linear": {"mae": 1.5, "rmse": math.inf},
            }
        },
        "validation": {"macro": {"log": {"mae": 2.0}, "linear": {"mae": "skip_me"}}},
    }
    flat = flatten_metrics(run_metrics)
    assert flat["metrics.train.macro.log.mae"] == 1.0
    assert flat["metrics.train.macro.linear.mae"] == 1.5
    assert flat["metrics.validation.macro.log.mae"] == 2.0
    assert not any("rmse" in k for k in flat.keys())  # nan/inf filtered out
    assert not any("skip_me" in str(v) for v in flat.values())


def test_set_mlflow_tracking_calls_both(monkeypatch) -> None:
    called = {"uri": None, "exp": None}

    class StubMlflow:
        def set_tracking_uri(self, uri):
            called["uri"] = uri

        def set_experiment(self, name):
            called["exp"] = name

    monkeypatch.setattr("admet.train.mlflow_utils.mlflow", StubMlflow())
    set_mlflow_tracking("file:///tmp/mlruns", "xgb")
    assert called["uri"] == "file:///tmp/mlruns"
    assert called["exp"] == "xgb"


def test_single_run_logs_mlflow_params_and_artifacts(
    fake_mlflow: FakeMlflow, tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    data_root = tmp_path / "data" / "hf_dataset"
    data_root.mkdir(parents=True)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True)
    cfg_path.write_text(
        dedent(
            f"""
            models:
              xgboost:
                objective: "mae"
                early_stopping_rounds: 5
                model_params:
                  n_estimators: 5
            training:
              output_dir: "{output_dir}"
              experiment_name: "xgb"
              tracking_uri: "file://{tmp_path}/mlruns"
              seed: 42
            ray:
              multi: false
            data:
              root: "{data_root}"
              endpoints: ["LogD"]
            """
        )
    )

    run_metrics = {
        "train": {"macro": {"log": {"mae": 1.0}, "linear": {"mae": 1.1}}},
        "validation": {"macro": {"log": {"mae": 2.0}, "linear": {"mae": 2.1}}},
        "test": {"macro": {"log": {"mae": 3.0}, "linear": {"mae": 3.1}}},
    }

    def stub_load_dataset(path, endpoints=None, n_fingerprint_bits=None, fingerprint_config=None):
        return object()

    def stub_train_model(*args, **kwargs):
        return run_metrics, None

    monkeypatch.setattr("admet.cli.train.load_dataset", stub_load_dataset)
    monkeypatch.setattr("admet.cli.train.train_model", stub_train_model)

    xgb(config=cfg_path, data_root=None)

    assert fake_mlflow.set_tracking_uri_calls == [f"file://{tmp_path}/mlruns"]
    assert fake_mlflow.set_experiment_calls == ["xgb"]
    assert fake_mlflow.start_run_calls and fake_mlflow.start_run_calls[0].run_name == data_root.name
    # Two param logs: flattened cfg + CLI args
    assert len(fake_mlflow.params_calls) == 2
    all_param_keys = set().union(*[set(d.keys()) for d in fake_mlflow.params_calls])
    assert "cfg.training.experiment_name" in all_param_keys
    assert "cli.effective_data_root" in all_param_keys
    assert fake_mlflow.metrics_calls and "metrics.train.macro.log.mae" in fake_mlflow.metrics_calls[0]
    assert ("status", "ok") in [(k, v) for _, k, v in fake_mlflow.tag_calls if k == "status"]
    assert output_dir.as_posix() in fake_mlflow.artifacts_calls
    assert str(cfg_path) in fake_mlflow.artifact_calls
    assert any(key == "duration_seconds" for key, _ in fake_mlflow.metric_calls)


def test_ensemble_run_logs_parent_status_and_child_metrics(
    fake_mlflow: FakeMlflow, tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    data_root = tmp_path / "data" / "root"
    data_root.mkdir(parents=True)
    output_dir = tmp_path / "ensemble_artifacts"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics_summary.csv").write_text("metric,value\nmae,1.0\n")
    (output_dir / "metrics_summary.json").write_text('[{"metric": "mae", "value": 1.0}]')

    cfg_path.write_text(
        dedent(
            f"""
            models:
              xgboost:
                objective: "mae"
                model_params:
                  n_estimators: 5
            training:
              output_dir: "{output_dir}"
              experiment_name: "xgb"
              tracking_uri: "file://{tmp_path}/mlruns"
            ray:
              multi: true
            data:
              root: "{data_root}"
              endpoints: ["LogD"]
            """
        )
    )

    child_metrics = {
        "train": {"macro": {"log": {"mae": 1.0}, "linear": {"mae": 1.1}}},
        "validation": {"macro": {"log": {"mae": 2.0}, "linear": {"mae": 2.1}}},
        "test": {"macro": {"log": {"mae": 3.0}, "linear": {"mae": 3.1}}},
    }
    captured_parent_ids = []

    def stub_train_ensemble(*args, **kwargs):
        captured_parent_ids.append(kwargs.get("mlflow_parent_run_id"))
        return {
            "split_0/fold_0": {"run_metrics": child_metrics, "status": "ok"},
            "split_1/fold_0": {"run_metrics": None, "status": "error"},
        }

    monkeypatch.setattr("admet.cli.train.train_ensemble", stub_train_ensemble)

    xgb(config=cfg_path, data_root=None)

    assert captured_parent_ids and captured_parent_ids[0] == fake_mlflow.start_run_calls[0].info.run_id
    # Two param logs: flattened cfg + CLI args
    assert len(fake_mlflow.params_calls) == 2
    # Metrics logged for child run + status counts
    assert len(fake_mlflow.metrics_calls) >= 2
    metric_keys = set().union(*[set(m.keys()) for m in fake_mlflow.metrics_calls])
    assert any(key.startswith("ensemble.split_0/fold_0.train.macro.log.mae") for key in metric_keys) or any(
        key.startswith("ensemble.split_0_fold_0.train.macro.log.mae") for key in metric_keys
    )
    assert any(key.startswith("ensemble.status") for key in metric_keys)
    # Parent run tagged as partial because one child failed
    assert ("status", "partial") in [(k, v) for _, k, v in fake_mlflow.tag_calls if k == "status"]
    assert str(output_dir / "metrics_summary.csv") in fake_mlflow.artifact_calls
    assert str(output_dir / "metrics_summary.json") in fake_mlflow.artifact_calls
