"""Unit tests for training optimizations such as dtype casting and
ray remote invocation parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet.train.base.ray_trainer import BaseEnsembleTrainer
from admet.train.base.utils import _extract_features, _extract_targets


@pytest.mark.unit
def test_extractors_cast_to_float32_without_copy() -> None:
    df = pd.DataFrame(
        {
            "fp1": np.array([1.0, 2.0], dtype=np.float64),
            "fp2": np.array([3.0, 4.0], dtype=np.float64),
            "y1": np.array([5.0, 6.0], dtype=np.float64),
        }
    )
    feats = _extract_features(df, ["fp1", "fp2"])
    targets = _extract_targets(df, ["y1"])
    assert feats.dtype == np.float32
    assert targets.dtype == np.float32
    # pandas may or may not return a view, but at least the dtype is narrowed and shape is preserved
    assert feats.shape == (2, 2)
    assert targets.shape == (2, 1)


@pytest.mark.unit
def test_worker_thread_limit_passed_to_remote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Stub ray API to avoid spinning up real cluster
    class StubRay:
        initialized = False
        init_calls: list[int] = []
        results: list[object] = []

        @classmethod
        def is_initialized(cls):
            return cls.initialized

        @classmethod
        def init(cls, num_cpus=None, address=None, ignore_reinit_error=True):
            cls.initialized = True
            cls.init_calls.append(num_cpus if address is None else address)

        @classmethod
        def cluster_resources(cls):
            return {"CPU": cls.init_calls[-1] if cls.init_calls else 0}

        @classmethod
        def wait(cls, remaining, num_returns=1):
            done = [remaining[0]]
            return done, set(remaining[1:])

        @classmethod
        def get(cls, obj):
            return cls.results.pop(0)

        @classmethod
        def shutdown(cls):
            cls.initialized = False

    # Capture worker_thread_limit passed into remote calls
    remote_calls = []

    class StubRemote:
        @staticmethod
        def remote(*args, **kwargs):
            remote_calls.append(kwargs.get("worker_thread_limit"))
            return f"obj_{len(remote_calls)}"

    # Prepare two fake results to satisfy the wait/get loop
    StubRay.results = [
        (
            "ds1",
            {
                "run_metrics": {"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}},
                "status": "ok",
            },
        ),
        (
            "ds2",
            {
                "run_metrics": {"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}},
                "status": "ok",
            },
        ),
    ]

    from admet.train.base import ray_trainer as rt

    monkeypatch.setattr(rt, "ray", StubRay)
    monkeypatch.setattr(rt, "_train_single_dataset_remote", StubRemote)

    class DummyEnsemble(BaseEnsembleTrainer):
        def discover_datasets(self, root: Path):
            return [root / "split_0" / "fold_0" / "hf_dataset", root / "split_1" / "fold_0" / "hf_dataset"]

    ensemble = DummyEnsemble(trainer_cls=object, trainer_kwargs={"model_params": {"foo": "bar"}})
    root = tmp_path / "data"
    root.mkdir()
    # Explicit num_cpus to control thread math
    ensemble.fit_ensemble(root, num_cpus=4, output_root=tmp_path / "out", worker_thread_limit=2)

    assert StubRay.init_calls and StubRay.init_calls[0] == 4
    assert remote_calls == [2, 2]
