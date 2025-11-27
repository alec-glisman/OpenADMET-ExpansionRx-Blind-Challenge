import types
from pathlib import Path

from admet.train.base import ray_trainer as rt


def test_fit_ensemble_assigns_unique_seeds(monkeypatch, tmp_path):
    """Each discovered dataset should receive a unique, deterministic seed."""

    # Create two dataset directories under the root to mimic discovered HF datasets.
    root = tmp_path / "root"
    ds_a = root / "clusterA" / "split_0" / "fold_0" / "hf_dataset"
    ds_b = root / "clusterB" / "split_1" / "fold_0" / "hf_dataset"
    for ds in (ds_a, ds_b):
        ds.mkdir(parents=True)

    captured_seeds: list[int | None] = []

    class DummyRef:
        def __init__(self, value):
            self.value = value

    def fake_remote(
        trainer_cls,
        trainer_kwargs,
        hf_path,
        root_dir,
        *,
        seed=None,
        **_: object,
    ):
        captured_seeds.append(seed)
        rel = Path(hf_path).relative_to(root_dir).as_posix()
        payload = {
            "run_metrics": {
                "train": {"macro": {}},
                "validation": {"macro": {}},
                "test": {"macro": {}},
            },
            "meta": {"relative_path": rel},
            "status": "ok",
            "start_time": "t0",
            "end_time": "t1",
            "duration_seconds": 1.0,
        }
        return DummyRef((rel, payload))

    # Stub Ray internals to avoid starting a real cluster while preserving control flow.
    monkeypatch.setattr(rt, "_train_single_dataset_remote", types.SimpleNamespace(remote=fake_remote))
    monkeypatch.setattr(rt.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(rt.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(rt.ray, "shutdown", lambda: None)
    monkeypatch.setattr(rt.ray, "wait", lambda seq, num_returns=1: ([seq[0]], seq[1:]) if seq else ([], []))
    monkeypatch.setattr(rt.ray, "get", lambda ref: ref.value)

    trainer = rt.BaseEnsembleTrainer(trainer_cls=object)
    base_seed = 123

    trainer.fit_ensemble(root, output_root=tmp_path / "models", seed=base_seed)

    # Seeds should increment deterministically with dataset order.
    assert captured_seeds == [base_seed + 0, base_seed + 1]
