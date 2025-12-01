"""
Unit tests for ensemble discovery and utilities.

Tests `discover_splits_folds` and `_flatten_dict` in `ChempropEnsemble`.
"""

from omegaconf import OmegaConf

from admet.model.chemprop.config import EnsembleConfig
from admet.model.chemprop.ensemble import ChempropEnsemble, SplitFoldInfo


def make_split_fold_tree(tmp_path, splits=(0, 1), folds=(0, 1)):
    base = tmp_path / "splits"
    base.mkdir()
    for s in splits:
        split_dir = base / f"split_{s}"
        split_dir.mkdir()
        for f in folds:
            fold_dir = split_dir / f"fold_{f}"
            fold_dir.mkdir()
            # Create dummy train/validation csv files
            (fold_dir / "train.csv").write_text("smiles,y\nC,1")
            (fold_dir / "validation.csv").write_text("smiles,y\nC,1")
    return base


def test_discover_splits_folds(tmp_path):
    base = make_split_fold_tree(tmp_path, splits=(0, 1), folds=(0, 1))

    cfg = OmegaConf.structured(EnsembleConfig)
    cfg.data.data_dir = str(base)
    cfg.data.splits = [0, 1]
    cfg.data.folds = [0, 1]
    # Disable MLflow tracking in unit tests to avoid starting runs
    cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(cfg)
    infos = ensemble.discover_splits_folds()
    assert isinstance(infos, list)
    assert all(isinstance(i, SplitFoldInfo) for i in infos)
    assert len(infos) == 4


def test_flatten_dict():
    cfg = {"a": {"b": {"c": 1}}, "x": 2}
    base = OmegaConf.structured(EnsembleConfig)
    base.mlflow.tracking = False
    ensemble = ChempropEnsemble(base)
    flat = ensemble._flatten_dict(cfg)
    # keys should be flattened
    assert "a.b.c" in flat
    assert flat["x"] == "2" or flat["x"] == 2
