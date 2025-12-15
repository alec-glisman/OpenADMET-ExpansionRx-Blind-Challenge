"""
Tests for ChempropEnsemble configuration propagation and discovery helpers.
"""

import pytest

from admet.model.chemprop.config import CurriculumConfig, EnsembleConfig
from admet.model.chemprop.ensemble import ChempropEnsemble, SplitFoldInfo


@pytest.mark.no_mlflow_runs
def test_create_single_model_config_passes_curriculum(tmp_path):
    # Create a fake data structure that the ensemble expects
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    # Simple minimal EnsembleConfig with curriculum enabled
    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Create a curriculum setting
    ens_cfg.curriculum = CurriculumConfig(
        enabled=True,
        quality_col="Quality",
        qualities=["h", "m"],
        patience=2,
        seed=42,
    )

    # Disable MLflow in unit tests to avoid starting real runs
    ens_cfg.mlflow.tracking = False
    ensemble = ChempropEnsemble(ens_cfg)

    # Create dummy SplitFoldInfo
    info = SplitFoldInfo(
        split_idx=0,
        fold_idx=0,
        data_dir=data_dir,
        train_file=data_dir / "train.csv",
        validation_file=data_dir / "validation.csv",
    )

    model_cfg = ensemble._create_single_model_config(info)

    assert model_cfg.curriculum.enabled is True
    assert model_cfg.curriculum.quality_col == "Quality"
    assert model_cfg.curriculum.qualities == ["h", "m"]
    assert model_cfg.curriculum.patience == 2
    assert model_cfg.curriculum.seed == 42


def test_discover_splits_folds_sorted(tmp_path):
    # Create a fake data structure with unsorted split/fold directory names
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    splits = ["split_10", "split_2", "split_1"]
    folds = ["fold_2", "fold_10", "fold_1"]

    for s in splits:
        split_dir = data_dir / s
        split_dir.mkdir()
        for f in folds:
            fold_dir = split_dir / f
            fold_dir.mkdir(parents=True)
            # Create expected files
            (fold_dir / "train.csv").write_text("a,b,c\n1,2,3\n")
            (fold_dir / "validation.csv").write_text("a,b,c\n4,5,6\n")

    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Disable MLflow for tests
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)
    infos = ensemble.discover_splits_folds()

    # Expected order is numeric: split 1, then 2, then 10; folds 1,2,10
    expected = []
    for split_idx in [1, 2, 10]:
        for fold_idx in [1, 2, 10]:
            expected.append((split_idx, fold_idx))

    discovered = [(info.split_idx, info.fold_idx) for info in infos]

    assert discovered == expected


def test_discover_splits_folds_sorted_with_filters(tmp_path):
    # Create a fake data structure with unsorted split/fold directories
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    splits = ["split_10", "split_2", "split_1"]
    folds = ["fold_2", "fold_10", "fold_1"]

    for s in splits:
        split_dir = data_dir / s
        split_dir.mkdir()
        for f in folds:
            fold_dir = split_dir / f
            fold_dir.mkdir(parents=True)
            (fold_dir / "train.csv").write_text("a,b,c\n1,2,3\n")
            (fold_dir / "validation.csv").write_text("a,b,c\n4,5,6\n")

    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Provide filters unordered - discovery should sort them numerically
    ens_cfg.data.splits = [2, 1]
    ens_cfg.data.folds = [10, 2]
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)
    infos = ensemble.discover_splits_folds()

    # Expected: splits 1 then 2 (numeric), folds 2 then 10 (numeric), filtered
    expected = [(1, 2), (1, 10), (2, 2), (2, 10)]
    discovered = [(info.split_idx, info.fold_idx) for info in infos]

    assert discovered == expected


def test_create_single_model_config_includes_inter_task_affinity(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    ens_cfg.inter_task_affinity.enabled = True
    ens_cfg.inter_task_affinity.log_to_mlflow = True
    ens_cfg.inter_task_affinity.compute_every_n_steps = 2
    # Skip MLflow initialization for unit test
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)

    info = SplitFoldInfo(
        split_idx=0,
        fold_idx=0,
        data_dir=data_dir,
        train_file=data_dir / "train.csv",
        validation_file=data_dir / "validation.csv",
    )

    cfg = ensemble._create_single_model_config(info)
    assert cfg.inter_task_affinity.enabled is True
    assert cfg.inter_task_affinity.log_to_mlflow is True
    assert cfg.inter_task_affinity.compute_every_n_steps == 2
