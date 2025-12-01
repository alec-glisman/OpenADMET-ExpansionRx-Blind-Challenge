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
