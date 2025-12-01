"""
Tests for ChempropModel dataloader creation and curriculum integration.
"""

import pytest

from admet.model.chemprop.config import CurriculumConfig
from admet.model.chemprop.model import ChempropModel


@pytest.mark.parametrize("enable_curriculum", [True, False])
def test_prepare_dataloaders_uses_sampler(monkeypatch, train_val_dataframes, enable_curriculum):
    """Test that dataloaders use curriculum sampler when enabled."""
    train_df, val_df = train_val_dataframes

    # Prepare a curriculum config enabled only if requested
    curr_cfg = CurriculumConfig(
        enabled=enable_curriculum,
        quality_col="Quality",
        qualities=["high", "medium", "low"],
        patience=1,
        seed=42,
    )

    # Avoid heavy dataloader by patching chemprop.data.build_dataloader and build_curriculum_sampler
    fake_dataloader = object()
    fake_sampler = object()

    monkeypatch.setattr(
        "admet.model.chemprop.model.build_curriculum_sampler",
        lambda *args, **kwargs: fake_sampler,
    )
    monkeypatch.setattr(
        "admet.model.chemprop.model.data.build_dataloader",
        lambda *args, **kwargs: fake_dataloader,
    )

    # Create a minimal ChempropModel - inject small hyperparams via ChempropConfig
    # Only construct model; _prepare_dataloaders will run and use our monkeypatches
    model = ChempropModel(
        df_train=train_df,
        df_validation=val_df,
        smiles_col="SMILES",
        target_cols=["LogD"],
        target_weights=[1.0],
        progress_bar=False,
        hyperparams=None,
        mlflow_tracking=False,
        curriculum_config=curr_cfg,
    )

    # Check that dataloaders assigned for train and validation
    assert model.dataloaders["train"] is not None
    assert model.dataloaders["validation"] is not None

    if enable_curriculum:
        assert model._quality_labels["train"] is not None
        assert model.dataloaders["train"] is fake_dataloader
    else:
        # no sampler applied; dataloader is still fake_dataloader but quality not set
        assert model._quality_labels["train"] is None
        assert model.dataloaders["train"] is fake_dataloader
