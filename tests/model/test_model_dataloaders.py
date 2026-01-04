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

    # Avoid heavy dataloader by patching chemprop.data.build_dataloader and DynamicCurriculumSampler
    fake_dataloader = object()

    # Create a fake sampler class that returns a fake sampler instance
    class FakeSampler:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    monkeypatch.setattr(
        "admet.model.chemprop.model.DynamicCurriculumSampler",
        FakeSampler,
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
        # With curriculum enabled, quality labels should be set
        assert model._quality_labels.get("train") is not None
        # Curriculum creates a custom DataLoader with sampler, not using build_dataloader
        from torch.utils.data import DataLoader

        assert isinstance(model.dataloaders["train"], DataLoader)
    else:
        # Without curriculum, quality labels not set and build_dataloader is used
        assert model._quality_labels.get("train") is None
        assert model.dataloaders["train"] is fake_dataloader
