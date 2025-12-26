"""Tests for ChemeleonModel and callbacks."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from admet.model.config import UnfreezeScheduleConfig
from admet.model.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clear_and_register():
    """Clear registry and re-register chemeleon model."""
    ModelRegistry.clear()

    # Force re-import to trigger decorator registration
    import admet.model.chemeleon.model

    importlib.reload(admet.model.chemeleon.model)

    yield
    ModelRegistry.clear()


from admet.model.chemeleon import ChemeleonModel, GradualUnfreezeCallback  # noqa: E402


class TestGradualUnfreezeCallback:
    """Tests for GradualUnfreezeCallback."""

    def test_init_frozen(self):
        """Test initialization with frozen encoder."""
        config = UnfreezeScheduleConfig(
            freeze_encoder=True,
            freeze_decoder_initially=True,
        )
        callback = GradualUnfreezeCallback(config)

        assert callback.is_encoder_frozen
        assert callback.is_decoder_frozen

    def test_init_unfrozen(self):
        """Test initialization with unfrozen components."""
        config = UnfreezeScheduleConfig(
            freeze_encoder=False,
            freeze_decoder_initially=False,
        )
        callback = GradualUnfreezeCallback(config)

        assert not callback.is_encoder_frozen
        assert not callback.is_decoder_frozen

    def test_unfreeze_at_epoch(self):
        """Test unfreezing at specified epoch."""
        config = UnfreezeScheduleConfig(
            freeze_encoder=True,
            unfreeze_encoder_epoch=5,
        )
        callback = GradualUnfreezeCallback(config)

        # Create mock trainer and module
        trainer = MagicMock()
        trainer.current_epoch = 5

        pl_module = MagicMock()
        mock_encoder = MagicMock()
        pl_module.message_passing = mock_encoder

        callback.on_train_epoch_start(trainer, pl_module)

        # Encoder should be unfrozen
        assert not callback.is_encoder_frozen

    def test_no_unfreeze_before_epoch(self):
        """Test encoder stays frozen before unfreeze epoch."""
        config = UnfreezeScheduleConfig(
            freeze_encoder=True,
            unfreeze_encoder_epoch=10,
        )
        callback = GradualUnfreezeCallback(config)

        trainer = MagicMock()
        trainer.current_epoch = 5

        pl_module = MagicMock()

        callback.on_train_epoch_start(trainer, pl_module)

        # Encoder should still be frozen
        assert callback.is_encoder_frozen


class TestChemeleonModel:
    """Tests for ChemeleonModel."""

    def test_registration(self):
        """Test model is registered with ModelRegistry."""
        assert ModelRegistry.is_registered("chemeleon")
        assert ModelRegistry.get("chemeleon").__name__ == "ChemeleonModel"

    def test_from_config(self):
        """Test creating model from config."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "chemeleon",
                    "chemeleon": {
                        "checkpoint_path": "auto",
                        "freeze_encoder": True,
                        "ffn_hidden_dim": 256,
                    },
                },
                "data": {
                    "smiles_col": "smiles",
                    "target_cols": ["target"],
                },
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel.from_config(config)

        assert model.model_type == "chemeleon"
        assert model.is_fitted is False

    def test_registry_create(self):
        """Test creating via ModelRegistry."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "chemeleon",
                    "chemeleon": {
                        "checkpoint_path": "auto",
                    },
                },
                "data": {"target_cols": ["target"]},
                "mlflow": {"enabled": False},
            }
        )

        model = ModelRegistry.create(config)

        assert model.__class__.__name__ == "ChemeleonModel"
        assert model.model_type == "chemeleon"

    def test_predict_without_fit_raises(self):
        """Test that predict raises if not fitted."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemeleon"},
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(["CCO"])

    def test_get_trainer_callbacks(self):
        """Test get_trainer_callbacks returns unfreeze callback."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemeleon"},
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)
        callbacks = model.get_trainer_callbacks()

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], GradualUnfreezeCallback)

    @patch("admet.model.chemeleon.model.urllib.request.urlretrieve")
    @patch("admet.model.chemeleon.model.Path.exists")
    def test_download_from_zenodo(self, mock_exists, mock_urlretrieve):
        """Test auto-download from Zenodo."""
        mock_exists.return_value = False

        config = OmegaConf.create(
            {
                "model": {
                    "type": "chemeleon",
                    "chemeleon": {"checkpoint_path": "auto"},
                },
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)
        path = model._download_from_zenodo()

        mock_urlretrieve.assert_called_once()
        assert "chemeleon_mp.pt" in path

    def test_get_best_checkpoint_path_from_callback(self):
        """Test _get_best_checkpoint_path finds checkpoint via ModelCheckpoint callback."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from lightning.pytorch.callbacks import ModelCheckpoint

        config = OmegaConf.create(
            {
                "model": {"type": "chemeleon"},
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)

        # Create a mock trainer with ModelCheckpoint callback
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.ckpt"
            checkpoint_path.touch()

            mock_callback = MagicMock(spec=ModelCheckpoint)
            mock_callback.best_model_path = str(checkpoint_path)

            mock_trainer = MagicMock()
            mock_trainer.callbacks = [mock_callback]
            model.trainer = mock_trainer

            result = model._get_best_checkpoint_path()
            assert result == str(checkpoint_path)

    def test_get_best_checkpoint_path_returns_none_when_not_found(self):
        """Test _get_best_checkpoint_path returns None when no checkpoint exists."""
        from unittest.mock import MagicMock

        config = OmegaConf.create(
            {
                "model": {"type": "chemeleon"},
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)

        # Create mock trainer with no ModelCheckpoint callback
        mock_trainer = MagicMock()
        mock_trainer.callbacks = []
        model.trainer = mock_trainer

        result = model._get_best_checkpoint_path()
        assert result is None

    def test_checkpoint_dir_persistence_during_setup(self):
        """Test that checkpoint directory persists after _setup_trainer."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemeleon"},
                "optimization": {"max_epochs": 1},
                "mlflow": {"enabled": False},
            }
        )

        model = ChemeleonModel(config)
        model._setup_trainer()

        # Checkpoint directory should exist and be accessible
        assert model._checkpoint_dir is not None
        from pathlib import Path

        assert Path(model._checkpoint_dir.name).exists()

        # Clean up
        model._checkpoint_dir.cleanup()
