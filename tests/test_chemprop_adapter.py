"""Tests for ChempropModelAdapter."""

from __future__ import annotations

import importlib

import pytest
from omegaconf import OmegaConf

from admet.model.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clear_and_register():
    """Clear registry and re-register chemprop adapter."""
    ModelRegistry.clear()

    # Force re-import to trigger decorator registration
    import admet.model.chemprop.adapter

    importlib.reload(admet.model.chemprop.adapter)

    yield
    ModelRegistry.clear()


# Import after fixture is defined to avoid registration before clear
from admet.model.chemprop.adapter import ChempropModelAdapter  # noqa: E402


class TestChempropModelAdapter:
    """Tests for ChempropModelAdapter."""

    def test_adapter_registration(self):
        """Test that adapter is registered with ModelRegistry."""
        assert ModelRegistry.is_registered("chemprop")
        # Check class name rather than identity (reload changes class identity)
        assert ModelRegistry.get("chemprop").__name__ == "ChempropModelAdapter"

    def test_from_config(self):
        """Test creating adapter from config."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "chemprop",
                    "chemprop": {
                        "depth": 3,
                        "hidden_dim": 256,
                    },
                },
                "data": {
                    "smiles_col": "smiles",
                    "target_cols": ["target1"],
                },
                "mlflow": {"tracking": False},
            }
        )

        model = ChempropModelAdapter.from_config(config)

        assert model.model_type == "chemprop"
        assert model.config == config
        assert model.is_fitted is False

    def test_registry_create(self):
        """Test creating adapter via ModelRegistry.create()."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "chemprop",
                    "chemprop": {
                        "depth": 3,
                    },
                },
                "data": {
                    "smiles_col": "smiles",
                    "target_cols": ["target1"],
                },
                "mlflow": {"tracking": False},
            }
        )

        model = ModelRegistry.create(config)

        # Check class name rather than identity (reload changes class identity)
        assert model.__class__.__name__ == "ChempropModelAdapter"
        assert model.model_type == "chemprop"

    def test_legacy_config_structure(self):
        """Test adapter works with legacy config (no model.type)."""
        # Legacy config has model params directly under model section
        config = OmegaConf.create(
            {
                "model": {
                    "depth": 3,
                    "hidden_dim": 256,
                },
                "data": {
                    "smiles_col": "smiles",
                    "target_cols": ["target1"],
                },
                "optimization": {
                    "max_epochs": 10,
                    "batch_size": 16,
                },
                "mlflow": {"tracking": False},
            }
        )

        model = ChempropModelAdapter(config)
        model_config = model._get_model_config()

        # Should extract params correctly from legacy structure
        assert model_config.get("depth") == 3
        assert model_config.get("hidden_dim") == 256

    def test_get_params(self):
        """Test get_params returns config."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemprop"},
                "mlflow": {"tracking": False},
            }
        )

        model = ChempropModelAdapter(config)
        params = model.get_params()

        assert "config" in params
        assert params["config"] == config

    def test_repr(self):
        """Test string representation."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemprop"},
                "mlflow": {"tracking": False},
            }
        )

        model = ChempropModelAdapter(config)
        repr_str = repr(model)

        assert "ChempropModelAdapter" in repr_str
        assert "fitted=False" in repr_str

    def test_predict_without_fit_raises(self):
        """Test that predict raises if not fitted."""
        config = OmegaConf.create(
            {
                "model": {"type": "chemprop"},
                "mlflow": {"tracking": False},
            }
        )

        model = ChempropModelAdapter(config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(["CCO"])
