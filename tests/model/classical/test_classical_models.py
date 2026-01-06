"""Tests for classical ML models (XGBoost, LightGBM, CatBoost)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from omegaconf import OmegaConf

from admet.model.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clear_and_register():
    """Clear registry and re-register classical models."""
    ModelRegistry.clear()

    # Force re-import to trigger decorator registrations
    import admet.model.classical.catboost_model
    import admet.model.classical.lightgbm_model
    import admet.model.classical.xgboost_model

    importlib.reload(admet.model.classical.xgboost_model)
    importlib.reload(admet.model.classical.lightgbm_model)
    importlib.reload(admet.model.classical.catboost_model)

    yield
    ModelRegistry.clear()


from admet.model.classical import CatBoostModel, ClassicalModelBase, LightGBMModel, XGBoostModel  # noqa: E402


@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return [
        "CCO",
        "CCCO",
        "CCCCO",
        "c1ccccc1",
        "CC(=O)O",
        "CC(C)O",
        "CCN",
        "CCC",
    ]


@pytest.fixture
def sample_targets():
    """Sample target values for testing."""
    return np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0], [3.5, 4.5], [4.0, 5.0], [4.5, 5.5]])


@pytest.fixture
def xgboost_config():
    """XGBoost configuration."""
    return OmegaConf.create(
        {
            "model": {
                "type": "xgboost",
                "xgboost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                },
                "fingerprint": {"type": "morgan"},
            },
            "mlflow": {"enabled": False},
        }
    )


@pytest.fixture
def lightgbm_config():
    """LightGBM configuration."""
    return OmegaConf.create(
        {
            "model": {
                "type": "lightgbm",
                "lightgbm": {
                    "n_estimators": 10,
                    "num_leaves": 15,
                    "learning_rate": 0.1,
                },
                "fingerprint": {"type": "morgan"},
            },
            "mlflow": {"enabled": False},
        }
    )


@pytest.fixture
def catboost_config():
    """CatBoost configuration."""
    return OmegaConf.create(
        {
            "model": {
                "type": "catboost",
                "catboost": {
                    "iterations": 10,
                    "depth": 3,
                    "learning_rate": 0.1,
                },
                "fingerprint": {"type": "morgan"},
            },
            "mlflow": {"enabled": False},
        }
    )


class TestModelRegistry:
    """Tests for classical model registration."""

    def test_xgboost_registered(self):
        """Test XGBoost is registered."""
        assert ModelRegistry.is_registered("xgboost")
        assert ModelRegistry.get("xgboost").__name__ == "XGBoostModel"

    def test_lightgbm_registered(self):
        """Test LightGBM is registered."""
        assert ModelRegistry.is_registered("lightgbm")
        assert ModelRegistry.get("lightgbm").__name__ == "LightGBMModel"

    def test_catboost_registered(self):
        """Test CatBoost is registered."""
        assert ModelRegistry.is_registered("catboost")
        assert ModelRegistry.get("catboost").__name__ == "CatBoostModel"


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_init(self, xgboost_config):
        """Test model initialization."""
        model = XGBoostModel(xgboost_config)

        assert model.model_type == "xgboost"
        assert not model.is_fitted

    def test_from_config(self, xgboost_config):
        """Test creating from config."""
        model = XGBoostModel.from_config(xgboost_config)

        assert model.model_type == "xgboost"

    def test_registry_create(self, xgboost_config):
        """Test creating via registry."""
        model = ModelRegistry.create(xgboost_config)

        assert model.__class__.__name__ == "XGBoostModel"

    def test_fit_predict(self, xgboost_config, sample_smiles, sample_targets):
        """Test fit and predict."""
        model = XGBoostModel(xgboost_config)

        model.fit(sample_smiles, sample_targets)
        assert model.is_fitted

        predictions = model.predict(sample_smiles)
        assert predictions.shape == sample_targets.shape

    def test_predict_without_fit_raises(self, xgboost_config):
        """Test predict raises if not fitted."""
        model = XGBoostModel(xgboost_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(["CCO"])

    def test_get_params(self, xgboost_config):
        """Test get_params returns model parameters."""
        model = XGBoostModel(xgboost_config)
        params = model.get_params()

        assert params["model_type"] == "xgboost"
        assert "n_estimators" in params


class TestLightGBMModel:
    """Tests for LightGBMModel."""

    def test_init(self, lightgbm_config):
        """Test model initialization."""
        model = LightGBMModel(lightgbm_config)

        assert model.model_type == "lightgbm"
        assert not model.is_fitted

    def test_from_config(self, lightgbm_config):
        """Test creating from config."""
        model = LightGBMModel.from_config(lightgbm_config)

        assert model.model_type == "lightgbm"

    def test_registry_create(self, lightgbm_config):
        """Test creating via registry."""
        model = ModelRegistry.create(lightgbm_config)

        assert model.__class__.__name__ == "LightGBMModel"

    def test_fit_predict(self, lightgbm_config, sample_smiles, sample_targets):
        """Test fit and predict."""
        model = LightGBMModel(lightgbm_config)

        model.fit(sample_smiles, sample_targets)
        assert model.is_fitted

        predictions = model.predict(sample_smiles)
        assert predictions.shape == sample_targets.shape

    def test_predict_without_fit_raises(self, lightgbm_config):
        """Test predict raises if not fitted."""
        model = LightGBMModel(lightgbm_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(["CCO"])


class TestCatBoostModel:
    """Tests for CatBoostModel."""

    def test_init(self, catboost_config):
        """Test model initialization."""
        model = CatBoostModel(catboost_config)

        assert model.model_type == "catboost"
        assert not model.is_fitted

    def test_from_config(self, catboost_config):
        """Test creating from config."""
        model = CatBoostModel.from_config(catboost_config)

        assert model.model_type == "catboost"

    def test_registry_create(self, catboost_config):
        """Test creating via registry."""
        model = ModelRegistry.create(catboost_config)

        assert model.__class__.__name__ == "CatBoostModel"

    def test_fit_predict(self, catboost_config, sample_smiles, sample_targets):
        """Test fit and predict."""
        model = CatBoostModel(catboost_config)

        model.fit(sample_smiles, sample_targets)
        assert model.is_fitted

        predictions = model.predict(sample_smiles)
        assert predictions.shape == sample_targets.shape

    def test_predict_without_fit_raises(self, catboost_config):
        """Test predict raises if not fitted."""
        model = CatBoostModel(catboost_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(["CCO"])


class TestClassicalModelBase:
    """Tests for ClassicalModelBase functionality."""

    def test_fingerprint_config(self, xgboost_config):
        """Test fingerprint configuration parsing."""
        model = XGBoostModel(xgboost_config)

        assert model._fingerprint_config.type == "morgan"

    def test_different_fingerprint_types(self):
        """Test model with different fingerprint types."""
        for fp_type in ["morgan", "rdkit", "maccs"]:
            config = OmegaConf.create(
                {
                    "model": {
                        "type": "xgboost",
                        "xgboost": {"n_estimators": 5},
                        "fingerprint": {"type": fp_type},
                    },
                    "mlflow": {"enabled": False},
                }
            )

            model = XGBoostModel(config)
            assert model._fingerprint_config.type == fp_type

    def test_single_target(self, xgboost_config, sample_smiles):
        """Test model with single target."""
        model = XGBoostModel(xgboost_config)
        targets = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

        model.fit(sample_smiles, targets)
        predictions = model.predict(sample_smiles)

        assert predictions.shape[0] == len(sample_smiles)

    def test_set_params(self, xgboost_config):
        """Test setting model parameters."""
        model = XGBoostModel(xgboost_config)

        original_n_estimators = model._model_params.get("n_estimators")
        model.set_params(n_estimators=50)

        assert model._model_params.get("n_estimators") == 50
        assert original_n_estimators != 50

    def test_is_base_model_subclass(self, xgboost_config):
        """Test that classical models inherit from BaseModel."""
        from admet.model.base import BaseModel

        model = XGBoostModel(xgboost_config)
        assert isinstance(model, BaseModel)
        assert isinstance(model, ClassicalModelBase)
